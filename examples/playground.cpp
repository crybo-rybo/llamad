#include "playground.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

namespace v1 = llamad::v1;

constexpr auto kRequestTimeout = std::chrono::minutes(2);

void check_status(const grpc::Status & status) {
    if (!status.ok()) {
        throw std::runtime_error("RPC " + std::to_string(status.error_code()) + ": " + status.error_message());
    }
}

v1::ChatRequest chat_request() {
    v1::ChatRequest request;
    request.mutable_sampling()->set_temperature(0);
    request.mutable_sampling()->set_max_tokens(512);
    auto * system = request.add_messages();
    system->set_role("system");
    system->set_content("You are a concise, friendly assistant in the llamad terminal playground.");
    return request;
}

void print_help(std::ostream & output) {
    output << "\nType a message to chat with the model.\n"
           << "/clear  Reset chat history.\n"
           << "/help   Show this help.\n"
           << "/quit   Exit (or press Ctrl+D).\n\n";
}

void print_stats(const v1::Finish & finish, std::ostream & output) {
    constexpr std::string_view prefix = "FINISH_REASON_";
    std::string reason(v1::FinishReason_Name(finish.reason()));
    if (reason.starts_with(prefix)) {
        reason.erase(0, prefix.size());
    }
    const auto & stats = finish.stats();
    output << "[finish " << reason
           << " | prompt " << stats.prompt_tokens() << " (cached " << stats.cached_prompt_tokens() << ")"
           << " | output " << stats.completion_tokens()
           << " | prefill " << std::fixed << std::setprecision(1) << stats.prompt_ms() << " ms";
    if (stats.completion_ms() > 0) {
        output << " | decode " << 1000 * stats.completion_tokens() / stats.completion_ms() << " tok/s";
    }
    output << "]\n";
}

void chat(v1::Llama::StubInterface & stub, v1::ChatRequest & request, std::ostream & output) {
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + kRequestTimeout);
    output << "assistant> " << std::flush;
    auto reader = stub.Chat(&context, request);
    v1::GenerateChunk chunk;
    std::string text;
    std::optional<v1::Finish> finish;
    while (reader->Read(&chunk)) {
        if (chunk.has_text()) {
            text += chunk.text();
            output << chunk.text() << std::flush;
        } else if (chunk.has_finish()) {
            finish = chunk.finish();
        }
    }
    output << '\n';
    check_status(reader->Finish());
    if (!finish) {
        throw std::runtime_error("stream ended without a finish chunk");
    }
    print_stats(*finish, output);

    auto * assistant = request.add_messages();
    assistant->set_role("assistant");
    assistant->set_content(text);
}

}  // namespace

int run_playground(v1::Llama::StubInterface & stub, std::istream & input, std::ostream & output) {
    try {
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        v1::ModelInfo model;
        check_status(stub.GetModelInfo(&context, v1::GetModelInfoRequest{}, &model));
        if (model.serves_embeddings()) {
            throw std::runtime_error("the playground needs a chat model, not an embedding model");
        }
        if (!model.has_chat_template()) {
            throw std::runtime_error("the playground needs a model with a chat template");
        }
        output << "llamad playground | " << model.description() << " | context " << model.n_ctx() << " tokens\n";
    } catch (const std::exception & e) {
        output << "error> " << e.what() << "\nStart llamad with a chat model and check --socket.\n";
        return 1;
    }

    print_help(output);
    auto history = chat_request();
    bool failed = false;
    std::string line;
    while (output << "you> " << std::flush, std::getline(input, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) {
            continue;
        }
        if (line == "/quit") {
            break;
        }
        if (line == "/help") {
            print_help(output);
            continue;
        }
        if (line == "/clear") {
            history = chat_request();
            output << "Chat history cleared.\n";
            continue;
        }
        if (line.starts_with('/')) {
            output << "Use /help, /clear or /quit.\n";
            continue;
        }
        try {
            // Work on a copy so a failed stream cannot leave partial history behind.
            auto request = history;
            auto * user = request.add_messages();
            user->set_role("user");
            user->set_content(line);
            chat(stub, request, output);
            history = std::move(request);
        } catch (const std::exception & e) {
            output << "error> " << e.what() << "\nUse /clear if the conversation exceeds the model's context.\n";
            failed = true;
        }
    }
    output << '\n';
    return failed ? 1 : 0;
}
