#include "check.h"
#include "playground.h"

#include <cstdlib>
#include <filesystem>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>

namespace {

namespace v1 = llamad::v1;
using Writer = grpc::ServerWriter<v1::GenerateChunk>;

void text(Writer & writer, const std::string & value) {
    v1::GenerateChunk chunk;
    chunk.set_text(value);
    CHECK(writer.Write(chunk));
}

v1::GenerateChunk finish() {
    v1::GenerateChunk chunk;
    auto * result = chunk.mutable_finish();
    result->set_reason(v1::FINISH_REASON_EOG);
    auto * stats = result->mutable_stats();
    stats->set_prompt_tokens(11);
    stats->set_cached_prompt_tokens(3);
    stats->set_completion_tokens(5);
    stats->set_prompt_ms(2);
    stats->set_completion_ms(1000);
    return chunk;
}

class ScriptedService final : public v1::Llama::Service {
public:
    bool embeddings = false;
    std::vector<v1::ChatRequest> requests;
    std::function<grpc::Status(const v1::ChatRequest &, Writer &, size_t)> respond;

    grpc::Status GetModelInfo(grpc::ServerContext *, const v1::GetModelInfoRequest *, v1::ModelInfo * model) override {
        model->set_description("test model");
        model->set_n_ctx(4096);
        model->set_has_chat_template(true);
        model->set_serves_embeddings(embeddings);
        return grpc::Status::OK;
    }

    grpc::Status Chat(grpc::ServerContext *, const v1::ChatRequest * request, Writer * writer) override {
        requests.push_back(*request);
        return respond(*request, *writer, requests.size() - 1);
    }
};

std::string run(ScriptedService & service, const std::string & input, int expected_status) {
    char directory[] = "/tmp/llamad-playground-test-XXXXXX";
    if (mkdtemp(directory) == nullptr) {
        throw std::runtime_error("cannot create test directory");
    }
    const std::string socket = std::string(directory) + "/socket";
    grpc::ServerBuilder builder;
    builder.AddListeningPort("unix:" + socket, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    if (!server) {
        throw std::runtime_error("cannot start test server");
    }
    auto stub = v1::Llama::NewStub(grpc::CreateChannel("unix:" + socket, grpc::InsecureChannelCredentials()));
    std::istringstream in(input);
    std::ostringstream out;
    CHECK_EQ(run_playground(*stub, in, out), expected_status);
    server->Shutdown();
    server->Wait();
    std::filesystem::remove(socket);
    std::filesystem::remove(directory);
    return out.str();
}

void test_history_and_clear() {
    ScriptedService service;
    service.respond = [](const v1::ChatRequest & request, Writer & writer, size_t turn) {
        CHECK_EQ(request.sampling().temperature(), 0);
        CHECK_EQ(request.sampling().max_tokens(), 512);
        CHECK_EQ(request.tools_size(), 0);
        CHECK(request.response_json_schema().empty());
        if (turn == 1) {
            CHECK_EQ(request.messages_size(), 4);
            CHECK_EQ(request.messages(2).role(), "assistant");
            CHECK_EQ(request.messages(2).content(), "Hi there.");
            CHECK_EQ(request.messages(3).content(), "again");
        } else {
            CHECK_EQ(request.messages_size(), 2);
        }
        text(writer, "Hi ");
        text(writer, "there.");
        CHECK(writer.Write(finish()));
        return grpc::Status::OK;
    };
    const auto output = run(service, "hello\nagain\n/json x\n/clear\nhello\n/quit\n", 0);
    CHECK_EQ(service.requests.size(), 3u);
    CHECK(output.find("Use /help, /clear or /quit.") != std::string::npos);
    CHECK(output.find("assistant> Hi there.\n") != std::string::npos);
    CHECK(output.find("[finish EOG | prompt 11 (cached 3) | output 5 | prefill 2.0 ms | decode 5.0 tok/s]") !=
          std::string::npos);
}

void test_failed_rpc_does_not_commit_history() {
    ScriptedService service;
    service.respond = [](const v1::ChatRequest & request, Writer & writer, size_t turn) {
        if (turn == 0) {
            text(writer, "Partial reply");
            return grpc::Status(grpc::StatusCode::INTERNAL, "test failure");
        }
        CHECK_EQ(request.messages_size(), 2);
        CHECK_EQ(request.messages(1).content(), "retry");
        CHECK(writer.Write(finish()));
        return grpc::Status::OK;
    };
    const auto output = run(service, "fail\nretry\n", 1);
    CHECK_EQ(service.requests.size(), 2u);
    CHECK(output.find("test failure") != std::string::npos);
}

void test_missing_finish() {
    ScriptedService service;
    service.respond = [](const v1::ChatRequest &, Writer & writer, size_t) {
        text(writer, "partial");
        return grpc::Status::OK;
    };
    const auto output = run(service, "hello\n", 1);
    CHECK(output.find("without a finish chunk") != std::string::npos);
}

void test_embedding_model_is_rejected() {
    ScriptedService service;
    service.embeddings = true;
    const auto output = run(service, "hello\n", 1);
    CHECK(service.requests.empty());
    CHECK(output.find("needs a chat model") != std::string::npos);
}

}  // namespace

int main() {
    test_history_and_clear();
    test_failed_rpc_does_not_commit_history();
    test_missing_finish();
    test_embedding_model_is_rejected();
    return tests::report();
}
