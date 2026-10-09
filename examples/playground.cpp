#include "playground.h"

#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>

namespace {

namespace v1 = llamad::v1;
using Object = google::protobuf::Struct;

constexpr int kMaxToolRounds = 4;
constexpr auto kRequestTimeout = std::chrono::minutes(2);
constexpr const char * kQuestSchema = R"({
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "objective": {"type": "string"},
        "reward": {"type": "string"}
    },
    "required": ["title", "objective", "reward"],
    "additionalProperties": false
})";

void check_status(const grpc::Status & status) {
    if (!status.ok()) {
        throw std::runtime_error("RPC " + std::to_string(status.error_code()) + ": " + status.error_message());
    }
}

std::string json(const Object & object) {
    std::string result;
    const auto status = google::protobuf::util::MessageToJsonString(object, &result);
    if (!status.ok()) {
        throw std::runtime_error("cannot encode tool result: " + status.ToString());
    }
    return result;
}

int integer_argument(const Object & args, const char * name, int minimum, int maximum) {
    const auto at = args.fields().find(name);
    if (at == args.fields().end() || at->second.kind_case() != google::protobuf::Value::kNumberValue) {
        throw std::runtime_error(std::string(name) + " must be an integer");
    }
    const double value = at->second.number_value();
    if (!std::isfinite(value) || value != std::floor(value) || value < minimum || value > maximum) {
        throw std::runtime_error(std::string(name) + " must be an integer from " +
                                 std::to_string(minimum) + " to " + std::to_string(maximum));
    }
    return static_cast<int>(value);
}

std::string run_tool(const v1::ToolCall & call, std::mt19937 & random) {
    Object result;
    try {
        Object args;
        if (!google::protobuf::util::JsonStringToMessage(call.arguments_json(), &args).ok()) {
            throw std::runtime_error("arguments must be a JSON object");
        }
        if (call.name() == "get_time") {
            if (!args.fields().empty()) {
                throw std::runtime_error("get_time takes no arguments");
            }
            const std::time_t time = std::time(nullptr);
            std::tm local{}, utc{};
            if (time == static_cast<std::time_t>(-1) || localtime_r(&time, &local) == nullptr ||
                gmtime_r(&time, &utc) == nullptr) {
                throw std::runtime_error("cannot read the clock");
            }
            char local_text[64], utc_text[64];
            std::strftime(local_text, sizeof(local_text), "%Y-%m-%dT%H:%M:%S%z", &local);
            std::strftime(utc_text, sizeof(utc_text), "%Y-%m-%dT%H:%M:%SZ", &utc);
            (*result.mutable_fields())["local"].set_string_value(local_text);
            (*result.mutable_fields())["utc"].set_string_value(utc_text);
        } else if (call.name() == "roll_dice") {
            if (args.fields().size() != 2) {
                throw std::runtime_error("roll_dice needs only count and sides");
            }
            const int count = integer_argument(args, "count", 1, 10);
            const int sides = integer_argument(args, "sides", 2, 100);
            std::uniform_int_distribution<int> roll(1, sides);
            auto * rolls = (*result.mutable_fields())["rolls"].mutable_list_value();
            int total = 0;
            for (int i = 0; i < count; ++i) {
                const int value = roll(random);
                rolls->add_values()->set_number_value(value);
                total += value;
            }
            (*result.mutable_fields())["total"].set_number_value(total);
        } else if (call.name() == "flip_coin") {
            if (!args.fields().empty()) {
                throw std::runtime_error("flip_coin takes no arguments");
            }
            (*result.mutable_fields())["result"].set_string_value(
                std::uniform_int_distribution<int>(0, 1)(random) == 0 ? "heads" : "tails");
        } else {
            throw std::runtime_error("unknown tool: " + call.name());
        }
    } catch (const std::exception & e) {
        result.Clear();
        (*result.mutable_fields())["error"].set_string_value(e.what());
    }
    return json(result);
}

v1::ChatRequest chat_request() {
    v1::ChatRequest request;
    request.mutable_sampling()->set_temperature(0);
    request.mutable_sampling()->set_max_tokens(512);
    auto * system = request.add_messages();
    system->set_role("system");
    system->set_content("You are a concise, playful assistant in the llamad terminal playground. "
                        "When asked the time, call get_time with {}. "
                        "When asked to roll dice, call roll_dice with count and sides. "
                        "When asked to flip a coin, call flip_coin with {}. "
                        "Always use these tools instead of guessing or asking the user for their results. "
                        "After a tool result, answer the user briefly.");
    const auto add_tool = [&](const char * name, const char * description, const char * schema) {
        auto * tool = request.add_tools();
        tool->set_name(name);
        tool->set_description(description);
        tool->set_parameters_json_schema(schema);
    };
    add_tool("get_time", "Get the current local time and UTC time from the computer's clock.",
             R"({"type":"object","properties":{},"additionalProperties":false})");
    add_tool("roll_dice", "Roll count dice, each with sides faces. Use count=1 and sides=6 for one ordinary die.",
             R"({
                 "type": "object",
                 "properties": {
                     "count": {"type": "integer", "minimum": 1, "maximum": 10},
                     "sides": {"type": "integer", "minimum": 2, "maximum": 100}
                 },
                 "required": ["count", "sides"],
                 "additionalProperties": false
             })");
    add_tool("flip_coin", "Flip a random coin and return heads or tails.",
             R"({"type":"object","properties":{},"additionalProperties":false})");
    return request;
}

v1::ChatRequest quest_request(const std::string & prompt) {
    v1::ChatRequest request;
    request.mutable_sampling()->set_temperature(0);
    request.mutable_sampling()->set_max_tokens(512);
    request.set_response_json_schema(kQuestSchema);
    auto * system = request.add_messages();
    system->set_role("system");
    system->set_content("Create a short, playful quest from the user's prompt.");
    auto * user = request.add_messages();
    user->set_role("user");
    user->set_content(prompt);
    return request;
}

void print_help(std::ostream & output) {
    output << "\nType a message to chat. Try:\n"
           << "  What time is it?\n"
           << "  Roll two six-sided dice for my goblin's attack.\n"
           << "  Flip a coin to decide whether we enter the haunted cave.\n\n"
           << "Tools run here in the client. Each call and result appears below the reply.\n"
           << "/json PROMPT  Generate a quest as JSON (title, objective, reward), without tools.\n"
           << "/clear        Reset chat history.\n"
           << "/help         Show this help.\n"
           << "/quit         Exit (or press Ctrl+D).\n\n";
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

void chat(v1::Llama::StubInterface & stub, v1::ChatRequest & request,
          std::mt19937 & random, std::ostream & output) {
    for (int round = 0; round <= kMaxToolRounds; ++round) {
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + kRequestTimeout);
        auto reader = stub.Chat(&context, request);
        v1::GenerateChunk chunk;
        std::string text;
        std::optional<v1::Finish> finish;
        while (reader->Read(&chunk)) {
            if (chunk.has_text()) {
                // Text chunks are never empty, so empty text means the first chunk of this round.
                if (text.empty()) {
                    output << "assistant> ";
                }
                text += chunk.text();
                output << chunk.text() << std::flush;
            } else if (chunk.has_finish()) {
                finish = chunk.finish();
            }
        }
        if (!text.empty()) {
            output << '\n';
        }
        check_status(reader->Finish());
        if (!finish) {
            throw std::runtime_error("stream ended without a finish chunk");
        }
        print_stats(*finish, output);

        auto * assistant = request.add_messages();
        assistant->set_role("assistant");
        assistant->set_content(text);
        *assistant->mutable_tool_calls() = finish->tool_calls();
        if (finish->reason() != v1::FINISH_REASON_TOOL_CALLS) {
            return;
        }
        if (round == kMaxToolRounds) {
            throw std::runtime_error("tool-round limit reached; try a simpler request");
        }
        // Keep the assistant's complete call list, then return one result for each call ID.
        for (const auto & call : finish->tool_calls()) {
            const std::string result = run_tool(call, random);
            output << "[tool] " << call.name() << '(' << call.arguments_json() << ") -> " << result << '\n';
            auto * message = request.add_messages();
            message->set_role("tool");
            message->set_tool_call_id(call.id());
            message->set_content(result);
        }
    }
}

void print_quest(const std::string & reply, std::ostream & output) {
    // The schema constrains the reply, so a parse fails only when max_tokens cuts it off.
    Object quest;
    if (!google::protobuf::util::JsonStringToMessage(reply, &quest).ok()) {
        throw std::runtime_error("JSON reply is incomplete; try a shorter quest");
    }
    const auto field = [&](const char * name) {
        const auto at = quest.fields().find(name);
        return at == quest.fields().end() ? std::string() : at->second.string_value();
    };
    output << "[quest] " << field("title") << '\n'
           << "  objective: " << field("objective") << '\n'
           << "  reward:    " << field("reward") << '\n';
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
    std::mt19937 random(std::random_device{}());
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
        const bool quest = line.starts_with("/json ") && line.find_first_not_of(" \t", 6) != std::string::npos;
        if (line.starts_with('/') && !quest) {
            output << "Use /help, /clear, /quit or /json PROMPT.\n";
            continue;
        }
        try {
            if (quest) {
                auto request = quest_request(line.substr(6));
                chat(stub, request, random, output);
                print_quest(request.messages(request.messages_size() - 1).content(), output);
            } else {
                // Work on a copy so a failed stream or tool loop cannot leave partial history behind.
                auto request = history;
                auto * user = request.add_messages();
                user->set_role("user");
                user->set_content(line);
                chat(stub, request, random, output);
                history = std::move(request);
            }
        } catch (const std::exception & e) {
            output << "error> " << e.what() << "\nUse /clear if the conversation exceeds the model's context.\n";
            failed = true;
        }
    }
    output << '\n';
    return failed ? 1 : 0;
}
