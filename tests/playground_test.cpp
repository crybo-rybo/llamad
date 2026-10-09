#include "check.h"
#include "playground.h"

#include <cstdlib>
#include <filesystem>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>
#include <grpcpp/grpcpp.h>

namespace {

namespace v1 = llamad::v1;
using Writer = grpc::ServerWriter<v1::GenerateChunk>;

void text(Writer & writer, const std::string & value) {
    v1::GenerateChunk chunk;
    chunk.set_text(value);
    CHECK(writer.Write(chunk));
}

v1::GenerateChunk finish(v1::FinishReason reason = v1::FINISH_REASON_EOG) {
    v1::GenerateChunk chunk;
    auto * result = chunk.mutable_finish();
    result->set_reason(reason);
    auto * stats = result->mutable_stats();
    stats->set_prompt_tokens(11);
    stats->set_cached_prompt_tokens(3);
    stats->set_completion_tokens(5);
    stats->set_prompt_ms(2);
    stats->set_completion_ms(1000);
    return chunk;
}

void add_call(v1::GenerateChunk & chunk, const char * id, const char * name, const char * args) {
    auto * call = chunk.mutable_finish()->add_tool_calls();
    call->set_id(id);
    call->set_name(name);
    call->set_arguments_json(args);
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

void test_tools_history_json_and_clear() {
    ScriptedService service;
    service.respond = [](const v1::ChatRequest & request, Writer & writer, size_t turn) {
        CHECK_EQ(request.sampling().temperature(), 0);
        CHECK_EQ(request.sampling().max_tokens(), 512);
        if (turn == 0) {
            CHECK_EQ(request.messages_size(), 2);
            CHECK_EQ(request.tools_size(), 3);
            text(writer, "Let me check.");
            auto chunk = finish(v1::FINISH_REASON_TOOL_CALLS);
            add_call(chunk, "clock", "get_time", "{}");
            add_call(chunk, "dice", "roll_dice", R"({"count":2,"sides":6})");
            add_call(chunk, "coin", "flip_coin", "{}");
            CHECK(writer.Write(chunk));
        } else {
            if (turn == 1) {
                CHECK_EQ(request.messages_size(), 6);
                CHECK_EQ(request.messages(2).content(), "Let me check.");
                CHECK_EQ(request.messages(2).tool_calls_size(), 3);
                for (int i = 0; i < 3; ++i) {
                    const auto & result = request.messages(3 + i);
                    CHECK_EQ(result.role(), "tool");
                    CHECK_EQ(result.tool_call_id(), request.messages(2).tool_calls(i).id());
                    google::protobuf::Struct object;
                    CHECK(google::protobuf::util::JsonStringToMessage(result.content(), &object).ok());
                    CHECK(object.fields().find("error") == object.fields().end());
                    if (i == 0) {
                        CHECK(!object.fields().at("local").string_value().empty());
                        CHECK(object.fields().at("utc").string_value().ends_with('Z'));
                    } else if (i == 1) {
                        const auto & rolls = object.fields().at("rolls").list_value();
                        CHECK_EQ(rolls.values_size(), 2);
                        double total = 0;
                        for (const auto & roll : rolls.values()) {
                            CHECK(roll.number_value() >= 1 && roll.number_value() <= 6);
                            total += roll.number_value();
                        }
                        CHECK_EQ(total, object.fields().at("total").number_value());
                    } else {
                        const auto & coin = object.fields().at("result").string_value();
                        CHECK(coin == "heads" || coin == "tails");
                    }
                }
            } else if (turn == 2) {
                CHECK_EQ(request.messages_size(), 8);
                CHECK_EQ(request.messages(6).content(), "A fun result.");
            } else if (turn == 3) {
                CHECK_EQ(request.messages_size(), 2);
                CHECK_EQ(request.tools_size(), 0);
                CHECK(!request.response_json_schema().empty());
                text(writer, R"({"title":"Pirates","objective":"Find treasure","reward":"Gold"})");
                CHECK(writer.Write(finish()));
                return grpc::Status::OK;
            } else if (turn == 4) {
                CHECK_EQ(request.messages_size(), 10);
                CHECK(request.response_json_schema().empty());
                CHECK_EQ(request.tools_size(), 3);
            } else if (turn == 5) {
                CHECK_EQ(request.messages_size(), 2);
            }
            text(writer, "A fun ");
            text(writer, "result.");
            CHECK(writer.Write(finish()));
        }
        return grpc::Status::OK;
    };
    const auto output = run(service, "time and dice\nremember\n/json A pirate quest\nremember again\n/clear\nhello\n/quit\n", 0);
    CHECK_EQ(service.requests.size(), 6u);
    CHECK(output.find("[tool] get_time({}) ->") != std::string::npos);
    CHECK(output.find("[tool] roll_dice(") != std::string::npos);
    CHECK(output.find("[tool] flip_coin({}) ->") != std::string::npos);
    CHECK(output.find("assistant> A fun result.") != std::string::npos);
    CHECK(output.find("prompt 11 (cached 3) | output 5 | prefill 2.0 ms | decode 5.0 tok/s") != std::string::npos);
}

void test_tool_errors_are_returned_to_the_model() {
    ScriptedService service;
    service.respond = [](const v1::ChatRequest & request, Writer & writer, size_t turn) {
        if (turn == 0) {
            auto chunk = finish(v1::FINISH_REASON_TOOL_CALLS);
            add_call(chunk, "a", "roll_dice", R"({"count":100,"sides":6})");
            add_call(chunk, "b", "roll_dice", R"({"count":1.5,"sides":6})");
            add_call(chunk, "c", "roll_dice", R"({"count":"two","sides":6})");
            add_call(chunk, "d", "unknown", "{}");
            add_call(chunk, "e", "flip_coin", "not JSON");
            add_call(chunk, "f", "get_time", R"({"extra":true})");
            CHECK(writer.Write(chunk));
        } else {
            CHECK_EQ(request.messages_size(), 9);
            for (int i = 3; i < request.messages_size(); ++i) {
                CHECK(request.messages(i).content().find("error") != std::string::npos);
            }
            text(writer, "The tools reported errors.");
            CHECK(writer.Write(finish()));
        }
        return grpc::Status::OK;
    };
    run(service, "bad tools\n", 0);
    CHECK_EQ(service.requests.size(), 2u);
}

void test_failed_rpc_does_not_run_tools_or_commit_history() {
    ScriptedService service;
    service.respond = [](const v1::ChatRequest & request, Writer & writer, size_t turn) {
        if (turn == 0) {
            text(writer, "Partial reply");
            auto chunk = finish(v1::FINISH_REASON_TOOL_CALLS);
            add_call(chunk, "clock", "get_time", "{}");
            CHECK(writer.Write(chunk));
            return grpc::Status(grpc::StatusCode::INTERNAL, "test failure");
        }
        CHECK_EQ(request.messages_size(), 2);
        CHECK_EQ(request.messages(1).content(), "retry");
        CHECK(writer.Write(finish()));
        return grpc::Status::OK;
    };
    const auto output = run(service, "fail\nretry\n", 1);
    CHECK(output.find("test failure") != std::string::npos);
    CHECK(output.find("[tool]") == std::string::npos);
}

void test_missing_finish_and_incomplete_json() {
    ScriptedService service;
    service.respond = [](const v1::ChatRequest &, Writer & writer, size_t turn) {
        text(writer, "partial");
        if (turn == 1) {
            CHECK(writer.Write(finish(v1::FINISH_REASON_LENGTH)));
        }
        return grpc::Status::OK;
    };
    const auto output = run(service, "hello\n/json pirate quest\n", 1);
    CHECK(output.find("without a finish chunk") != std::string::npos);
    CHECK(output.find("JSON reply is incomplete") != std::string::npos);
}

void test_tool_round_limit() {
    ScriptedService service;
    service.respond = [](const v1::ChatRequest &, Writer & writer, size_t) {
        auto chunk = finish(v1::FINISH_REASON_TOOL_CALLS);
        add_call(chunk, "coin", "flip_coin", "{}");
        CHECK(writer.Write(chunk));
        return grpc::Status::OK;
    };
    const auto output = run(service, "keep flipping\n", 1);
    CHECK_EQ(service.requests.size(), 5u);
    CHECK(output.find("tool-round limit reached") != std::string::npos);
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
    test_tools_history_json_and_clear();
    test_tool_errors_are_returned_to_the_model();
    test_failed_rpc_does_not_run_tools_or_commit_history();
    test_missing_finish_and_incomplete_json();
    test_tool_round_limit();
    test_embedding_model_is_rejected();
    return tests::report();
}
