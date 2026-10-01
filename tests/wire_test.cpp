/** @file
 * @brief Conversion between wire messages and engine types, checked against llamad.proto itself.
 *
 * Each message wire.cpp converts is checked two ways. Values: every field set to something
 * distinctive comes out the other side unchanged. Coverage: the message's descriptor lists exactly
 * the fields the test knows the conversion carries, so a field added to llamad.proto fails here,
 * by name, until wire.cpp and this test both handle it. No model and no daemon.
 */

#include "check.h"
#include "wire.h"

#include <google/protobuf/descriptor.h>

#include <algorithm>
#include <string>
#include <vector>

namespace {

namespace v1   = llamad::v1;
namespace wire = llamad::wire;

// The descriptor's field names, sorted, against the ones the conversion carries.
template <typename Message>
void check_fields(std::vector<std::string> carried) {
    const google::protobuf::Descriptor * descriptor = Message::descriptor();

    std::vector<std::string> declared;
    for (int i = 0; i < descriptor->field_count(); ++i) {
        declared.emplace_back(descriptor->field(i)->name());
    }
    std::sort(declared.begin(), declared.end());
    std::sort(carried.begin(), carried.end());

    ++tests::checks;
    if (declared != carried) {
        ++tests::failures;
        std::fprintf(stderr, "%s: the fields in llamad.proto are not the ones wire.cpp converts\n",
                     std::string(descriptor->full_name()).c_str());
        for (const std::string & name : declared) {
            if (!std::binary_search(carried.begin(), carried.end(), name)) {
                std::fprintf(stderr, "    not converted: %s\n", name.c_str());
            }
        }
        for (const std::string & name : carried) {
            if (!std::binary_search(declared.begin(), declared.end(), name)) {
                std::fprintf(stderr, "    not in the message: %s\n", name.c_str());
            }
        }
    }
}

void test_field_coverage() {
    check_fields<v1::SamplingParams>({"temperature", "top_k", "top_p", "min_p", "seed", "max_tokens", "stop"});
    check_fields<v1::ChatMessage>({"role", "content", "tool_calls", "tool_call_id"});
    check_fields<v1::Tool>({"name", "description", "parameters_json_schema"});
    check_fields<v1::ToolCall>({"id", "name", "arguments_json"});
    check_fields<v1::ModelInfo>({"description", "n_params", "size_bytes", "n_ctx", "n_ctx_train",
                                 "has_chat_template", "n_embd", "serves_embeddings"});
    check_fields<v1::GenerateStats>(
        {"prompt_tokens", "completion_tokens", "prompt_ms", "completion_ms", "cached_prompt_tokens"});
    check_fields<v1::Finish>({"reason", "stats", "tool_calls"});
    check_fields<v1::GenerateChunk>({"text", "finish"});
    check_fields<v1::EmbedResponse>({"embeddings", "input_tokens"});
    check_fields<v1::Embedding>({"values"});

    // The engine's FinishReason has a counterpart for every wire reason but TOOL_CALLS, which
    // finish_chunk derives, and UNSPECIFIED, which is never sent.
    CHECK_EQ(v1::FinishReason_descriptor()->value_count(), 6);
}

void test_unset_sampling_keeps_engine_defaults() {
    const llamad::SamplingParams defaults;
    const llamad::SamplingParams params = wire::from_proto(v1::SamplingParams{});

    CHECK(params.temperature == defaults.temperature);
    CHECK(params.top_k == defaults.top_k);
    CHECK(params.top_p == defaults.top_p);
    CHECK(params.min_p == defaults.min_p);
    CHECK(!params.seed);
    CHECK(params.max_tokens == defaults.max_tokens);
    CHECK(params.stop.empty());
}

void test_set_sampling_overrides() {
    v1::SamplingParams in;
    in.set_temperature(0.0f);  // set to the proto default, which still counts as set
    in.set_top_k(7);
    in.set_top_p(0.5f);
    in.set_min_p(0.25f);
    in.set_seed(42);
    in.set_max_tokens(0);
    in.add_stop("</s>");
    in.add_stop("END");

    const llamad::SamplingParams params = wire::from_proto(in);
    CHECK(params.temperature == 0.0f);
    CHECK(params.top_k == 7);
    CHECK(params.top_p == 0.5f);
    CHECK(params.min_p == 0.25f);
    CHECK(params.seed && *params.seed == 42u);
    CHECK(params.max_tokens == 0);
    CHECK_EQ(params.stop.size(), size_t(2));
    CHECK_EQ(params.stop[1], "END");
}

void test_chat_message() {
    v1::ChatMessage in;
    in.set_role("assistant");
    in.set_content("checking");
    in.set_tool_call_id("call-9");
    v1::ToolCall * call = in.add_tool_calls();
    call->set_id("call-1");
    call->set_name("get_time");
    call->set_arguments_json(R"({"tz":"UTC"})");

    const llamad::ChatMessage message = wire::from_proto(in);
    CHECK_EQ(message.role, "assistant");
    CHECK_EQ(message.content, "checking");
    CHECK_EQ(message.tool_call_id, "call-9");
    CHECK_EQ(message.tool_calls.size(), size_t(1));
    CHECK_EQ(message.tool_calls[0].id, "call-1");
    CHECK_EQ(message.tool_calls[0].name, "get_time");
    CHECK_EQ(message.tool_calls[0].arguments_json, R"({"tz":"UTC"})");
}

void test_tool() {
    v1::Tool in;
    in.set_name("search");
    in.set_description("Search the web");
    in.set_parameters_json_schema(R"({"type":"object"})");

    const llamad::Tool tool = wire::from_proto(in);
    CHECK_EQ(tool.name, "search");
    CHECK_EQ(tool.description, "Search the web");
    CHECK_EQ(tool.parameters_json_schema, R"({"type":"object"})");
}

void test_model_info() {
    llamad::ModelInfo info;
    info.description       = "qwen2 0.5B Q4_K - Medium";
    info.n_params          = 494032768;
    info.size_bytes        = 397808192;
    info.n_ctx             = 4096;
    info.n_ctx_train       = 32768;
    info.has_chat_template = true;
    info.n_embd            = 896;
    info.serves_embeddings = true;

    v1::ModelInfo out;
    wire::to_proto(info, &out);
    CHECK_EQ(out.description(), "qwen2 0.5B Q4_K - Medium");
    CHECK(out.n_params() == 494032768u);
    CHECK(out.size_bytes() == 397808192u);
    CHECK(out.n_ctx() == 4096u);
    CHECK(out.n_ctx_train() == 32768u);
    CHECK(out.has_chat_template());
    CHECK(out.n_embd() == 896u);
    CHECK(out.serves_embeddings());
}

void test_embed_response() {
    llamad::EmbedResult result;
    result.embeddings   = {{{0.6f, 0.8f}}, {{1.0f, 0.0f}}};
    result.input_tokens = 11;

    v1::EmbedResponse out;
    wire::to_proto(result, &out);
    CHECK_EQ(out.embeddings_size(), 2);
    CHECK_EQ(out.embeddings(0).values_size(), 2);
    CHECK(out.embeddings(0).values(1) == 0.8f);
    CHECK(out.embeddings(1).values(0) == 1.0f);
    CHECK_EQ(out.input_tokens(), 11);
}

std::vector<llamad::ToolCall> one_call() {
    return {{"call-1", "get_time", R"({"tz":"UTC"})"}};
}

void test_finish_carries_reason_and_stats() {
    llamad::GenerateStats stats;
    stats.prompt_tokens        = 12;
    stats.completion_tokens    = 5;
    stats.prompt_ms            = 1.5;
    stats.completion_ms        = 2.5;
    stats.cached_prompt_tokens = 4;

    const v1::GenerateChunk chunk = wire::finish_chunk(llamad::FinishReason::Stop, stats, {});
    CHECK(chunk.chunk_case() == v1::GenerateChunk::kFinish);
    CHECK(chunk.finish().reason() == v1::FINISH_REASON_STOP);
    CHECK_EQ(chunk.finish().tool_calls_size(), 0);
    CHECK_EQ(chunk.finish().stats().prompt_tokens(), 12);
    CHECK_EQ(chunk.finish().stats().completion_tokens(), 5);
    CHECK(chunk.finish().stats().prompt_ms() == 1.5);
    CHECK(chunk.finish().stats().completion_ms() == 2.5);
    CHECK_EQ(chunk.finish().stats().cached_prompt_tokens(), 4);

    CHECK(wire::finish_chunk(llamad::FinishReason::Eog, stats, {}).finish().reason() == v1::FINISH_REASON_EOG);
    CHECK(wire::finish_chunk(llamad::FinishReason::Length, stats, {}).finish().reason() == v1::FINISH_REASON_LENGTH);
    CHECK(wire::finish_chunk(llamad::FinishReason::Cancelled, stats, {}).finish().reason() ==
          v1::FINISH_REASON_CANCELLED);
}

void test_tool_calls_only_when_generation_completed() {
    const llamad::GenerateStats stats;

    // A completed reply that called a tool says so, and carries the call whole.
    for (const llamad::FinishReason reason : {llamad::FinishReason::Eog, llamad::FinishReason::Stop}) {
        const v1::GenerateChunk chunk = wire::finish_chunk(reason, stats, one_call());
        CHECK(chunk.finish().reason() == v1::FINISH_REASON_TOOL_CALLS);
        CHECK_EQ(chunk.finish().tool_calls_size(), 1);
        CHECK_EQ(chunk.finish().tool_calls(0).id(), "call-1");
        CHECK_EQ(chunk.finish().tool_calls(0).name(), "get_time");
        CHECK_EQ(chunk.finish().tool_calls(0).arguments_json(), R"({"tz":"UTC"})");
    }

    // One cut short by the token limit or a cancel keeps its reason and drops what was parsed.
    for (const llamad::FinishReason reason : {llamad::FinishReason::Length, llamad::FinishReason::Cancelled}) {
        const v1::GenerateChunk chunk = wire::finish_chunk(reason, stats, one_call());
        CHECK(chunk.finish().reason() != v1::FINISH_REASON_TOOL_CALLS);
        CHECK_EQ(chunk.finish().tool_calls_size(), 0);
    }
}

}  // namespace

int main() {
    test_field_coverage();
    test_unset_sampling_keeps_engine_defaults();
    test_set_sampling_overrides();
    test_chat_message();
    test_tool();
    test_model_info();
    test_embed_response();
    test_finish_carries_reason_and_stats();
    test_tool_calls_only_when_generation_completed();

    return tests::report();
}
