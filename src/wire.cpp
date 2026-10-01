/** @file
 * @brief Field-by-field conversion between wire messages and the engine and chat layer's types.
 */

#include "wire.h"

#include <utility>

namespace llamad::wire {
namespace {

/// The wire spelling of an engine finish reason; TOOL_CALLS is decided by finish_chunk.
v1::FinishReason to_proto(FinishReason reason) {
    switch (reason) {
        case FinishReason::Eog:       return v1::FINISH_REASON_EOG;
        case FinishReason::Length:    return v1::FINISH_REASON_LENGTH;
        case FinishReason::Stop:      return v1::FINISH_REASON_STOP;
        case FinishReason::Cancelled: return v1::FINISH_REASON_CANCELLED;
    }
    return v1::FINISH_REASON_UNSPECIFIED;
}

/// Token counts and timings of one generation.
void to_proto(const GenerateStats & stats, v1::GenerateStats * out) {
    out->set_prompt_tokens(stats.prompt_tokens);
    out->set_completion_tokens(stats.completion_tokens);
    out->set_prompt_ms(stats.prompt_ms);
    out->set_completion_ms(stats.completion_ms);
    out->set_cached_prompt_tokens(stats.cached_prompt_tokens);
}

/// One parsed tool call.
void to_proto(const ToolCall & call, v1::ToolCall * out) {
    out->set_id(call.id);
    out->set_name(call.name);
    out->set_arguments_json(call.arguments_json);
}

}  // namespace

SamplingParams from_proto(const v1::SamplingParams & params) {
    SamplingParams out;
    if (params.has_temperature()) { out.temperature = params.temperature(); }
    if (params.has_top_k())       { out.top_k       = params.top_k(); }
    if (params.has_top_p())       { out.top_p       = params.top_p(); }
    if (params.has_min_p())       { out.min_p       = params.min_p(); }
    if (params.has_seed())        { out.seed        = params.seed(); }
    if (params.has_max_tokens())  { out.max_tokens  = params.max_tokens(); }
    out.stop.assign(params.stop().begin(), params.stop().end());
    return out;
}

ChatMessage from_proto(const v1::ChatMessage & message) {
    ChatMessage out;
    out.role         = message.role();
    out.content      = message.content();
    out.tool_call_id = message.tool_call_id();
    out.tool_calls.reserve(static_cast<size_t>(message.tool_calls_size()));
    for (const v1::ToolCall & call : message.tool_calls()) {
        out.tool_calls.push_back(from_proto(call));
    }
    return out;
}

Tool from_proto(const v1::Tool & tool) {
    return {tool.name(), tool.description(), tool.parameters_json_schema()};
}

ToolCall from_proto(const v1::ToolCall & call) {
    return {call.id(), call.name(), call.arguments_json()};
}

void to_proto(const ModelInfo & info, v1::ModelInfo * out) {
    out->set_description(info.description);
    out->set_n_params(info.n_params);
    out->set_size_bytes(info.size_bytes);
    out->set_n_ctx(info.n_ctx);
    out->set_n_ctx_train(info.n_ctx_train);
    out->set_has_chat_template(info.has_chat_template);
    out->set_n_embd(info.n_embd);
    out->set_serves_embeddings(info.serves_embeddings);
}

void to_proto(const EmbedResult & result, v1::EmbedResponse * out) {
    out->mutable_embeddings()->Reserve(static_cast<int>(result.embeddings.size()));
    for (const Embedding & embedding : result.embeddings) {
        out->add_embeddings()->mutable_values()->Add(embedding.values.begin(), embedding.values.end());
    }
    out->set_input_tokens(result.input_tokens);
}

v1::GenerateChunk finish_chunk(FinishReason reason, const GenerateStats & stats, std::vector<ToolCall> tool_calls) {
    const bool ran_to_completion = reason == FinishReason::Eog || reason == FinishReason::Stop;
    if (!ran_to_completion) {
        tool_calls.clear();
    }

    v1::GenerateChunk chunk;
    v1::Finish *      finish = chunk.mutable_finish();
    finish->set_reason(tool_calls.empty() ? to_proto(reason) : v1::FINISH_REASON_TOOL_CALLS);
    to_proto(stats, finish->mutable_stats());
    for (const ToolCall & call : tool_calls) {
        to_proto(call, finish->add_tool_calls());
    }
    return chunk;
}

}  // namespace llamad::wire
