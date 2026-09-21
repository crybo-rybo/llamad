/** @file
 * @brief Stateless RPC handlers, stream finalization and error-to-status translation.
 */

#include "service.h"

#include <chrono>
#include <cstdio>
#include <exception>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llamad/v1/convert.h"

namespace llamad {
namespace {

/// Only fields the client actually set override the engine defaults from engine.h.
SamplingParams from_proto(const v1::SamplingParams & p) {
    SamplingParams out;  // engine defaults
    if (p.has_temperature()) { out.temperature = p.temperature(); }
    if (p.has_top_k())       { out.top_k       = p.top_k(); }
    if (p.has_top_p())       { out.top_p       = p.top_p(); }
    if (p.has_min_p())       { out.min_p       = p.min_p(); }
    if (p.has_seed())        { out.seed        = p.seed(); }
    if (p.has_max_tokens())  { out.max_tokens  = p.max_tokens(); }
    out.stop.assign(p.stop().begin(), p.stop().end());
    return out;
}

/// `tool_calls` < 0 leaves the count out of the line entirely (Generate has no tool calls).
void log_request(const char * rpc_name, const GenerateStats & stats, const char * reason, double wall_ms,
                 int tool_calls) {
    char tools[32] = "";
    if (tool_calls >= 0) {
        std::snprintf(tools, sizeof(tools), " tool_calls=%d", tool_calls);
    }
    std::fprintf(stderr, "[llamad] %s prompt_tokens=%d completion_tokens=%d finish=%s%s %.0fms\n",
                 rpc_name, stats.prompt_tokens, stats.completion_tokens, reason, tools, wall_ms);
}

}  // namespace

grpc::Status LlamaService::GetModelInfo(grpc::ServerContext *,
                                        const v1::GetModelInfoRequest *,
                                        v1::ModelInfo * response) {
    try {
        wire::to_proto(engine_.info(), response);
        std::fprintf(stderr, "[llamad] GetModelInfo\n");
        return grpc::Status::OK;
    } catch (const EngineError & e) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
    } catch (const std::exception & e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
}

grpc::Status LlamaService::Tokenize(grpc::ServerContext *,
                                    const v1::TokenizeRequest * request,
                                    v1::TokenizeResponse * response) {
    try {
        const std::vector<int32_t> tokens =
            engine_.tokenize(request->text(), request->add_special(), request->parse_special());
        response->mutable_tokens()->Reserve(static_cast<int>(tokens.size()));
        for (int32_t t : tokens) {
            response->add_tokens(t);
        }
        std::fprintf(stderr, "[llamad] Tokenize tokens=%zu\n", tokens.size());
        return grpc::Status::OK;
    } catch (const EngineError & e) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
    } catch (const std::exception & e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
}

grpc::Status LlamaService::stream_generation(const char * rpc_name,
                                             grpc::ServerContext * context,
                                             const std::string & prompt,
                                             const SamplingParams & params,
                                             grpc::ServerWriter<v1::GenerateChunk> * writer,
                                             const StreamHooks * hooks) {
    const auto started = std::chrono::steady_clock::now();

    // Set as soon as the client is known to be gone (cancel or broken stream);
    // it suppresses the final chunk, which would only fail to write anyway.
    bool client_gone = false;

    // Writes one ordinary text chunk. Returns false once the client is gone.
    const auto write_text = [&](const std::string & text) -> bool {
        v1::GenerateChunk chunk;
        chunk.set_text(text);
        if (!writer->Write(chunk)) {
            client_gone = true;
            return false;
        }
        return true;
    };

    const ChunkCallback on_chunk = [&](const std::string & text) -> bool {
        if (context->IsCancelled()) {
            client_gone = true;
            return false;
        }
        if (hooks == nullptr) {
            return write_text(text);
        }
        // Chat: the parser decides what is visible content. An empty delta means the text is
        // part of a tool call (or not yet known to be), so nothing goes on the wire for it.
        const std::string visible = hooks->filter(text);
        return visible.empty() ? true : write_text(visible);
    };

    try {
        const GenerateResult result = engine_.generate(prompt, params, on_chunk);

        if (context->IsCancelled()) {
            client_gone = true;
        }

        v1::FinishReason      reason = wire::enum_cast<v1::FinishReason>(result.reason);
        std::vector<ToolCall> tool_calls;

        if (!client_gone && hooks != nullptr) {
            std::string tail;
            hooks->finish(&tail, &tool_calls);

            // Invariant, both ways: tool_calls is non-empty on the wire iff finish_reason is
            // TOOL_CALLS. A run cut short by the token limit or by the client stopped mid-thought,
            // so whatever the parser salvaged is not a call anyone should execute.
            const bool ran_to_completion =
                result.reason == FinishReason::Eog || result.reason == FinishReason::Stop;
            if (ran_to_completion && !tool_calls.empty()) {
                reason = v1::FINISH_REASON_TOOL_CALLS;
            } else {
                tool_calls.clear();
            }

            // Content the parser only became sure about at the end is just more text.
            if (!tail.empty()) {
                write_text(tail);
            }
        }

        if (!client_gone) {
            // Exactly one final chunk, carrying finish_reason, stats and any tool calls.
            v1::GenerateChunk final_chunk;
            final_chunk.set_finish_reason(reason);
            wire::to_proto(result.stats, final_chunk.mutable_stats());
            for (const ToolCall & call : tool_calls) {
                wire::to_proto(call, final_chunk.add_tool_calls());
            }
            if (!writer->Write(final_chunk)) {
                client_gone = true;
            }
        }

        const double wall_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        log_request(rpc_name, result.stats, wire::value_name(reason), wall_ms,
                    hooks != nullptr ? static_cast<int>(tool_calls.size()) : -1);

        return grpc::Status::OK;
    } catch (const EngineError & e) {
        std::fprintf(stderr, "[llamad] %s error: %s\n", rpc_name, e.what());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[llamad] %s internal error: %s\n", rpc_name, e.what());
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    } catch (...) {
        std::fprintf(stderr, "[llamad] %s internal error: unknown exception\n", rpc_name);
        return grpc::Status(grpc::StatusCode::INTERNAL, "unknown error");
    }
}

grpc::Status LlamaService::Generate(grpc::ServerContext * context,
                                    const v1::GenerateRequest * request,
                                    grpc::ServerWriter<v1::GenerateChunk> * writer) {
    return stream_generation("Generate", context, request->prompt(), from_proto(request->sampling()), writer,
                             /*hooks*/ nullptr);
}

grpc::Status LlamaService::Chat(grpc::ServerContext * context,
                                const v1::ChatRequest * request,
                                grpc::ServerWriter<v1::GenerateChunk> * writer) {
    RenderedChat   rendered;
    SamplingParams params;

    // Rendering happens before the engine's generate mutex, so a queued request pays for its own
    // template rendering rather than the one holding the mutex.
    try {
        if (request->messages().empty()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "messages must not be empty");
        }
        if (chat_format_ == nullptr) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, chat_unavailable_reason_);
        }

        std::vector<llamad::ChatMessage> messages;
        messages.reserve(static_cast<size_t>(request->messages().size()));
        for (const v1::ChatMessage & m : request->messages()) {
            messages.push_back(wire::from_proto<llamad::ChatMessage>(m));
        }

        std::vector<llamad::Tool> tools;
        tools.reserve(static_cast<size_t>(request->tools().size()));
        for (const v1::Tool & t : request->tools()) {
            tools.push_back(wire::from_proto<llamad::Tool>(t));
        }

        rendered = chat_format_->render(messages, tools);

        params = from_proto(request->sampling());
        // With tools the model's arguments are grammar-constrained; without them the chat layer
        // leaves the grammar empty and this is all a no-op.
        params.grammar                  = rendered.grammar.grammar;
        params.grammar_lazy             = rendered.grammar.lazy;
        params.grammar_trigger_patterns = rendered.grammar.trigger_patterns;
        params.grammar_trigger_words    = rendered.grammar.trigger_words;
        params.preserved_tokens         = rendered.preserved_tokens;
        params.stop.insert(params.stop.end(), rendered.additional_stops.begin(),
                           rendered.additional_stops.end());
    } catch (const ChatFormatError & e) {
        std::fprintf(stderr, "[llamad] Chat error: %s\n", e.what());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
    } catch (const EngineError & e) {
        std::fprintf(stderr, "[llamad] Chat error: %s\n", e.what());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[llamad] Chat internal error: %s\n", e.what());
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    } catch (...) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "unknown error");
    }

    // One parser per request; the ChatFormat it came from is shared and immutable.
    std::optional<ChatFormat::Stream> stream;
    try {
        stream = chat_format_->stream(rendered);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[llamad] Chat internal error: %s\n", e.what());
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }

    StreamHooks hooks;
    hooks.filter = [&](const std::string & text) { return stream->push(text); };
    hooks.finish = [&](std::string * tail, std::vector<ToolCall> * calls) {
        ChatFormat::Stream::Final final = stream->finish();
        *tail  = std::move(final.content_tail);
        *calls = std::move(final.tool_calls);
    };

    return stream_generation("Chat", context, rendered.prompt, params, writer, &hooks);
}

}  // namespace llamad
