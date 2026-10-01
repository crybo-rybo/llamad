/** @file
 * @brief Stateless RPC handlers, stream finalization and error-to-status translation.
 */

#include "service.h"

#include <chrono>
#include <cstdio>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "wire.h"

namespace llamad {
namespace {

/// `tool_calls` < 0 leaves the count out of the line entirely (Generate has no tool calls).
void log_request(const char * rpc_name, const GenerateStats & stats, const std::string & reason, double wall_ms,
                 int tool_calls) {
    char tools[32] = "";
    if (tool_calls >= 0) {
        std::snprintf(tools, sizeof(tools), " tool_calls=%d", tool_calls);
    }
    std::fprintf(stderr,
                 "[llamad] %s prompt_tokens=%d cached_prompt_tokens=%d completion_tokens=%d finish=%s%s %.0fms\n",
                 rpc_name, stats.prompt_tokens, stats.cached_prompt_tokens, stats.completion_tokens, reason.c_str(), tools,
                 wall_ms);
}

/// Runs one RPC body and turns what it throws into the status the contract promises: caller
/// mistakes (EngineError, ChatFormatError) INVALID_ARGUMENT, anything else INTERNAL.
template <typename Body>
grpc::Status guarded(const char * rpc_name, Body && body) {
    try {
        return body();
    } catch (const EngineError & e) {
        std::fprintf(stderr, "[llamad] %s error: %s\n", rpc_name, e.what());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
    } catch (const ChatFormatError & e) {
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

/// Why Generate and Chat refuse an embedding model.
constexpr const char * kEmbeddingModel = "the model is an embedding model: it serves Embed, not Generate or Chat";

/// Milliseconds since `started`, for the per-request log line.
double ms_since(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

}  // namespace

grpc::Status LlamaService::GetModelInfo(grpc::ServerContext *,
                                        const v1::GetModelInfoRequest *,
                                        v1::ModelInfo * response) {
    return guarded("GetModelInfo", [&] {
        wire::to_proto(engine_.info(), response);
        std::fprintf(stderr, "[llamad] GetModelInfo\n");
        return grpc::Status::OK;
    });
}

grpc::Status LlamaService::Tokenize(grpc::ServerContext *,
                                    const v1::TokenizeRequest * request,
                                    v1::TokenizeResponse * response) {
    return guarded("Tokenize", [&] {
        const std::vector<int32_t> tokens =
            engine_.tokenize(request->text(), request->add_special(), request->parse_special());
        response->mutable_tokens()->Add(tokens.begin(), tokens.end());
        std::fprintf(stderr, "[llamad] Tokenize tokens=%zu\n", tokens.size());
        return grpc::Status::OK;
    });
}

grpc::Status LlamaService::stream_generation(const char * rpc_name,
                                             grpc::ServerContext * context,
                                             const std::string & prompt,
                                             const SamplingParams & params,
                                             grpc::ServerWriter<v1::GenerateChunk> * writer,
                                             ChatFormat::Stream * stream) {
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
        if (stream == nullptr) {
            return write_text(text);
        }
        // Chat: the parser decides what is visible content. An empty delta means the text is
        // part of a tool call (or not yet known to be), so nothing goes on the wire for it.
        const std::string visible = stream->push(text);
        return visible.empty() ? true : write_text(visible);
    };

    const GenerateResult result = engine_.generate(prompt, params, on_chunk);

    if (context->IsCancelled()) {
        client_gone = true;
    }

    std::vector<ToolCall> tool_calls;
    if (!client_gone && stream != nullptr) {
        ChatFormat::Stream::Final final = stream->finish();
        tool_calls                      = std::move(final.tool_calls);
        // Content the parser only became sure about at the end is just more text.
        if (!final.content_tail.empty()) {
            write_text(final.content_tail);
        }
    }

    const v1::GenerateChunk finish = wire::finish_chunk(result.reason, result.stats, std::move(tool_calls));
    if (!client_gone && !writer->Write(finish)) {
        client_gone = true;
    }

    log_request(rpc_name, result.stats, v1::FinishReason_Name(finish.finish().reason()), ms_since(started),
                stream != nullptr ? finish.finish().tool_calls_size() : -1);

    return grpc::Status::OK;
}

grpc::Status LlamaService::Generate(grpc::ServerContext * context,
                                    const v1::GenerateRequest * request,
                                    grpc::ServerWriter<v1::GenerateChunk> * writer) {
    return guarded("Generate", [&] {
        if (engine_.serves_embeddings()) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, kEmbeddingModel);
        }
        return stream_generation("Generate", context, request->prompt(), wire::from_proto(request->sampling()), writer,
                                 /*stream*/ nullptr);
    });
}

grpc::Status LlamaService::Chat(grpc::ServerContext * context,
                                const v1::ChatRequest * request,
                                grpc::ServerWriter<v1::GenerateChunk> * writer) {
    return guarded("Chat", [&] {
        if (request->messages().empty()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "messages must not be empty");
        }
        if (engine_.serves_embeddings()) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, kEmbeddingModel);
        }
        if (chat_format_ == nullptr) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, chat_unavailable_reason_);
        }

        std::vector<llamad::ChatMessage> messages;
        messages.reserve(static_cast<size_t>(request->messages().size()));
        for (const v1::ChatMessage & m : request->messages()) {
            messages.push_back(wire::from_proto(m));
        }

        std::vector<llamad::Tool> tools;
        tools.reserve(static_cast<size_t>(request->tools().size()));
        for (const v1::Tool & t : request->tools()) {
            tools.push_back(wire::from_proto(t));
        }

        // Rendering happens before the engine's generate mutex, so a queued request pays for its
        // own template rendering rather than the one holding the mutex.
        const RenderedChat rendered = chat_format_->render(messages, tools, request->response_json_schema());

        SamplingParams params = wire::from_proto(request->sampling());
        // With tools or a response schema the output is grammar-constrained; with neither the chat
        // layer leaves the grammar empty and this is all a no-op.
        params.grammar          = rendered.grammar;
        params.preserved_tokens = rendered.preserved_tokens;
        params.stop.insert(params.stop.end(), rendered.additional_stops.begin(),
                           rendered.additional_stops.end());

        // One parser per request; the ChatFormat it came from is shared and immutable.
        ChatFormat::Stream stream = chat_format_->stream(rendered);

        return stream_generation("Chat", context, rendered.prompt, params, writer, &stream);
    });
}

grpc::Status LlamaService::Embed(grpc::ServerContext * context,
                                 const v1::EmbedRequest * request,
                                 v1::EmbedResponse * response) {
    return guarded("Embed", [&] {
        if (request->inputs().empty()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "inputs must not be empty");
        }
        if (!engine_.serves_embeddings()) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                "the model is not an embedding model: its GGUF declares no pooling type");
        }

        const auto        started = std::chrono::steady_clock::now();
        const EmbedResult result  = engine_.embed({request->inputs().begin(), request->inputs().end()},
                                                  [context] { return !context->IsCancelled(); });

        // The engine stops short of the whole batch only when the client has gone (a cancel or
        // a deadline), and then the vectors it did make have no one to go to.
        const bool cancelled = result.embeddings.size() < static_cast<size_t>(request->inputs().size());
        std::fprintf(stderr, "[llamad] Embed inputs=%d embedded=%zu input_tokens=%d%s %.0fms\n",
                     request->inputs().size(), result.embeddings.size(), result.input_tokens,
                     cancelled ? " cancelled" : "", ms_since(started));
        if (cancelled) {
            return grpc::Status(grpc::StatusCode::CANCELLED, "the client cancelled the call");
        }

        wire::to_proto(result, response);
        return grpc::Status::OK;
    });
}

}  // namespace llamad
