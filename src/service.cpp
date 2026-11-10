/** @file
 * @brief Stateless RPC handlers, stream finalization and error-to-status translation.
 */

#include "service.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "wire.h"

namespace llamad {
namespace {

/// The initial-metadata key that carries the request id, as llamad.proto documents it.
constexpr char kRequestIdKey[] = "x-request-id";

/// 64 bits from std::random_device, as 16 hex digits. A counter or a seeded generator would
/// repeat ids after a restart, and a log file outlives one run of the daemon.
std::string new_request_id() {
    std::random_device device;
    const uint64_t     value = (static_cast<uint64_t>(device()) << 32) | device();
    char               text[17];
    std::snprintf(text, sizeof(text), "%016" PRIx64, value);
    return text;
}

}  // namespace

/// One request's id and its `[llamad]` log lines. Every line a request writes goes through here,
/// so each one starts the same way: `[llamad] <rpc> id=<id>`, then the details.
class RequestLog {
public:
    /// Draws the id and adds it to the initial metadata, which gRPC sends with the first message,
    /// the status or an explicit SendInitialMetadata, whichever comes first.
    RequestLog(grpc::ServerContext * context, const char * rpc_name) : rpc_name_(rpc_name), id_(new_request_id()) {
        context->AddInitialMetadata(kRequestIdKey, id_);
    }

    /// Writes a line with no details.
    void line() { write({}); }

    /// Writes a line with printf-style details.
    [[gnu::format(printf, 2, 3)]] void line(const char * format, ...) {
        std::va_list args;
        va_start(args, format);
        std::va_list size_args;
        va_copy(size_args, args);
        const int size = std::vsnprintf(nullptr, 0, format, size_args);
        va_end(size_args);
        std::string details(static_cast<size_t>(std::max(size, 0)), '\0');
        std::vsnprintf(details.data(), details.size() + 1, format, args);
        va_end(args);
        write(details);
    }

    /// Whether the request has written a line yet.
    bool wrote() const { return wrote_; }

private:
    /// One fprintf per line, so lines from concurrent requests do not interleave.
    void write(const std::string & details) {
        std::fprintf(stderr, "[llamad] %s id=%s%s%s\n", rpc_name_, id_.c_str(), details.empty() ? "" : " ",
                     details.c_str());
        wrote_ = true;
    }

    const char *      rpc_name_;
    const std::string id_;
    bool              wrote_ = false;
};

namespace {

/// The engine's spelling of a stage, for the log line of a cancelled request.
const char * stage_name(GenerateStage stage) {
    switch (stage) {
        case GenerateStage::Queued:     return "queued";
        case GenerateStage::Prompt:     return "prompt";
        case GenerateStage::Generating: return "generating";
    }
    return "?";
}

/// One line per Generate or Chat request. `cancelled_at` is null for a request that finished,
/// and otherwise the last step the engine reported: where the request stopped and how much of its
/// prompt is in the cache. `tool_calls` < 0 leaves the count out of the line entirely (Generate
/// has no tool calls).
void log_request(RequestLog & log, const GenerateStats & stats, const std::string & reason,
                 const GenerateProgress * cancelled_at, double wall_ms, int tool_calls) {
    char tools[32] = "";
    if (tool_calls >= 0) {
        std::snprintf(tools, sizeof(tools), " tool_calls=%d", tool_calls);
    }
    char where[80] = "";
    if (cancelled_at != nullptr) {
        std::snprintf(where, sizeof(where), " stage=%s ready_prompt_tokens=%d", stage_name(cancelled_at->stage),
                      cancelled_at->ready_prompt_tokens);
    }
    log.line("prompt_tokens=%d cached_prompt_tokens=%d completion_tokens=%d finish=%s%s%s %.0fms", stats.prompt_tokens,
             stats.cached_prompt_tokens, stats.completion_tokens, reason.c_str(), where, tools, wall_ms);
}

/// How long a stream may be silent before the next engine step sends a progress chunk.
constexpr std::chrono::seconds kProgressAfter{1};

/// Runs one RPC body with the request's RequestLog and turns what it throws into the status the
/// contract promises: an input too long for the model OUT_OF_RANGE, because the caller can shorten
/// it and retry; other caller mistakes (EngineError, ChatFormatError) INVALID_ARGUMENT; anything
/// else INTERNAL. A failure the body returns rather than throws gets its error line here, so
/// every request writes at least one line.
template <typename Body>
grpc::Status guarded(grpc::ServerContext * context, const char * rpc_name, Body && body) {
    RequestLog log(context, rpc_name);
    try {
        grpc::Status status = body(log);
        if (!status.ok() && !log.wrote()) {
            log.line("error: %s", status.error_message().c_str());
        }
        return status;
    } catch (const InputTooLongError & e) {
        log.line("error: %s", e.what());
        return grpc::Status(grpc::StatusCode::OUT_OF_RANGE, e.what());
    } catch (const EngineError & e) {
        log.line("error: %s", e.what());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
    } catch (const ChatFormatError & e) {
        log.line("error: %s", e.what());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
    } catch (const std::exception & e) {
        log.line("internal error: %s", e.what());
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    } catch (...) {
        log.line("internal error: unknown exception");
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

grpc::Status LlamaService::GetModelInfo(grpc::ServerContext * context,
                                        const v1::GetModelInfoRequest *,
                                        v1::ModelInfo * response) {
    return guarded(context, "GetModelInfo", [&](RequestLog & log) {
        wire::to_proto(engine_.info(), daemon_, response);
        log.line();
        return grpc::Status::OK;
    });
}

grpc::Status LlamaService::Tokenize(grpc::ServerContext * context,
                                    const v1::TokenizeRequest * request,
                                    v1::TokenizeResponse * response) {
    return guarded(context, "Tokenize", [&](RequestLog & log) {
        const std::vector<int32_t> tokens =
            engine_.tokenize(request->text(), request->add_special(), request->parse_special());
        response->mutable_tokens()->Add(tokens.begin(), tokens.end());
        log.line("tokens=%zu", tokens.size());
        return grpc::Status::OK;
    });
}

grpc::Status LlamaService::stream_generation(RequestLog & log,
                                             grpc::ServerContext * context,
                                             const std::string & prompt,
                                             const SamplingParams & params,
                                             grpc::ServerWriter<v1::GenerateChunk> * writer,
                                             ChatFormat::Stream * stream) {
    const auto started = std::chrono::steady_clock::now();

    // Set as soon as the client is known to be gone (cancel or broken stream); it stops the
    // engine and suppresses the final chunk, which would only fail to write anyway.
    bool client_gone = false;

    // When the stream last carried a chunk of any kind; the start counts as one, so a request
    // that is fast end to end never sends a progress chunk.
    auto last_write = started;

    // Writes one chunk. Returns false once the client is gone.
    const auto write = [&](const v1::GenerateChunk & chunk) -> bool {
        if (!writer->Write(chunk)) {
            client_gone = true;
            return false;
        }
        last_write = std::chrono::steady_clock::now();
        return true;
    };

    const auto write_text = [&](const std::string & text) -> bool {
        v1::GenerateChunk chunk;
        chunk.set_text(text);
        return write(chunk);
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

    // The engine asks at every step, so a client that leaves while the request is queued or its
    // prompt decodes stops it at the next step, not only once text flows. The steps also drive
    // progress chunks, so there is no timer thread: one goes out at the first step after a second
    // of silence. The last step is kept for the log line.
    GenerateProgress last_progress;
    const ProgressCallback on_progress = [&](const GenerateProgress & progress) -> bool {
        last_progress = progress;
        if (context->IsCancelled()) {
            client_gone = true;
            return false;
        }
        if (std::chrono::steady_clock::now() - last_write < kProgressAfter) {
            return true;
        }
        return write(wire::progress_chunk(progress));
    };

    const GenerateResult result = engine_.generate(prompt, params, on_chunk, on_progress);

    // The engine ends with Cancelled only when one of the callbacks above saw the client go.
    if (result.reason == FinishReason::Cancelled || context->IsCancelled()) {
        client_gone = true;
    }

    // A finish chunk is built only for a generation the client is still there for; the log line
    // still gives its reason if writing it then fails.
    std::string reason     = "cancelled";
    int         tool_count = 0;
    const bool  finished   = !client_gone;
    if (finished) {
        std::vector<ToolCall> tool_calls;
        if (stream != nullptr) {
            ChatFormat::Stream::Final final = stream->finish();
            tool_calls                      = std::move(final.tool_calls);
            // Content the parser only became sure about at the end is just more text.
            if (!final.content_tail.empty()) {
                write_text(final.content_tail);
            }
        }

        const v1::GenerateChunk finish = wire::finish_chunk(result.reason, result.stats, std::move(tool_calls));
        reason                         = v1::FinishReason_Name(finish.finish().reason());
        tool_count                     = finish.finish().tool_calls_size();
        if (!client_gone) {
            write(finish);
        }
    }

    log_request(log, result.stats, reason, finished ? nullptr : &last_progress, ms_since(started),
                stream != nullptr ? tool_count : -1);

    // The client is gone, so it never reads this status; it tells gRPC and the log the truth.
    if (client_gone) {
        return grpc::Status(grpc::StatusCode::CANCELLED, "the client cancelled the call");
    }
    return grpc::Status::OK;
}

grpc::Status LlamaService::Generate(grpc::ServerContext * context,
                                    const v1::GenerateRequest * request,
                                    grpc::ServerWriter<v1::GenerateChunk> * writer) {
    return guarded(context, "Generate", [&](RequestLog & log) {
        // The id goes out now: gRPC would otherwise hold it until the first chunk, which a request
        // waiting in the queue does not send for a second.
        writer->SendInitialMetadata();
        if (engine_.serves_embeddings()) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, kEmbeddingModel);
        }
        return stream_generation(log, context, request->prompt(), wire::from_proto(request->sampling()), writer,
                                 /*stream*/ nullptr);
    });
}

grpc::Status LlamaService::Chat(grpc::ServerContext * context,
                                const v1::ChatRequest * request,
                                grpc::ServerWriter<v1::GenerateChunk> * writer) {
    return guarded(context, "Chat", [&](RequestLog & log) {
        writer->SendInitialMetadata();  // as in Generate
        if (request->messages().empty()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "messages must not be empty");
        }
        if (engine_.serves_embeddings()) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, kEmbeddingModel);
        }
        if (chat_format_ == nullptr) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, chat_unavailable_reason_);
        }

        // Rendering happens before the engine's generate mutex, so a queued request pays for its
        // own template rendering rather than the one holding the mutex.
        const RenderedChat rendered = chat_format_->render(wire::from_proto(*request));

        SamplingParams params = wire::from_proto(request->sampling());
        // With tools or a response schema the output is grammar-constrained; with neither the chat
        // layer leaves the grammar empty and this is all a no-op.
        params.grammar          = rendered.grammar;
        params.preserved_tokens = rendered.preserved_tokens;
        params.stop.insert(params.stop.end(), rendered.additional_stops.begin(),
                           rendered.additional_stops.end());

        // One parser per request; the ChatFormat it came from is shared and immutable.
        ChatFormat::Stream stream = chat_format_->stream(rendered);

        return stream_generation(log, context, rendered.prompt, params, writer, &stream);
    });
}

grpc::Status LlamaService::Embed(grpc::ServerContext * context,
                                 const v1::EmbedRequest * request,
                                 v1::EmbedResponse * response) {
    return guarded(context, "Embed", [&](RequestLog & log) {
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
        log.line("inputs=%d embedded=%zu input_tokens=%d%s %.0fms", request->inputs().size(),
                 result.embeddings.size(), result.input_tokens, cancelled ? " cancelled" : "", ms_since(started));
        if (cancelled) {
            return grpc::Status(grpc::StatusCode::CANCELLED, "the client cancelled the call");
        }

        wire::to_proto(result, response);
        return grpc::Status::OK;
    });
}

}  // namespace llamad
