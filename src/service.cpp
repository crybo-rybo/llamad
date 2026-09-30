/** @file
 * @brief Stateless RPC handlers, stream finalization and error-to-status translation.
 */

#include "service.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <mutex>
#include <random>
#include <string>
#include <thread>
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
std::string generation_details(const GenerateStats & stats, const char * reason, int tool_calls) {
    char tools[32] = "";
    if (tool_calls >= 0) {
        std::snprintf(tools, sizeof(tools), " tool_calls=%d", tool_calls);
    }
    char details[256];
    std::snprintf(details, sizeof(details),
                  "prompt_tokens=%d cached_prompt_tokens=%d completion_tokens=%d finish=%s%s",
                  stats.prompt_tokens, stats.cached_prompt_tokens, stats.completion_tokens, reason, tools);
    return details;
}

/// Allocate a process-scoped identifier without request contents or a shared request registry.
std::string next_request_id() {
    // Each process chooses a random prefix; the counter separates concurrent RPCs.
    static const uint64_t prefix = [] {
        std::random_device random;
        return (static_cast<uint64_t>(random()) << 32) | random();
    }();
    static std::atomic<uint64_t> sequence{0};
    char id[34];
    std::snprintf(id, sizeof(id), "%016llx-%016llx", static_cast<unsigned long long>(prefix),
                  static_cast<unsigned long long>(sequence.fetch_add(1, std::memory_order_relaxed)));
    return id;
}

/// One log site covers returned errors, exceptions and successful RPCs. Caller mistakes
/// (EngineError, ChatFormatError) map to INVALID_ARGUMENT, context overflow OUT_OF_RANGE,
/// anything else INTERNAL.
template <typename Body>
grpc::Status guarded(grpc::ServerContext * context, const char * rpc_name, Body && body) {
    const auto started = std::chrono::steady_clock::now();
    const std::string id = next_request_id();
    context->AddInitialMetadata("x-request-id", id);
    std::string details;
    grpc::Status status;
    try {
        status = body(details);
    } catch (const ContextOverflowError & e) {
        status = grpc::Status(grpc::StatusCode::OUT_OF_RANGE, e.what());
    } catch (const EngineError & e) {
        status = grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
    } catch (const ChatFormatError & e) {
        status = grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
    } catch (const std::exception & e) {
        status = grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    } catch (...) {
        status = grpc::Status(grpc::StatusCode::INTERNAL, "unknown error");
    }
    // Keep the daemon record on one line; the client's status retains the complete error text.
    std::string error = status.error_message();
    std::replace(error.begin(), error.end(), '\n', ' ');
    std::replace(error.begin(), error.end(), '\r', ' ');
    const double wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    std::fprintf(stderr, "[llamad] %s request_id=%s%s%s status=%d%s%s%s %.0fms\n",
                  rpc_name, id.c_str(), details.empty() ? "" : " ", details.c_str(),
                  static_cast<int>(status.error_code()), context->IsCancelled() ? " cancelled" : "",
                  status.ok() ? "" : " error: ", error.c_str(), wall_ms);
    return status;
}

/// One stream's writes and liveness timer. The synchronous gRPC writer permits one write at
/// a time; its context and writer outlive this object, whose destructor joins the timer.
class StreamOutput {
public:
    /// Borrow the live RPC objects and start its independent liveness timer.
    StreamOutput(grpc::ServerContext * context, grpc::ServerWriter<v1::GenerateChunk> * writer)
        : context_(context), writer_(writer), timer_([this](std::stop_token stop) {
            try {
                std::unique_lock lock(mutex_);
                while (!stop.stop_requested()) {
                    wake_.wait_for(lock, stop, std::chrono::seconds(1), [] { return false; });
                    if (stop.stop_requested() || !write_locked(v1::GenerateChunk{})) {
                        break;
                    }
                }
            } catch (...) {
                error_ = std::current_exception();
                gone_.store(true);
            }
        }) {}

    /// Join before the borrowed RPC objects can be destroyed.
    ~StreamOutput() { stop_timer(); }

    /// Remember cancellation so no further writes or inference are attempted.
    bool active() {
        if (context_->IsCancelled()) {
            gone_.store(true);
        }
        return !gone_.load();
    }

    /// Serialize text and final writes with activity frames.
    bool write(const v1::GenerateChunk & chunk) {
        std::lock_guard lock(mutex_);
        return write_locked(chunk);
    }

    /// Stop reporting before finalization; propagate any timer failure after joining.
    void finish_activity() {
        stop_timer();
        if (error_) {
            std::rethrow_exception(error_);
        }
    }

private:
    /// A failed transport write makes cancellation visible to the inference thread.
    bool write_locked(const v1::GenerateChunk & chunk) {
        if (!active()) {
            return false;
        }
        if (!writer_->Write(chunk)) {
            gone_.store(true);
            return false;
        }
        return true;
    }

    /// Interrupt the timer wait and join, including when generation throws.
    void stop_timer() {
        if (timer_.joinable()) {
            timer_.request_stop();
            timer_.join();
        }
    }

    grpc::ServerContext *                 context_;  ///< Borrowed live server context.
    grpc::ServerWriter<v1::GenerateChunk> * writer_;  ///< Borrowed synchronous stream writer.
    std::atomic<bool>                     gone_{false};  ///< Cancellation shared with inference.
    std::mutex                            mutex_;  ///< Guards every writer operation.
    std::condition_variable_any           wake_;  ///< Timer wait interrupted by its stop token.
    std::exception_ptr                    error_;  ///< Timer failure, read only after joining.
    std::jthread                          timer_;  ///< Declared last so it joins before shared state dies.
};

/// Why Generate and Chat refuse an embedding model.
constexpr const char * kEmbeddingModel = "the model is an embedding model: it serves Embed, not Generate or Chat";

}  // namespace

grpc::Status LlamaService::GetModelInfo(grpc::ServerContext * context,
                                        const v1::GetModelInfoRequest *,
                                        v1::ModelInfo * response) {
    return guarded(context, "GetModelInfo", [&](std::string &) {
        wire::to_proto(engine_.info(), response);
        return grpc::Status::OK;
    });
}

grpc::Status LlamaService::Tokenize(grpc::ServerContext * context,
                                    const v1::TokenizeRequest * request,
                                    v1::TokenizeResponse * response) {
    return guarded(context, "Tokenize", [&](std::string & details) {
        const std::vector<int32_t> tokens =
            engine_.tokenize(request->text(), request->add_special(), request->parse_special());
        response->mutable_tokens()->Add(tokens.begin(), tokens.end());
        details = "tokens=" + std::to_string(tokens.size());
        return grpc::Status::OK;
    });
}

grpc::Status LlamaService::stream_generation(std::string & log_details,
                                             grpc::ServerContext * context,
                                             const std::string & prompt,
                                             const SamplingParams & params,
                                             grpc::ServerWriter<v1::GenerateChunk> * writer,
                                             ChatFormat::Stream * stream) {
    // Activity frames start here, after Chat's template rendering. A timer sends them rather
    // than the engine's loop because a single decode batch can outlast the one-second interval.
    StreamOutput output(context, writer);

    const auto write_text = [&](const std::string & text) {
        v1::GenerateChunk chunk;
        chunk.set_text(text);
        return output.write(chunk);
    };

    const ChunkCallback on_chunk = [&](const std::string & text) -> bool {
        if (!output.active()) {
            return false;
        }
        if (stream == nullptr) {
            return write_text(text);
        }
        // Chat: the parser decides what is visible content. An empty delta means the text is
        // part of a tool call (or not yet known to be), so nothing goes on the wire for it; a
        // departed client is still reported, since generation stops only through this callback.
        const std::string visible = stream->push(text);
        return visible.empty() ? output.active() : write_text(visible);
    };

    const GenerateResult result = engine_.generate(prompt, params, on_chunk, [&] { return output.active(); });
    output.finish_activity();

    v1::FinishReason      reason = wire::enum_cast<v1::FinishReason>(result.reason);
    std::vector<ToolCall> tool_calls;

    if (output.active() && stream != nullptr) {
        ChatFormat::Stream::Final final = stream->finish();
        tool_calls                      = std::move(final.tool_calls);

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
        if (!final.content_tail.empty()) {
            write_text(final.content_tail);
        }
    }

    if (output.active()) {
        // Exactly one final chunk, carrying finish_reason, stats and any tool calls.
        v1::GenerateChunk final_chunk;
        final_chunk.set_finish_reason(reason);
        wire::to_proto(result.stats, final_chunk.mutable_stats());
        for (const ToolCall & call : tool_calls) {
            wire::to_proto(call, final_chunk.add_tool_calls());
        }
        output.write(final_chunk);
    }

    log_details = generation_details(result.stats, wire::value_name(reason),
                                     stream != nullptr ? static_cast<int>(tool_calls.size()) : -1);

    return grpc::Status::OK;
}

grpc::Status LlamaService::Generate(grpc::ServerContext * context,
                                    const v1::GenerateRequest * request,
                                    grpc::ServerWriter<v1::GenerateChunk> * writer) {
    return guarded(context, "Generate", [&](std::string & details) {
        writer->SendInitialMetadata();
        if (engine_.serves_embeddings()) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, kEmbeddingModel);
        }
        return stream_generation(details, context, request->prompt(), from_proto(request->sampling()), writer,
                                 /*stream*/ nullptr);
    });
}

grpc::Status LlamaService::Chat(grpc::ServerContext * context,
                                const v1::ChatRequest * request,
                                grpc::ServerWriter<v1::GenerateChunk> * writer) {
    return guarded(context, "Chat", [&](std::string & details) {
        writer->SendInitialMetadata();
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
            messages.push_back(wire::from_proto<llamad::ChatMessage>(m));
        }

        std::vector<llamad::Tool> tools;
        tools.reserve(static_cast<size_t>(request->tools().size()));
        for (const v1::Tool & t : request->tools()) {
            tools.push_back(wire::from_proto<llamad::Tool>(t));
        }

        // Rendering happens before the engine's generate mutex, so a queued request pays for its
        // own template rendering rather than the one holding the mutex.
        const RenderedChat rendered = chat_format_->render(messages, tools, request->response_json_schema());

        SamplingParams params = from_proto(request->sampling());
        // With tools or a response schema the output is grammar-constrained; with neither the chat
        // layer leaves the grammar empty and this is all a no-op.
        params.grammar          = rendered.grammar;
        params.preserved_tokens = rendered.preserved_tokens;
        params.stop.insert(params.stop.end(), rendered.additional_stops.begin(),
                           rendered.additional_stops.end());

        // One parser per request; the ChatFormat it came from is shared and immutable.
        ChatFormat::Stream stream = chat_format_->stream(rendered);

        return stream_generation(details, context, rendered.prompt, params, writer, &stream);
    });
}

grpc::Status LlamaService::Embed(grpc::ServerContext * context,
                                 const v1::EmbedRequest * request,
                                 v1::EmbedResponse * response) {
    return guarded(context, "Embed", [&](std::string & details) {
        if (request->inputs().empty()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "inputs must not be empty");
        }
        if (!engine_.serves_embeddings()) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                "the model is not an embedding model: its GGUF declares no pooling type");
        }

        const EmbedResult result  = engine_.embed({request->inputs().begin(), request->inputs().end()},
                                                  [context] { return !context->IsCancelled(); });

        // The engine stops short of the whole batch only when the client has gone (a cancel or
        // a deadline), and then the vectors it did make have no one to go to.
        const bool cancelled = result.embeddings.size() < static_cast<size_t>(request->inputs().size());
        details = "inputs=" + std::to_string(request->inputs_size()) +
                  " embedded=" + std::to_string(result.embeddings.size()) +
                  " input_tokens=" + std::to_string(result.input_tokens);
        if (cancelled) {
            return grpc::Status(grpc::StatusCode::CANCELLED, "the client cancelled the call");
        }

        wire::to_proto(result, response);
        return grpc::Status::OK;
    });
}

}  // namespace llamad
