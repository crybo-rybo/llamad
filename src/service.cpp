#include "service.h"

#include <chrono>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace llamad {
namespace {

const char * finish_reason_name(FinishReason reason) {
    switch (reason) {
        case FinishReason::Eog:       return "EOG";
        case FinishReason::Length:    return "LENGTH";
        case FinishReason::Stop:      return "STOP";
        case FinishReason::Cancelled: return "CANCELLED";
    }
    return "UNSPECIFIED";
}

v1::FinishReason to_proto(FinishReason reason) {
    switch (reason) {
        case FinishReason::Eog:       return v1::FINISH_REASON_EOG;
        case FinishReason::Length:    return v1::FINISH_REASON_LENGTH;
        case FinishReason::Stop:      return v1::FINISH_REASON_STOP;
        case FinishReason::Cancelled: return v1::FINISH_REASON_CANCELLED;
    }
    return v1::FINISH_REASON_UNSPECIFIED;
}

// Only fields the client actually set override the engine defaults from engine.h.
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

void fill_stats(v1::GenerateStats * out, const GenerateStats & in) {
    out->set_prompt_tokens(in.prompt_tokens);
    out->set_completion_tokens(in.completion_tokens);
    out->set_prompt_ms(in.prompt_ms);
    out->set_completion_ms(in.completion_ms);
}

void log_request(const char * rpc_name, const GenerateStats & stats, const char * reason, double wall_ms) {
    std::fprintf(stderr, "[llamad] %s prompt_tokens=%d completion_tokens=%d finish=%s %.0fms\n",
                 rpc_name, stats.prompt_tokens, stats.completion_tokens, reason, wall_ms);
}

}  // namespace

grpc::Status LlamaService::GetModelInfo(grpc::ServerContext *,
                                        const v1::GetModelInfoRequest *,
                                        v1::ModelInfo * response) {
    try {
        const llamad::ModelInfo info = engine_.info();
        response->set_description(info.description);
        response->set_n_params(info.n_params);
        response->set_size_bytes(info.size_bytes);
        response->set_n_ctx(info.n_ctx);
        response->set_n_ctx_train(info.n_ctx_train);
        response->set_has_chat_template(info.has_chat_template);
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
                                             const v1::SamplingParams & sampling,
                                             grpc::ServerWriter<v1::GenerateChunk> * writer) {
    const auto started = std::chrono::steady_clock::now();

    // Set as soon as the client is known to be gone (cancel or broken stream);
    // it suppresses the final chunk, which would only fail to write anyway.
    bool client_gone = false;

    const ChunkCallback on_chunk = [&](const std::string & text) -> bool {
        if (context->IsCancelled()) {
            client_gone = true;
            return false;
        }
        v1::GenerateChunk chunk;
        chunk.set_text(text);
        if (!writer->Write(chunk)) {
            client_gone = true;
            return false;
        }
        return true;
    };

    try {
        const GenerateResult result = engine_.generate(prompt, from_proto(sampling), on_chunk);

        if (context->IsCancelled()) {
            client_gone = true;
        }
        if (!client_gone) {
            // Exactly one final chunk, carrying finish_reason and stats.
            v1::GenerateChunk final_chunk;
            final_chunk.set_finish_reason(to_proto(result.reason));
            fill_stats(final_chunk.mutable_stats(), result.stats);
            if (!writer->Write(final_chunk)) {
                client_gone = true;
            }
        }

        const double wall_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        log_request(rpc_name, result.stats, finish_reason_name(result.reason), wall_ms);

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
    return stream_generation("Generate", context, request->prompt(), request->sampling(), writer);
}

grpc::Status LlamaService::Chat(grpc::ServerContext * context,
                                const v1::ChatRequest * request,
                                grpc::ServerWriter<v1::GenerateChunk> * writer) {
    std::string prompt;
    try {
        if (request->messages().empty()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "messages must not be empty");
        }
        if (!engine_.info().has_chat_template) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                "the model has no built-in chat template");
        }

        std::vector<llamad::ChatMessage> messages;
        messages.reserve(static_cast<size_t>(request->messages().size()));
        for (const v1::ChatMessage & m : request->messages()) {
            messages.push_back({m.role(), m.content()});
        }
        prompt = engine_.apply_chat_template(messages);
    } catch (const EngineError & e) {
        std::fprintf(stderr, "[llamad] Chat error: %s\n", e.what());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[llamad] Chat internal error: %s\n", e.what());
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    } catch (...) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "unknown error");
    }

    return stream_generation("Chat", context, prompt, request->sampling(), writer);
}

}  // namespace llamad
