#include "llamad/client.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

#include <grpcpp/grpcpp.h>

#include "llamad/v1/llamad.grpc.pb.h"

namespace llamad {
namespace client {
namespace {

FinishReason from_proto(v1::FinishReason reason) {
    switch (reason) {
        case v1::FINISH_REASON_EOG:       return FinishReason::Eog;
        case v1::FINISH_REASON_LENGTH:    return FinishReason::Length;
        case v1::FINISH_REASON_STOP:      return FinishReason::Stop;
        case v1::FINISH_REASON_CANCELLED: return FinishReason::Cancelled;
        default:                          return FinishReason::Eog;
    }
}

void fill_sampling(v1::SamplingParams * out, const SamplingParams & in) {
    if (in.temperature) { out->set_temperature(*in.temperature); }
    if (in.top_k)       { out->set_top_k(*in.top_k); }
    if (in.top_p)       { out->set_top_p(*in.top_p); }
    if (in.min_p)       { out->set_min_p(*in.min_p); }
    if (in.seed)        { out->set_seed(*in.seed); }
    if (in.max_tokens)  { out->set_max_tokens(*in.max_tokens); }
    for (const std::string & s : in.stop) {
        out->add_stop(s);
    }
}

[[noreturn]] void throw_rpc_error(const grpc::Status & status) {
    throw RpcError(static_cast<int>(status.error_code()), status.error_message());
}

// The daemon must be there already: a missing socket is an error to report, not
// something to wait on, so wait_for_ready stays off (the gRPC default, set
// explicitly here because it is the whole point of the fail-fast behaviour).
void init_context(grpc::ClientContext & context) {
    context.set_wait_for_ready(false);
}

}  // namespace

struct Client::Impl {
    std::shared_ptr<grpc::Channel>    channel;
    std::unique_ptr<v1::Llama::Stub>  stub;

    // Runs a server-streaming call to completion, honouring cancellation.
    template <typename Reader>
    GenerateResult consume(grpc::ClientContext & context,
                           Reader & reader,
                           const ChunkCallback & on_chunk) {
        GenerateResult result{FinishReason::Eog, {}};
        bool cancelled = false;

        v1::GenerateChunk chunk;
        while (reader->Read(&chunk)) {
            if (!cancelled && on_chunk && !chunk.text().empty()) {
                if (!on_chunk(chunk.text())) {
                    cancelled = true;
                    context.TryCancel();
                }
            }
            if (chunk.finish_reason() != v1::FINISH_REASON_UNSPECIFIED) {
                result.reason                   = from_proto(chunk.finish_reason());
                result.stats.prompt_tokens      = chunk.stats().prompt_tokens();
                result.stats.completion_tokens  = chunk.stats().completion_tokens();
                result.stats.prompt_ms          = chunk.stats().prompt_ms();
                result.stats.completion_ms      = chunk.stats().completion_ms();
            }
        }

        const grpc::Status status = reader->Finish();
        if (cancelled) {
            // We asked for the cancel, so CANCELLED here is the expected outcome.
            if (!status.ok() && status.error_code() != grpc::StatusCode::CANCELLED) {
                throw_rpc_error(status);
            }
            result.reason = FinishReason::Cancelled;
            return result;
        }
        if (!status.ok()) {
            throw_rpc_error(status);
        }
        return result;
    }
};

std::string Client::default_socket_path() {
    const char * runtime_dir = std::getenv("XDG_RUNTIME_DIR");
    if (runtime_dir != nullptr && runtime_dir[0] != '\0') {
        return std::string(runtime_dir) + "/llamad.sock";
    }
    return "/tmp/llamad-" + std::to_string(static_cast<unsigned>(getuid())) + ".sock";
}

Client::Client(const std::string & socket_path) : impl_(new Impl) {
    grpc::ChannelArguments args;
    // Nothing in this client benefits from gRPC's own retry/backoff: the daemon
    // is either on the socket or it is not.
    args.SetInt(GRPC_ARG_INITIAL_RECONNECT_BACKOFF_MS, 100);
    args.SetInt(GRPC_ARG_MIN_RECONNECT_BACKOFF_MS, 100);
    impl_->channel = grpc::CreateCustomChannel("unix:" + socket_path,
                                               grpc::InsecureChannelCredentials(),
                                               args);
    impl_->stub = v1::Llama::NewStub(impl_->channel);
}

Client::~Client() = default;

ModelInfo Client::get_model_info() {
    grpc::ClientContext context;
    init_context(context);

    v1::GetModelInfoRequest request;
    v1::ModelInfo          response;
    const grpc::Status status = impl_->stub->GetModelInfo(&context, request, &response);
    if (!status.ok()) {
        throw_rpc_error(status);
    }

    ModelInfo info;
    info.description       = response.description();
    info.n_params          = response.n_params();
    info.size_bytes        = response.size_bytes();
    info.n_ctx             = response.n_ctx();
    info.n_ctx_train       = response.n_ctx_train();
    info.has_chat_template = response.has_chat_template();
    return info;
}

std::vector<int32_t> Client::tokenize(const std::string & text, bool add_special, bool parse_special) {
    grpc::ClientContext context;
    init_context(context);

    v1::TokenizeRequest request;
    request.set_text(text);
    request.set_add_special(add_special);
    request.set_parse_special(parse_special);

    v1::TokenizeResponse response;
    const grpc::Status status = impl_->stub->Tokenize(&context, request, &response);
    if (!status.ok()) {
        throw_rpc_error(status);
    }
    return std::vector<int32_t>(response.tokens().begin(), response.tokens().end());
}

GenerateResult Client::generate(const std::string & prompt,
                                const SamplingParams & params,
                                const ChunkCallback & on_chunk) {
    grpc::ClientContext context;
    init_context(context);

    v1::GenerateRequest request;
    request.set_prompt(prompt);
    fill_sampling(request.mutable_sampling(), params);

    auto reader = impl_->stub->Generate(&context, request);
    return impl_->consume(context, reader, on_chunk);
}

GenerateResult Client::chat(const std::vector<ChatMessage> & messages,
                            const SamplingParams & params,
                            const ChunkCallback & on_chunk) {
    grpc::ClientContext context;
    init_context(context);

    v1::ChatRequest request;
    for (const ChatMessage & m : messages) {
        v1::ChatMessage * out = request.add_messages();
        out->set_role(m.role);
        out->set_content(m.content);
    }
    fill_sampling(request.mutable_sampling(), params);

    auto reader = impl_->stub->Chat(&context, request);
    return impl_->consume(context, reader, on_chunk);
}

}  // namespace client
}  // namespace llamad
