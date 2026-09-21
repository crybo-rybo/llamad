#include "llamad/client.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

#include <grpcpp/grpcpp.h>

#include "llamad/v1/convert.h"
#include "llamad/v1/llamad.grpc.pb.h"

namespace llamad {
namespace client {
namespace {

[[noreturn]] void throw_rpc_error(const grpc::Status & status) {
    throw RpcError(static_cast<int>(status.error_code()), status.error_message());
}

// What a tool that could not run reports back to the model: {"error":"..."}.
struct ToolError {
    std::string error;
};

std::string tool_error(const std::string & message) {
    return json::write(ToolError{message});
}

// The daemon must be there already: a missing socket is an error to report, not
// something to wait on, so wait_for_ready stays off (the gRPC default, set
// explicitly here because it is the whole point of the fail-fast behaviour).
void init_context(grpc::ClientContext & context) {
    context.set_wait_for_ready(false);
}

}  // namespace

std::string ToolSet::call(const ToolCall & call) const {
    for (size_t i = 0; i < definitions_.size(); ++i) {
        if (definitions_[i].name != call.name) {
            continue;
        }
        try {
            return invoke_[i](call.arguments_json);
        } catch (const std::exception & e) {
            return tool_error(e.what());
        }
    }
    return tool_error("no such tool: " + call.name);
}

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
                // A reason a later daemon knows and this client does not reads as Eog.
                result.reason = wire::enum_cast(chunk.finish_reason(), FinishReason::Eog);
                result.stats  = wire::from_proto<GenerateStats>(chunk.stats());
                result.tool_calls.clear();
                for (const v1::ToolCall & call : chunk.tool_calls()) {
                    result.tool_calls.push_back(wire::from_proto<ToolCall>(call));
                }
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

    return wire::from_proto<ModelInfo>(response);
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
    wire::to_proto(params, request.mutable_sampling());

    auto reader = impl_->stub->Generate(&context, request);
    return impl_->consume(context, reader, on_chunk);
}

GenerateResult Client::chat(const std::vector<ChatMessage> & messages,
                            const SamplingParams & params,
                            const ChunkCallback & on_chunk) {
    return chat(messages, {}, params, on_chunk);
}

GenerateResult Client::chat(const std::vector<ChatMessage> & messages,
                            const std::vector<Tool> & tools,
                            const SamplingParams & params,
                            const ChunkCallback & on_chunk) {
    grpc::ClientContext context;
    init_context(context);

    v1::ChatRequest request;
    for (const ChatMessage & m : messages) {
        wire::to_proto(m, request.add_messages());
    }
    for (const Tool & t : tools) {
        wire::to_proto(t, request.add_tools());
    }
    wire::to_proto(params, request.mutable_sampling());

    auto reader = impl_->stub->Chat(&context, request);
    return impl_->consume(context, reader, on_chunk);
}

GenerateResult Client::chat(std::vector<ChatMessage> & history,
                            const ToolSet & tools,
                            const SamplingParams & params,
                            const ChunkCallback & on_chunk,
                            int max_rounds) {
    GenerateResult result{FinishReason::Eog, {}, {}};
    GenerateStats  total{};

    for (int round = 0; round < max_rounds; ++round) {
        std::string reply;
        result = chat(history, tools.definitions(), params, [&](const std::string & text) {
            reply += text;
            return on_chunk ? on_chunk(text) : true;
        });

        total.prompt_tokens     += result.stats.prompt_tokens;
        total.completion_tokens += result.stats.completion_tokens;
        total.prompt_ms         += result.stats.prompt_ms;
        total.completion_ms     += result.stats.completion_ms;
        result.stats             = total;

        // tool_calls is non-empty iff the reason is ToolCalls, so this is the only branch that
        // has a tool to run; a cancelled or truncated round falls through and returns.
        if (result.reason == FinishReason::ToolCalls && !result.tool_calls.empty()) {
            history.push_back({"assistant", reply, result.tool_calls, ""});
            for (const ToolCall & call : result.tool_calls) {
                history.push_back({"tool", tools.call(call), {}, call.id});
            }
            continue;
        }

        if (!reply.empty()) {
            history.push_back({"assistant", reply});
        }
        return result;
    }

    // Every round spent and the model is still asking. The last round's tools ran and their
    // results are in `history`; the reason stays ToolCalls because no reply to them was generated.
    return result;
}

}  // namespace client
}  // namespace llamad
