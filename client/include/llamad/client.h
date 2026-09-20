#pragma once

// C++ client for the llamad daemon. This header exposes no gRPC or protobuf types,
// so applications only need this header and the llamad_client library.

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace llamad {
namespace client {

struct ModelInfo {
    std::string description;
    uint64_t    n_params          = 0;
    uint64_t    size_bytes        = 0;
    uint32_t    n_ctx             = 0;
    uint32_t    n_ctx_train       = 0;
    bool        has_chat_template = false;
};

// Unset fields use the daemon defaults (see llamad.proto).
struct SamplingParams {
    std::optional<float>     temperature;
    std::optional<int32_t>   top_k;
    std::optional<float>     top_p;
    std::optional<float>     min_p;
    std::optional<uint32_t>  seed;
    std::optional<int32_t>   max_tokens;
    std::vector<std::string> stop;
};

// A tool the model may call. The daemon executes nothing and keeps no registry: definitions
// travel with each request, and the caller runs the tool and sends the result back.
struct Tool {
    std::string name;
    std::string description;
    std::string parameters_json_schema;  // JSON Schema object for the arguments, as a JSON string
};

struct ToolCall {
    std::string id;              // assigned by the daemon; echo it back as tool_call_id
    std::string name;
    std::string arguments_json;  // complete, valid JSON object
};

struct ChatMessage {
    std::string           role;  // "system" | "user" | "assistant" | "tool"
    std::string           content;
    std::vector<ToolCall> tool_calls;    // assistant turns being replayed from history
    std::string           tool_call_id;  // role "tool": the call this message answers
};

enum class FinishReason { Eog, Length, Stop, Cancelled, ToolCalls };

struct GenerateStats {
    int32_t prompt_tokens     = 0;
    int32_t completion_tokens = 0;
    double  prompt_ms         = 0;
    double  completion_ms     = 0;
};

struct GenerateResult {
    FinishReason          reason;
    GenerateStats         stats;
    std::vector<ToolCall> tool_calls;  // set when reason == ToolCalls
};

// Called with each piece of generated text, in order. Return false to cancel the request.
using ChunkCallback = std::function<bool(const std::string & text)>;

// Thrown when an RPC fails (daemon unreachable, invalid request, ...).
struct RpcError : std::runtime_error {
    RpcError(int code, const std::string & message) : std::runtime_error(message), code(code) {}
    int code;  // grpc::StatusCode value
};

class Client {
public:
    // Returns $XDG_RUNTIME_DIR/llamad.sock, falling back to /tmp/llamad-<uid>.sock.
    static std::string default_socket_path();

    // Connects lazily; the first RPC fails with RpcError if the daemon is not there.
    explicit Client(const std::string & socket_path = default_socket_path());
    ~Client();

    Client(const Client &)             = delete;
    Client & operator=(const Client &) = delete;

    ModelInfo get_model_info();

    std::vector<int32_t> tokenize(const std::string & text, bool add_special = true, bool parse_special = false);

    // Both block until the stream ends, invoking on_chunk from the calling thread.
    // If on_chunk returns false the request is cancelled and reason is Cancelled.
    GenerateResult generate(const std::string & prompt, const SamplingParams & params, const ChunkCallback & on_chunk);
    GenerateResult chat(const std::vector<ChatMessage> & messages, const SamplingParams & params, const ChunkCallback & on_chunk);

    // Same, with tools the model may call. The caller owns the loop: on FinishReason::ToolCalls,
    // append an assistant message carrying result.tool_calls (content = the streamed text, if any),
    // then one "tool" message per call with tool_call_id set and the tool's result as content,
    // and call chat again.
    GenerateResult chat(const std::vector<ChatMessage> & messages,
                        const std::vector<Tool> & tools,
                        const SamplingParams & params,
                        const ChunkCallback & on_chunk);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace client
}  // namespace llamad
