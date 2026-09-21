#pragma once

// C++ client for the llamad daemon. This header exposes no gRPC or protobuf types,
// so applications only need this header and the llamad_client library.

#include <cstdint>
#include <functional>
#include <memory>
#include <meta>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "llamad/json.h"

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

namespace detail {

consteval std::vector<std::meta::info> argument_members(std::meta::info function) {
    std::vector<std::meta::info> members;
    for (std::meta::info parameter : std::meta::parameters_of(function)) {
        members.push_back(std::meta::data_member_spec(
            std::meta::dealias(std::meta::remove_cvref(std::meta::type_of(parameter))),
            {.name = std::meta::identifier_of(parameter)}));
    }
    return members;
}

// define_aggregate has to be called from a consteval block in the scope that encloses the class
// it completes, which is why Struct is a member here rather than a class template of its own.
template <std::meta::info Function>
struct ArgumentsHolder {
    struct Struct;
    consteval { std::meta::define_aggregate(^^Struct, argument_members(Function)); }
};

// A tool function's parameter list as a struct, so the walk over a type's members that
// json::schema and json::read already do serves functions and structs alike. Synthesised members
// carry no annotations, so the parameters' descriptions reach the schema on their own.
template <std::meta::info Function>
using Arguments = typename ArgumentsHolder<Function>::Struct;

template <std::meta::info Function>
consteval std::span<const char * const> argument_descriptions() {
    std::vector<const char *> texts;
    template for (constexpr std::meta::info parameter :
                  std::define_static_array(std::meta::parameters_of(Function))) {
        texts.push_back(json::detail::description<parameter>());
    }
    return std::define_static_array(texts);
}

template <std::meta::info Function, typename Args, std::size_t... I>
decltype(auto) apply(const Args & arguments, std::index_sequence<I...>) {
    return [:Function:](arguments.[: json::detail::fields_of(^^Args)[I] :]...);
}

// Parses one call's arguments and runs the function. A std::string result is what the model
// reads; anything else is written as JSON.
template <std::meta::info Function>
std::string run(const std::string & arguments_json) {
    using Args = Arguments<Function>;

    Args arguments;
    json::read(arguments_json, arguments);

    decltype(auto) result =
        apply<Function>(arguments, std::make_index_sequence<json::detail::fields_of(^^Args).size()>{});
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(result)>, std::string>) {
        return result;
    } else {
        return json::write(result);
    }
}

}  // namespace detail

// The tools an application offers the model, each one an ordinary C++ function. Its identifier
// is the tool's name, its desc annotations are the descriptions, and its parameter list is the
// argument schema:
//
//     [[=desc{"Get the current date and time in a given IANA timezone."}]]
//     std::string get_current_time([[=desc{"IANA timezone, e.g. Europe/Paris"}]] std::string timezone);
//
//     ToolSet tools;
//     tools.add<^^get_current_time>();
//
// Free and static functions only. A tool returning std::string is handed to the model as it is;
// any other return type is written as JSON.
class ToolSet {
public:
    template <std::meta::info Function>
    void add();

    // What travels with a request; this is all the daemon ever sees of a tool.
    const std::vector<Tool> & definitions() const { return definitions_; }

    // Runs the tool the model asked for and returns what the "tool" message should carry. Never
    // throws: an unknown name, arguments that do not parse and an exception from the tool itself
    // all come back as {"error":"..."}, which the model can read and recover from.
    std::string call(const ToolCall & call) const;

private:
    using Invoke = std::function<std::string(const std::string & arguments_json)>;

    std::vector<Tool>   definitions_;
    std::vector<Invoke> invoke_;  // indexed alike
};

template <std::meta::info Function>
void ToolSet::add() {
    using Args = detail::Arguments<Function>;

    definitions_.push_back({std::define_static_string(std::meta::identifier_of(Function)),
                            json::detail::description<Function>(),
                            json::schema<Args>(detail::argument_descriptions<Function>())});
    invoke_.push_back(&detail::run<Function>);
}

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
    // If on_chunk returns false the request is cancelled and reason is Cancelled. Cancelling ends
    // the stream before the daemon's final chunk, so that result carries no stats.
    GenerateResult generate(const std::string & prompt, const SamplingParams & params, const ChunkCallback & on_chunk);
    GenerateResult chat(const std::vector<ChatMessage> & messages, const SamplingParams & params, const ChunkCallback & on_chunk);

    // Same, with tools the model may call, in the form where the caller owns the loop: on
    // FinishReason::ToolCalls, append an assistant message carrying result.tool_calls
    // (content = the streamed text, if any), then one "tool" message per call with tool_call_id
    // set and the tool's result as content, and call chat again.
    GenerateResult chat(const std::vector<ChatMessage> & messages,
                        const std::vector<Tool> & tools,
                        const SamplingParams & params,
                        const ChunkCallback & on_chunk);

    // The same loop, run here. The caller appends the user message to `history` and gets back a
    // history holding every assistant and "tool" turn the answer took, with `stats` summed over
    // the rounds. A result of ToolCalls means max_rounds was spent with the model still asking.
    // max_rounds must be positive; anything else throws std::invalid_argument.
    GenerateResult chat(std::vector<ChatMessage> & history,
                        const ToolSet & tools,
                        const SamplingParams & params,
                        const ChunkCallback & on_chunk,
                        int max_rounds = 8);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace client
}  // namespace llamad
