/** @file
 * @brief Application-facing synchronous streaming client and reflected tool dispatch.
 *
 * C++ client for the llamad daemon. This header exposes no gRPC or protobuf types,
 * so applications only need this header and the llamad_client library.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <meta>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "llamad/json.h"

namespace llamad {
/// Application API for streaming requests, tool execution and reflected JSON.
namespace client {

/// Metadata for the loaded model and its configured context.
struct ModelInfo {
    std::string description;                ///< Human-readable model architecture and quantization description.
    uint64_t    n_params          = 0;      ///< Number of model parameters.
    uint64_t    size_bytes        = 0;      ///< Size of model tensors in bytes.
    uint32_t    n_ctx             = 0;      ///< Actual context capacity configured by the daemon, in tokens.
    uint32_t    n_ctx_train       = 0;      ///< Training context length recorded by the model, in tokens.
    bool        has_chat_template = false;  ///< Whether the model stores a template; does not guarantee it parses successfully.
};

/// Unset fields use the daemon defaults (see llamad.proto).
struct SamplingParams {
    std::optional<float>     temperature;  ///< Default 0.8; values at or below zero select greedy sampling.
    std::optional<int32_t>   top_k;        ///< Default 40; values at or below zero disable top-k filtering.
    std::optional<float>     top_p;        ///< Default 0.95; values at or above one disable nucleus filtering.
    std::optional<float>     min_p;        ///< Default 0.05; values at or below zero disable min-p filtering.
    std::optional<uint32_t>  seed;         ///< Random seed; unset asks the daemon to choose randomly.
    std::optional<int32_t>   max_tokens;   ///< Completion budget; unset or negative means remaining context capacity.
    std::vector<std::string> stop;         ///< Literal stop strings whose matched bytes are excluded from output.
};

/// A tool the model may call. The daemon executes nothing and keeps no registry: definitions
/// travel with each request, and the caller runs the tool and sends the result back.
struct Tool {
    std::string name;                    ///< Tool function name.
    std::string description;             ///< Model-facing explanation of the tool.
    std::string parameters_json_schema;  ///< JSON Schema object for the arguments, as a JSON string
};

/// One complete model-requested function call, identified for a matching tool reply.
struct ToolCall {
    std::string id;              ///< Call identifier; echo it in the corresponding tool reply.
    std::string name;            ///< Tool function name.
    std::string arguments_json;  ///< Complete JSON arguments; an argument-free call carries an empty object.
};

/// Implementation details of the reflected function adapter.
namespace detail {

/// Synthesize aggregate fields from a function's named, cvref-stripped parameter types.
consteval std::vector<std::meta::info> argument_members(std::meta::info function) {
    std::vector<std::meta::info> members;
    for (std::meta::info parameter : std::meta::parameters_of(function)) {
        members.push_back(std::meta::data_member_spec(
            std::meta::dealias(std::meta::remove_cvref(std::meta::type_of(parameter))),
            {.name = std::meta::identifier_of(parameter)}));
    }
    return members;
}

/// define_aggregate has to be called from a consteval block in the scope that encloses the class
/// it completes, which is why Struct is a member here rather than a class template of its own.
template <std::meta::info Function>
struct ArgumentsHolder {
    /// Aggregate completed by the enclosing compile-time block.
    struct Struct;
    // Doxygen misclassifies the consteval block as a data member; Struct is documented above.
    /// @cond
    consteval { std::meta::define_aggregate(^^Struct, argument_members(Function)); }
    /// @endcond
};

/// A tool function's parameter list as a struct, so the walk over a type's members that
/// json::schema and json::read already do serves functions and structs alike. Synthesised members
/// carry no annotations, so the parameters' descriptions reach the schema on their own.
template <std::meta::info Function>
using Arguments = typename ArgumentsHolder<Function>::Struct;

/// Preserve parameter descriptions separately from the synthesized argument aggregate.
template <std::meta::info Function>
consteval std::span<const char * const> argument_descriptions() {
    std::vector<const char *> texts;
    template for (constexpr std::meta::info parameter :
                  std::define_static_array(std::meta::parameters_of(Function))) {
        texts.push_back(json::detail::description<parameter>());
    }
    return std::define_static_array(texts);
}

/// Whether a function is called on an object, and so needs one to be registered with.
consteval bool is_member_function(std::meta::info function) {
    return std::meta::is_class_member(function) && !std::meta::is_static_member(function);
}

/// Invoke the reflected function with its argument members in parameter order, on the object
/// when it is a member function.
template <std::meta::info Function, typename Args, std::size_t... I, typename... Object>
decltype(auto) apply(const Args & arguments, std::index_sequence<I...>, Object &... object) {
    return std::invoke(&[:Function:], object..., arguments.[: json::detail::fields_of(^^Args)[I] :]...);
}

/// Parses one call's arguments and runs the function, on the object when it is a member
/// function. A std::string result is what the model reads; anything else is written as JSON.
template <std::meta::info Function, typename... Object>
std::string run(const std::string & arguments_json, Object &... object) {
    using Args = Arguments<Function>;

    Args arguments;
    json::read(arguments_json, arguments);

    decltype(auto) result = apply<Function>(
        arguments, std::make_index_sequence<json::detail::fields_of(^^Args).size()>{}, object...);
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(result)>, std::string>) {
        return result;
    } else {
        return json::write(result);
    }
}

/// The definition that travels with a request: the function's name, its description and the
/// schema of its parameter list.
template <std::meta::info Function>
Tool definition() {
    return {std::define_static_string(std::meta::identifier_of(Function)),
            json::detail::description<Function>(),
            json::schema<Arguments<Function>>(argument_descriptions<Function>())};
}

}  // namespace detail

/// The tools an application offers the model, each one an ordinary C++ function. Its identifier
/// is the tool's name, its desc annotations are the descriptions, and its parameter list is the
/// argument schema:
///
///     [[=desc{"Get the current date and time in a given IANA timezone."}]]
///     std::string get_current_time([[=desc{"IANA timezone, e.g. Europe/Paris"}]] std::string timezone);
///
///     ToolSet tools;
///     tools.add<^^get_current_time>();
///
/// A tool that needs state is a member function, registered with the object it is called on:
///
///     tools.add<^^Inventory::take>(inventory);
///
/// A tool returning std::string is handed to the model as it is; any other return type is
/// written as JSON.
class ToolSet {
public:
    /// Register a free or static function, deriving its name, schema and descriptions.
    /// @tparam Function Reflection of a named function with named, JSON-readable parameters.
    /// The result must be std::string or supported by json::write(); void is not supported.
    /// Register before concurrent use; definitions and dispatch share registration order.
    template <std::meta::info Function>
    void add();

    /// Register a member function, called on object each time the model asks for it; the name,
    /// schema and descriptions come from the function as for add().
    /// The ToolSet holds a reference: object must outlive it and every copy of it.
    /// @tparam Function Reflection of a non-static member function of Object or of its base.
    template <std::meta::info Function, typename Object>
    void add(Object & object);

    /// What travels with a request; this is all the daemon ever sees of a tool.
    const std::vector<Tool> & definitions() const { return definitions_; }

    /// Runs the tool the model asked for and returns what the "tool" message should carry. Never
    /// throws: an unknown name, arguments that do not parse and an exception from the tool itself
    /// all come back as {"error":"..."}, which the model can read and recover from.
    std::string call(const ToolCall & call) const;

private:
    /// Type-erased local function accepting JSON arguments and producing model-facing text.
    using Invoke = std::function<std::string(const std::string & arguments_json)>;

    std::vector<Tool>   definitions_;  ///< Wire definitions in registration order.
    std::vector<Invoke> invoke_;       ///< indexed alike
};

template <std::meta::info Function>
void ToolSet::add() {
    static_assert(!detail::is_member_function(Function),
                  "add<F>(): a member function is registered with its object, add<F>(object)");

    definitions_.push_back(detail::definition<Function>());
    invoke_.push_back(&detail::run<Function>);
}

template <std::meta::info Function, typename Object>
void ToolSet::add(Object & object) {
    static_assert(detail::is_member_function(Function),
                  "add<F>(object): F must be a non-static member function");

    definitions_.push_back(detail::definition<Function>());
    invoke_.push_back([&object](const std::string & arguments_json) {
        return detail::run<Function>(arguments_json, object);
    });
}

/// One history turn; the full ordered history travels with every chat request.
struct ChatMessage {
    std::string           role;          ///< "system" | "user" | "assistant" | "tool"
    std::string           content;       ///< Message text; tool results are opaque strings.
    std::vector<ToolCall> tool_calls;    ///< Calls from an assistant turn replayed in history.
    std::string           tool_call_id;  ///< role "tool": the call this message answers
};

/// Reason a generation terminates.
enum class FinishReason {
    Eog,  ///< The model emitted an end-of-generation token.
    Length,  ///< The token budget or context capacity was reached.
    Stop,  ///< A configured stop string matched; its bytes are withheld.
    Cancelled,  ///< The callback or CallOptions::stop requested cancellation.
    ToolCalls  ///< Complete tool calls require client execution.
};

/// Token counts and wall-clock milliseconds for one generation, excluding queue time.
struct GenerateStats {
    int32_t prompt_tokens        = 0;  ///< Prompt length in tokens, including special tokens and cached ones.
    int32_t completion_tokens    = 0;  ///< Generated non-EOG tokens, including any withheld stop or tool markup.
    double  prompt_ms            = 0;  ///< Time decoding the uncached prompt tokens, in ms, including backend synchronization.
    double  completion_ms        = 0;  ///< Generation time in milliseconds, including streaming callback time.
    int32_t cached_prompt_tokens = 0;  ///< Leading prompt tokens the daemon reused from its KV cache instead of decoded.
};

/// Generation outcome; streamed text is delivered separately through the callback.
struct GenerateResult {
    FinishReason          reason;      ///< Reason generation ended.
    GenerateStats         stats;       ///< Counts and timings reported for the generation.
    std::vector<ToolCall> tool_calls;  ///< Calls from an assistant turn replayed in history.
};

/// A reply constrained to T's JSON Schema. value is set when the reply is a complete JSON
/// document: the model finished it, or a stop string matched after it. A reply cut short by the
/// token budget, a cancel, or a stop string matched inside the JSON leaves value empty, and
/// result.reason says which.
template <typename T>
struct Typed {
    std::optional<T> value;   ///< The parsed reply, or unset when the reply is not a complete document.
    GenerateResult   result;  ///< Reason generation ended, plus the stats for it.
};

/// Called with each piece of generated text, in order. Return false to cancel the request.
using ChunkCallback = std::function<bool(const std::string & text)>;

/// Thrown when an RPC fails (daemon unreachable, invalid request, ...).
struct RpcError : std::runtime_error {
    /// Preserve the numeric gRPC status and human-readable error message.
    RpcError(int code, const std::string & message) : std::runtime_error(message), code(code) {}
    int code;  ///< grpc::StatusCode value
};

/// How a call can be called off from outside it, for a caller that runs it on a worker thread or
/// must not wait on it forever. The default has no deadline and is never stopped.
struct CallOptions {
    /// request_stop() on its source, from any thread, cancels the call even while no text is
    /// arriving: a long prompt, a queue behind another client, a tool-call round, a wedged daemon.
    /// A streaming call then returns FinishReason::Cancelled, unless its final chunk had already
    /// arrived, whose reason stands. get_model_info() and tokenize() throw RpcError with CANCELLED (1).
    std::stop_token stop;
    /// Time allowed for each RPC, from its start. When it runs out the call throws RpcError with
    /// DEADLINE_EXCEEDED (4).
    std::optional<std::chrono::milliseconds> timeout;
};

/// Synchronous client for one Unix socket; calls carry all request state.
/// RPC failures throw RpcError. Streaming callbacks run on the calling thread and their
/// exceptions propagate. An empty callback discards text; false requests cancellation, and so
/// does CallOptions::stop from any thread. A cancelled stream can end before final statistics
/// arrive.
class Client {
public:
    /// Return `$XDG_RUNTIME_DIR/llamad.sock`, falling back to `/tmp/llamad-<uid>.sock`.
    static std::string default_socket_path();

    /// Connects lazily; the first RPC fails with RpcError if the daemon is not there.
    explicit Client(const std::string & socket_path = default_socket_path());
    /// Release the channel and stub after outstanding calls have finished.
    ~Client();

    /// Copying is disabled because this object owns its state.
    Client(const Client &)             = delete;
    /// Copying is disabled because this object owns its state.
    Client & operator=(const Client &) = delete;

    /// Fetch metadata for the daemon's loaded model.
    /// @param options Stop token and deadline for the call.
    /// @throws RpcError If the daemon is unreachable or the request fails.
    ModelInfo get_model_info(const CallOptions & options = {});

    /// Tokenize using the daemon's vocabulary.
    /// @param text Input bytes.
    /// @param add_special Add BOS/EOS as expected by the model.
    /// @param parse_special Recognize literal special-token spellings.
    /// @param options Stop token and deadline for the call.
    /// @throws RpcError If the daemon rejects or cannot complete the request.
    std::vector<int32_t> tokenize(const std::string & text,
                                  bool add_special = true,
                                  bool parse_special = false,
                                  const CallOptions & options = {});

    /// Complete a raw prompt without applying the chat template.
    /// Blocks until the stream ends, invoking on_chunk from the calling thread.
    /// If on_chunk returns false or options.stop is requested, the request is cancelled and reason
    /// is Cancelled. Cancelling ends the stream before the daemon's final chunk; stats are
    /// available only if that chunk arrives.
    /// @param prompt Raw model input.
    /// @param params Optional overrides of the daemon sampling defaults.
    /// @param on_chunk Text receiver, or empty to discard text.
    /// @param options Stop token and deadline for the call.
    /// @throws RpcError If the RPC fails for a reason other than requested cancellation,
    ///         DEADLINE_EXCEEDED (4) among them.
    GenerateResult generate(const std::string & prompt,
                            const SamplingParams & params,
                            const ChunkCallback & on_chunk,
                            const CallOptions & options = {});
    /// Generate a single chat turn without tools; messages must contain the full history.
    /// Uses the blocking callback and cancellation behavior of generate().
    /// @throws RpcError For transport failure, invalid history or unavailable chat support.
    GenerateResult chat(const std::vector<ChatMessage> & messages,
                        const SamplingParams & params,
                        const ChunkCallback & on_chunk,
                        const CallOptions & options = {});

    /// Generate one chat turn offering opaque tool definitions; the caller owns execution.
    /// Uses the blocking callback and cancellation behavior of generate(). On
    /// FinishReason::ToolCalls, append an assistant message carrying result.tool_calls
    /// (content = the streamed text, if any), then one "tool" message per call with tool_call_id
    /// set and the tool's result as content, and call chat again.
    GenerateResult chat(const std::vector<ChatMessage> & messages,
                        const std::vector<Tool> & tools,
                        const SamplingParams & params,
                        const ChunkCallback & on_chunk,
                        const CallOptions & options = {});

    /// The same loop, run here. The caller appends the user message to `history` and gets back a
    /// history holding every assistant and "tool" turn the answer took, with `stats` summed over
    /// the rounds. A result of ToolCalls means max_rounds was spent with the model still asking.
    /// The last round's tools are executed even when no generation budget remains.
    /// Once options.stop is requested no further tools run: a round whose calls have not started
    /// ends the loop with Cancelled, its calls unanswered and left out of history.
    /// @param history Full conversation, already including the latest user message; updated in place.
    /// @param tools Registered functions to execute locally.
    /// @param params Sampling overrides applied to each round.
    /// @param on_chunk Text receiver shared across rounds; false cancels the current request.
    /// @param max_rounds Maximum generation requests, including the initial request; must be positive.
    /// @param options Stop token for the whole loop; the timeout applies to each round.
    /// @throws std::invalid_argument If max_rounds is not positive.
    /// @throws RpcError If any request fails; completed turns remain in history.
    GenerateResult chat(std::vector<ChatMessage> & history,
                        const ToolSet & tools,
                        const SamplingParams & params,
                        const ChunkCallback & on_chunk,
                        int max_rounds = 8,
                        const CallOptions & options = {});

    /// One chat turn whose reply is an instance of T: json::schema<T>() travels with the request,
    /// the daemon holds the model to it, and the finished reply is read back with json::read. T is
    /// an aggregate whose members json.h supports; its desc annotations reach the model as property
    /// descriptions. The JSON still streams through on_chunk as it is generated. Tools are not
    /// offered on a typed turn. Messages must contain the full history, as for chat().
    /// The reply is read back when it is a complete JSON document; a reply the token budget, a
    /// cancel or a stop string cut short leaves value empty, with result.reason saying which.
    /// @throws RpcError As chat(). json::Error if a complete reply does not fit T, which the
    ///         grammar makes a disagreement between schema and reader rather than a model mistake.
    template <typename T>
    Typed<T> chat(const std::vector<ChatMessage> & messages,
                  const SamplingParams & params,
                  const ChunkCallback & on_chunk,
                  const CallOptions & options = {});

private:
    /// One Chat request: the tools overload of chat() and the typed chat() are both written over it.
    /// @param messages Complete conversation history in template order.
    /// @param tools Opaque tool definitions offered on this turn; empty offers none.
    /// @param response_json_schema JSON Schema object, as a JSON string, the reply must fit; empty
    ///        leaves the reply unconstrained.
    /// @param params Sampling controls and stop strings for this turn.
    /// @param on_chunk Receives generated text; return false to cancel.
    /// @param options Stop token and deadline for the call.
    GenerateResult chat_request(const std::vector<ChatMessage> & messages,
                                const std::vector<Tool> & tools,
                                const std::string & response_json_schema,
                                const SamplingParams & params,
                                const ChunkCallback & on_chunk,
                                const CallOptions & options);

    /// Dependency-specific state hidden behind the public contract.
    struct Impl;
    std::unique_ptr<Impl> impl_;  ///< Sole owner of the hidden implementation.
};

template <typename T>
Typed<T> Client::chat(const std::vector<ChatMessage> & messages,
                      const SamplingParams & params,
                      const ChunkCallback & on_chunk,
                      const CallOptions & options) {
    static_assert(std::is_aggregate_v<T>, "chat<T>: T must be an aggregate; json::schema describes an object");

    std::string reply;
    Typed<T>    typed;
    // The daemon refuses a schema alongside tools, so a constrained turn offers none.
    typed.result = chat_request(messages, /*tools*/ {}, json::schema<T>(), params, [&](const std::string & text) {
        reply += text;
        return on_chunk ? on_chunk(text) : true;
    }, options);

    // The grammar makes a reply the model finished whole JSON. A stop string is matched on the
    // generated text and removed from it, so it can just as well land inside the document: parse
    // a stopped reply only once it is complete, and otherwise let the reason say why there is no
    // value. A truncated or cancelled reply cannot parse either.
    const bool complete = typed.result.reason == FinishReason::Eog ||
                          (typed.result.reason == FinishReason::Stop && json::detail::Json::accept(reply));
    if (complete) {
        json::read(reply, typed.value.emplace());
    }
    return typed;
}

}  // namespace client
}  // namespace llamad
