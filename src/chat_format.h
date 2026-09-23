/** @file
 * @brief Model-template rendering and incremental tool-call parsing contracts.
 *
 * Prompt rendering and tool-call parsing, on top of llama.cpp's `common` chat API.
 * chat_format.cpp is the only TU in llamad that touches that API — it is unstable, so a submodule
 * bump can break at most that one file. No llama.cpp, gRPC or protobuf types may appear here.
 */

#pragma once

#include "engine.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace llamad {

/// A tool offered for one request; the daemon formats its schema but never executes it.
struct Tool {
    std::string name;                    ///< Tool function name.
    std::string description;             ///< Model-facing explanation of the tool.
    std::string parameters_json_schema;  ///< JSON Schema object, as a JSON string
};

/// One complete model-requested function call, identified for a matching tool reply.
struct ToolCall {
    std::string id;              ///< Call identifier; echo it in the corresponding tool reply.
    std::string name;            ///< Tool function name.
    std::string arguments_json;  ///< Complete JSON arguments; an argument-free call carries an empty object.
};

/// One history turn; the full ordered history travels with every chat request.
struct ChatMessage {
    std::string           role;          ///< "system" | "user" | "assistant" | "tool"
    std::string           content;       ///< Message text; tool results are opaque strings.
    std::vector<ToolCall> tool_calls;    ///< Calls from an assistant turn replayed in history.
    std::string           tool_call_id;  ///< role "tool": which call this is the result of
};

/// Opaque parser state shared by a RenderedChat and every Stream made from it.
struct ParseState;

/// A rendered prompt and the matching grammar and immutable parser state.
struct RenderedChat {
    std::string              prompt;            ///< Ready-to-tokenize model input including the generation prefix.
    GrammarSpec              grammar;           ///< Sampler grammar and activation triggers for this prompt.
    std::vector<std::string> preserved_tokens;  ///< special tokens whose text must be rendered in the output
    std::vector<std::string> additional_stops;  ///< Template-specific stops to append to the request stop strings.

    /// Everything Stream needs to parse the generated text (chat format, serialized PEG parser,
    /// generation prompt). Opaque and immutable, so a RenderedChat is cheap to copy and share.
    std::shared_ptr<const ParseState> parse_state;
};

/// Thrown for caller errors (unparsable template, bad tool schema JSON, template rejects the messages).
struct ChatFormatError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// One per daemon; immutable after construction, so render()/stream() are safe to call concurrently.
class ChatFormat {
public:
    /// Compile a model Jinja template with its BOS and EOS token text.
    /// @param template_source Nonempty Jinja source stored by the model.
    /// @param bos_token Beginning-of-sequence token text, or empty if the model has none.
    /// @param eos_token End-of-sequence token text, or empty if the model has none.
    /// @throws ChatFormatError If the template is empty or cannot be parsed.
    ChatFormat(const std::string & template_source, const std::string & bos_token, const std::string & eos_token);
    /// Release the compiled templates; streams retain their own shared parser state.
    ~ChatFormat();

    /// Copying is disabled because this object owns its state.
    ChatFormat(const ChatFormat &)             = delete;
    /// Copying is disabled because this object owns its state.
    ChatFormat & operator=(const ChatFormat &) = delete;

    /// Render complete history and request-local tool definitions through the model template.
    /// @param messages Conversation in template order, including prior calls and tool replies.
    /// @param tools Opaque tool definitions offered during this generation; empty disables tools.
    /// @param response_json_schema JSON Schema object, as a JSON string, the reply must fit; empty
    ///        leaves the reply unconstrained. The JSON reaches the caller as ordinary content.
    /// @return Prompt, grammar, stops and shared parser state for a matching output stream.
    /// @throws ChatFormatError For invalid schema JSON, rejected messages or parser setup failure;
    ///         also if response_json_schema is not a JSON object, is the empty object, or is
    ///         combined with tools.
    RenderedChat render(const std::vector<ChatMessage> & messages, const std::vector<Tool> & tools,
                        const std::string & response_json_schema) const;

    /// One parser per request; not thread-safe. Recognized tool-call markup is withheld.
    class Stream {
    public:
        /// Release this request's incremental parser state.
        ~Stream();

        /// Transfer a request parser; the source is only suitable for destruction or assignment.
        Stream(Stream &&) noexcept;
        /// Replace this request parser by transferring another stream's state.
        Stream & operator=(Stream &&) noexcept;

        /// Copying is disabled because this object owns its state.
        Stream(const Stream &)             = delete;
        /// Copying is disabled because this object owns its state.
        Stream & operator=(const Stream &) = delete;

        /// Feed generated text; returns the newly visible assistant content ("" while inside a tool call).
        /// @param text Next fragment from the engine, in generation order.
        std::string push(const std::string & text);

        /// Final visible tail and complete calls from the accumulated output.
        struct Final {
            std::string           content_tail;  ///< Visible text not returned by push().
            std::vector<ToolCall> tool_calls;    ///< Complete calls; empty if any parsed call is incomplete.
        };

        /// Generation ended: final (non-partial) parse. content_tail = visible content not yet
        /// returned by push(). Parser errors are contained. An incomplete call suppresses all
        /// calls while preserving visible content. A rejected final parse falls back to the
        /// un-emitted raw suffix if it can be located; that fallback can include markup.
        /// Call once after generation. The service discards calls on LENGTH or CANCELLED.
        Final finish();

    private:
        friend class ChatFormat;

        /// Dependency-specific state hidden behind the public contract.
        struct Impl;
        /// Take ownership of parser state prepared by ChatFormat::stream().
        explicit Stream(std::unique_ptr<Impl> impl);

        std::unique_ptr<Impl> impl_;  ///< Sole owner of the hidden implementation.
    };

    /// Create an independent parser for output generated from rendered.prompt.
    /// @throws ChatFormatError If rendered has no parser state from render().
    Stream stream(const RenderedChat & rendered) const;

private:
    /// Dependency-specific state hidden behind the public contract.
    struct Impl;
    std::unique_ptr<Impl> impl_;  ///< Sole owner of the hidden implementation.
};

}  // namespace llamad
