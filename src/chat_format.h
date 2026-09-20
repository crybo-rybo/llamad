#pragma once

// Prompt rendering and tool-call parsing, on top of llama.cpp's `common` chat API.
// chat_format.cpp is the only TU in llamad that touches that API — it is unstable, so a submodule
// bump can break at most that one file. No llama.cpp, gRPC or protobuf types may appear here.

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace llamad {

// A tool offered to the model for one request. The daemon never validates or executes tools:
// they are opaque data, like the message history.
struct Tool {
    std::string name;
    std::string description;
    std::string parameters_json_schema;   // JSON Schema object, as a JSON string
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

struct ChatMessage {
    std::string           role;           // "system" | "user" | "assistant" | "tool"
    std::string           content;
    std::vector<ToolCall> tool_calls;     // assistant turns being replayed
    std::string           tool_call_id;   // role "tool": which call this is the result of
};

// Grammar constraint for the sampler, as plain strings; the engine resolves tokens.
struct GrammarSpec {
    std::string              grammar;            // GBNF; empty = unconstrained
    bool                     lazy = false;       // only active once a trigger fires
    std::vector<std::string> trigger_patterns;   // regexes (PATTERN as-is, PATTERN_FULL anchored ^...$)
    std::vector<std::string> trigger_words;      // literal WORD triggers, NOT regex-escaped; the engine turns a
                                                 // word into a token trigger if it is a single token, else escapes it
};

// Opaque parser state shared by a RenderedChat and every Stream made from it.
struct ParseState;

struct RenderedChat {
    std::string              prompt;
    GrammarSpec              grammar;
    std::vector<std::string> preserved_tokens;   // special tokens whose text must be rendered in the output
    std::vector<std::string> additional_stops;

    // Everything Stream needs to parse the generated text (chat format, serialized PEG parser,
    // generation prompt). Opaque and immutable, so a RenderedChat is cheap to copy and share.
    std::shared_ptr<const ParseState> parse_state;
};

// Thrown for caller errors (unparsable template, bad tool schema JSON, template rejects the messages).
struct ChatFormatError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// One per daemon; immutable after construction, so render()/stream() are safe to call concurrently.
class ChatFormat {
public:
    // Throws ChatFormatError if the template cannot be parsed.
    ChatFormat(const std::string & template_source, const std::string & bos_token, const std::string & eos_token);
    ~ChatFormat();

    ChatFormat(const ChatFormat &)             = delete;
    ChatFormat & operator=(const ChatFormat &) = delete;

    // Throws ChatFormatError for caller errors (bad tool schema JSON, template rejects the messages, ...).
    RenderedChat render(const std::vector<ChatMessage> & messages, const std::vector<Tool> & tools) const;

    // One per request; not thread-safe. Tool-call markup never appears in anything it returns.
    class Stream {
    public:
        ~Stream();

        Stream(Stream &&) noexcept;
        Stream & operator=(Stream &&) noexcept;

        Stream(const Stream &)             = delete;
        Stream & operator=(const Stream &) = delete;

        // Feed generated text; returns the newly visible assistant content ("" while inside a tool call).
        std::string push(const std::string & text);

        struct Final {
            std::string           content_tail;
            std::vector<ToolCall> tool_calls;
        };

        // Generation ended: final (non-partial) parse. content_tail = visible content not yet
        // returned by push(). Never throws. A call the model was cut off in the middle of yields no
        // tool calls at all rather than a fragment: the content parsed so far is still returned, and
        // its markup is not. The finish reason (LENGTH / CANCELLED) is what tells the caller why.
        Final finish();

    private:
        friend class ChatFormat;

        struct Impl;
        explicit Stream(std::unique_ptr<Impl> impl);

        std::unique_ptr<Impl> impl_;
    };

    Stream stream(const RenderedChat & rendered) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace llamad
