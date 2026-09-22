/** @file
 * @brief The sole adapter to llama.cpp common templates, grammars and chat parsers.
 */

#include "chat_format.h"

#include "chat.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace llamad {

namespace {

// ---------------------------------------------------------------------------
// `common` logging: the same policy as engine.cpp — only warnings and errors,
// and only on stderr. Left at llama.cpp's default (INFO, partly on stdout) the
// chat layer would write template chatter into the daemon's stdout.
// ---------------------------------------------------------------------------

/// Suppress common-library informational output so daemon stdout stays empty.
void init_common_log_once() {
    static std::once_flag once;
    std::call_once(once, [] { common_log_set_verbosity_thold(LOG_LEVEL_WARN); });
}

// ---------------------------------------------------------------------------
// Tool call ids
// ---------------------------------------------------------------------------

/// Same shape as llama-server's: 32 random alphanumeric characters. Ids only have to be
/// unique within one response, so a per-call thread-local generator is plenty.
std::string gen_tool_call_id() {
    static const char alnum[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<size_t> pick(0, sizeof(alnum) - 2);

    std::string id(32, ' ');
    for (char & c : id) {
        c = alnum[pick(rng)];
    }
    return id;
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

/// Translate one history turn without exposing common types in the header.
common_chat_msg to_common(const ChatMessage & message) {
    common_chat_msg msg;
    msg.role         = message.role;
    msg.content      = message.content;
    msg.tool_call_id = message.tool_call_id;

    msg.tool_calls.reserve(message.tool_calls.size());
    for (const ToolCall & call : message.tool_calls) {
        common_chat_tool_call out;
        out.id        = call.id;
        out.name      = call.name;
        out.arguments = call.arguments_json;
        msg.tool_calls.push_back(std::move(out));
    }
    return msg;
}

/// Validate schema syntax and provide common with an object for argument-free tools.
common_chat_tool to_common(const Tool & tool) {
    // common parses `parameters` with nlohmann and lets the exception escape into the middle of
    // template rendering; validate here instead so the caller gets a useful message.
    const std::string & schema = tool.parameters_json_schema;
    if (!schema.empty() && !nlohmann::json::accept(schema)) {
        throw ChatFormatError("tool '" + tool.name + "': parameters_json_schema is not valid JSON");
    }

    common_chat_tool out;
    out.name        = tool.name;
    out.description = tool.description;
    // An empty schema means "no arguments"; common expects a JSON object either way.
    out.parameters  = schema.empty() ? "{}" : schema;
    return out;
}

/// Convert a parsed call, spelling an argument-free call as the JSON object {}.
ToolCall from_common(const common_chat_tool_call & call) {
    ToolCall out;
    out.id             = call.id;
    out.name           = call.name;
    // A tool taking no arguments parses to an empty string; the wire contract is always JSON.
    out.arguments_json = call.arguments.empty() ? "{}" : call.arguments;
    return out;
}

/// The PEG parsers are lenient, so text cut off inside a tool call still parses: the call comes back
/// with a half-finished name or a fragment of its arguments ("{"). Complete calls are the only ones
/// worth handing to a client, and valid JSON arguments are the test for that.
bool is_complete(const common_chat_tool_call & call) {
    return !call.name.empty() && (call.arguments.empty() || nlohmann::json::accept(call.arguments));
}

/// The engine resolves trigger words against the vocab, so words stay raw here and only regexes
/// get the treatment common/sampling.cpp:220-255 applies. TOKEN triggers cannot occur: they are
/// produced from a vocab, and a ChatFormat is built from a template string with no model.
void split_triggers(const std::vector<common_grammar_trigger> & triggers, GrammarSpec & spec) {
    for (const common_grammar_trigger & trigger : triggers) {
        switch (trigger.type) {
            case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                spec.trigger_words.push_back(trigger.value);
                break;
            case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                spec.trigger_patterns.push_back(trigger.value);
                break;
            case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: {
                const std::string & pattern = trigger.value;
                std::string         anchored = "^$";
                if (!pattern.empty()) {
                    anchored = (pattern.front() != '^' ? "^" : "") + pattern +
                               (pattern.back() != '$' ? "$" : "");
                }
                spec.trigger_patterns.push_back(std::move(anchored));
                break;
            }
            case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                // No vocab here, so the token id is meaningless; the literal is the best we have.
                if (!trigger.value.empty()) {
                    spec.trigger_words.push_back(trigger.value);
                }
                break;
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Parser state
// ---------------------------------------------------------------------------

/// Everything common_chat_parse needs, built once per render and shared by every Stream.
struct ParseState {
    common_chat_parser_params params;  ///< Format identifier, parser and generation prefix fixed by a render.
};

// ---------------------------------------------------------------------------
// ChatFormat
// ---------------------------------------------------------------------------

/// Own immutable compiled templates and their parallel-call capability.
struct ChatFormat::Impl {
    common_chat_templates_ptr templates;                    ///< Owned compiled model templates.
    bool                      parallel_tool_calls = false;  ///< Whether the template supports more than one call in a turn.
};

ChatFormat::ChatFormat(const std::string & template_source,
                       const std::string & bos_token,
                       const std::string & eos_token)
    : impl_(new Impl()) {
    init_common_log_once();

    if (template_source.empty()) {
        throw ChatFormatError("the model has no chat template");
    }

    try {
        // A null model is explicitly supported when the template is given as a string: bos/eos are
        // then taken verbatim from the overrides instead of the vocab (common/chat.cpp:808-834).
        impl_->templates = common_chat_templates_init(/*model*/ nullptr, template_source, bos_token, eos_token);
    } catch (const std::exception & e) {
        throw ChatFormatError(std::string("failed to parse the chat template: ") + e.what());
    }

    // llama-server defaults this to the template's own capability (tools/server/server-common.cpp:1295);
    // for Qwen2.5 it is true, which is what makes two <tool_call> blocks in one turn parse as two calls.
    const std::map<std::string, bool> caps = common_chat_templates_get_caps(impl_->templates.get());
    const auto                        it   = caps.find("supports_parallel_tool_calls");
    impl_->parallel_tool_calls             = it != caps.end() && it->second;
}

ChatFormat::~ChatFormat() = default;

RenderedChat ChatFormat::render(const std::vector<ChatMessage> & messages, const std::vector<Tool> & tools,
                                const std::string & response_json_schema) const {
    if (!response_json_schema.empty()) {
        if (!tools.empty()) {
            // common's parser matches the response format before tools, so under a response schema
            // a tool call would never be parsed as one. Refusing beats ignoring one of the two.
            throw ChatFormatError("tools and response_json_schema cannot be used together");
        }
        // common parses the schema in the middle of template rendering, and silently ignores it
        // unless it is an object; both are better caught here, where the message can say what is wrong.
        const nlohmann::json schema =
            nlohmann::json::parse(response_json_schema, /*cb*/ nullptr, /*allow_exceptions*/ false);
        if (schema.is_discarded()) {
            throw ChatFormatError("response_json_schema is not valid JSON");
        }
        if (!schema.is_object()) {
            throw ChatFormatError("response_json_schema is not a JSON object");
        }
        // common only builds a response grammar from a nonempty object, so `{}` would leave the
        // reply unconstrained prose while the request promised JSON. A schema that constrains
        // nothing has no useful reading here, so say so rather than quietly dropping it.
        if (schema.empty()) {
            throw ChatFormatError("response_json_schema is an empty object, which constrains nothing");
        }
    }

    common_chat_templates_inputs inputs;
    inputs.use_jinja             = true;
    inputs.add_generation_prompt = true;
    inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_AUTO;
    inputs.parallel_tool_calls   = impl_->parallel_tool_calls;
    inputs.reasoning_format      = COMMON_REASONING_FORMAT_NONE;   // thinking text stays in the content
    inputs.json_schema           = response_json_schema;           // empty leaves the reply unconstrained

    inputs.messages.reserve(messages.size());
    for (const ChatMessage & message : messages) {
        inputs.messages.push_back(to_common(message));
    }
    inputs.tools.reserve(tools.size());
    for (const Tool & tool : tools) {
        inputs.tools.push_back(to_common(tool));
    }

    common_chat_params params;
    try {
        params = common_chat_templates_apply(impl_->templates.get(), inputs);
    } catch (const std::exception & e) {
        throw ChatFormatError(std::string("failed to render the chat template: ") + e.what());
    }

    RenderedChat rendered;
    rendered.prompt           = std::move(params.prompt);
    rendered.grammar.grammar  = std::move(params.grammar);
    rendered.grammar.lazy     = params.grammar_lazy;
    rendered.grammar.prefill  = params.generation_prompt;
    rendered.preserved_tokens = std::move(params.preserved_tokens);
    rendered.additional_stops = std::move(params.additional_stops);
    split_triggers(params.grammar_triggers, rendered.grammar);

    auto state                       = std::make_shared<ParseState>();
    state->params.format             = params.format;
    state->params.generation_prompt  = params.generation_prompt;
    state->params.reasoning_format   = COMMON_REASONING_FORMAT_NONE;
    state->params.parse_tool_calls   = true;
    if (!params.parser.empty()) {
        // The serialized PEG parser: without it common_chat_parse falls back to "everything is
        // content" and tool-call markup would be streamed straight to the client.
        try {
            state->params.parser.load(params.parser);
        } catch (const std::exception & e) {
            throw ChatFormatError(std::string("failed to load the chat parser: ") + e.what());
        }
    }
    rendered.parse_state = std::move(state);

    return rendered;
}

// ---------------------------------------------------------------------------
// ChatFormat::Stream
// ---------------------------------------------------------------------------

/// Accumulate one request's output and track exactly which visible bytes were delivered.
struct ChatFormat::Stream::Impl {
    std::shared_ptr<const ParseState> state;  ///< Shared immutable parser state from the rendered request.

    std::string              accumulated;  ///< every byte pushed, verbatim
    common_chat_msg          parsed;       ///< last successful parse
    std::vector<std::string> call_ids;     ///< ids handed out so far, by call index
    std::string              emitted;      ///< exactly what push() has returned so far

    /// Result of reparsing the accumulated output as partial or final text.
    struct Advance {
        bool        ok = false;  ///< the parser accepted the text
        std::string delta;       ///< content that became visible since the last parse
    };

    /// Re-parses `accumulated`. A partial parse that fails just means "not enough input": the text
    /// stays accumulated and the next push tries again. A non-partial one that fails means the
    /// output does not match the format at all.
    Advance advance(bool is_partial) {
        Advance result;

        common_chat_msg next;
        try {
            next = common_chat_parse(accumulated, is_partial, state->params);
        } catch (const std::exception &) {
            return result;
        }
        result.ok = true;

        if (next.empty()) {
            return result;
        }

        next.set_tool_call_ids(call_ids, gen_tool_call_id);
        parsed = std::move(next);

        // Diff against what push() actually returned, not against the previous parse: the parser is
        // allowed to change its mind about text it has already classified (a provisional tool call
        // that later turns out to be ordinary prose reappears in `content`), and only the emitted
        // text says what the caller has really seen. common_chat_msg_diff::compute_diffs rejects
        // exactly those re-interpretations, which would lose the recovered bytes.
        const std::string & content = parsed.content;
        if (content.size() > emitted.size() && content.compare(0, emitted.size(), emitted) == 0) {
            result.delta = content.substr(emitted.size());
            emitted += result.delta;
        }
        // Otherwise the content shrank or diverged from what was already delivered. Nothing can be
        // un-said, so emit nothing and keep the delivered text as the baseline; a later parse that
        // re-extends it past `emitted` resumes streaming. Neither has been observed at this pin:
        // ambiguous text is held back until the parser can classify it.
        return result;
    }

    /// Where the emitted content ends in the raw output. The parser returns content, not offsets,
    /// and content is a verbatim substring of the raw text (with tools, leading whitespace is
    /// trimmed, so it is not always a prefix) -- hence a search rather than an offset.
    /// npos = the emitted text is not a substring of the raw output, so the two cannot be aligned.
    size_t raw_offset_after_emitted() const {
        if (emitted.empty()) {
            return 0;
        }
        const size_t pos = accumulated.find(emitted);
        return pos == std::string::npos ? std::string::npos : pos + emitted.size();
    }
};

ChatFormat::Stream::Stream(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

ChatFormat::Stream::~Stream() = default;

ChatFormat::Stream::Stream(Stream &&) noexcept = default;

ChatFormat::Stream & ChatFormat::Stream::operator=(Stream &&) noexcept = default;

std::string ChatFormat::Stream::push(const std::string & text) {
    if (text.empty()) {
        return {};
    }
    impl_->accumulated += text;

    return impl_->advance(/*is_partial*/ true).delta;
}

ChatFormat::Stream::Final ChatFormat::Stream::finish() {
    Final final;

    const Impl::Advance advanced = impl_->advance(/*is_partial*/ false);
    final.content_tail           = advanced.delta;

    if (!advanced.ok) {
        // The parser rejected the output outright. There is no parse to take content from, so the
        // raw remainder is the only honest answer: nothing generated is silently dropped. This is
        // the one path on which tool-call markup can reach the caller, and it is unreachable for
        // the formats at this pin (the PEG parsers are lenient and fall back to pure content).
        const size_t offset = impl_->raw_offset_after_emitted();
        if (offset != std::string::npos && offset < impl_->accumulated.size()) {
            final.content_tail = impl_->accumulated.substr(offset);
            impl_->emitted += final.content_tail;
        }
        // offset == npos: the emitted text cannot be located in the raw output, so the remainder
        // cannot be identified either. Emitting the whole raw text would duplicate what push()
        // already returned, which is worse than dropping it.
        return final;
    }

    for (const common_chat_tool_call & call : impl_->parsed.tool_calls) {
        if (!is_complete(call)) {
            // Generation ended inside the markup of a call (a length limit or a cancellation): the
            // name or the arguments are a fragment. All or nothing — a half-parsed call is worth
            // less to a client than no call at all, and the finish reason already says the response
            // was cut short. The content parsed so far still streams; the markup never does.
            final.tool_calls.clear();
            return final;
        }
    }

    final.tool_calls.reserve(impl_->parsed.tool_calls.size());
    for (const common_chat_tool_call & call : impl_->parsed.tool_calls) {
        final.tool_calls.push_back(from_common(call));
    }
    return final;
}

ChatFormat::Stream ChatFormat::stream(const RenderedChat & rendered) const {
    if (!rendered.parse_state) {
        throw ChatFormatError("RenderedChat has no parser state (not produced by ChatFormat::render)");
    }

    auto impl   = std::unique_ptr<Stream::Impl>(new Stream::Impl());
    impl->state = rendered.parse_state;
    return Stream(std::move(impl));
}

}  // namespace llamad
