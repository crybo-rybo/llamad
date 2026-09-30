/** @file
 * @brief Model-free template rendering and incremental tool-call parsing regressions.
 *
 * Parser and renderer tests for src/chat_format.cpp. No model file: ChatFormat is built from a
 * chat template string, so the Qwen2.5 template shipped with llama.cpp is enough.
 */

#include "chat_format.h"
#include "check.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool contains(const std::string & haystack, const std::string & needle) {
    return haystack.find(needle) != std::string::npos;
}

std::string read_file(const char * path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(1);
    }
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

// Every character of JSON that is not inside a string literal, with whitespace dropped. Enough to
// compare small argument objects without pulling a JSON library into the test.
std::string normalize_json(const std::string & json) {
    std::string out;
    bool        in_string = false;
    bool        escaped   = false;
    for (char c : json) {
        if (in_string) {
            out += c;
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            out += c;
        } else if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            out += c;
        }
    }
    return out;
}

llamad::ChatFormat make_format() {
    // The template's `eos_token` is ChatML's; there is no BOS in Qwen2.5.
    return llamad::ChatFormat(read_file(QWEN25_TEMPLATE_PATH), /*bos*/ "", /*eos*/ "<|im_end|>");
}

// Qwen3.5 opens a <think> block in its generation prompt unless thinking is turned off.
llamad::ChatFormat make_thinking_format() {
    return llamad::ChatFormat(read_file(QWEN35_TEMPLATE_PATH), /*bos*/ "", /*eos*/ "<|im_end|>");
}

llamad::Tool weather_tool() {
    llamad::Tool tool;
    tool.name                   = "get_weather";
    tool.description            = "Get the current weather in a city";
    tool.parameters_json_schema = R"({"type":"object","properties":{"city":{"type":"string",)"
                                  R"("description":"City name"}},"required":["city"]})";
    return tool;
}

// What the client sends for a function with an empty parameter list.
llamad::Tool clock_tool() {
    llamad::Tool tool;
    tool.name                   = "get_time";
    tool.description            = "Say what time it is";
    tool.parameters_json_schema = R"({"type":"object","properties":{},"required":[]})";
    return tool;
}

std::string spam_schema() {
    return R"({"type":"object","properties":{"spam":{"type":"boolean"},)"
           R"("reasons":{"type":"array","items":{"type":"string"}}},"required":["spam","reasons"]})";
}

llamad::ChatMessage user(const std::string & text) {
    llamad::ChatMessage msg;
    msg.role    = "user";
    msg.content = text;
    return msg;
}

// Pushes `text` one byte at a time and returns everything push() handed back.
std::string push_bytewise(llamad::ChatFormat::Stream & stream, const std::string & text) {
    std::string seen;
    for (char c : text) {
        seen += stream.push(std::string(1, c));
    }
    return seen;
}

struct Run {
    std::string                    streamed;   // concatenated push() results
    std::string                    tail;       // finish().content_tail
    std::vector<llamad::ToolCall>  calls;

    std::string content() const { return streamed + tail; }
};

// Feeds `text` in fixed-size chunks; chunk_size 0 means "the whole string in one push".
Run run_stream(const llamad::ChatFormat & format, const llamad::RenderedChat & rendered,
               const std::string & text, size_t chunk_size) {
    llamad::ChatFormat::Stream stream = format.stream(rendered);

    Run run;
    if (chunk_size == 0) {
        run.streamed = stream.push(text);
    } else {
        for (size_t i = 0; i < text.size(); i += chunk_size) {
            run.streamed += stream.push(text.substr(i, chunk_size));
        }
    }

    llamad::ChatFormat::Stream::Final final = stream.finish();
    run.tail                                = std::move(final.content_tail);
    run.calls                               = std::move(final.tool_calls);
    return run;
}

// ---------------------------------------------------------------------------

void test_render_with_tool() {
    const llamad::ChatFormat format   = make_format();
    const llamad::RenderedChat rendered =
        format.render({user("What is the weather in Paris?")}, {weather_tool()}, "");

    CHECK(contains(rendered.prompt, "get_weather"));
    CHECK(contains(rendered.prompt, "Get the current weather in a city"));
    CHECK(contains(rendered.prompt, "\"city\""));
    CHECK(contains(rendered.prompt, "What is the weather in Paris?"));

    CHECK(!rendered.grammar.grammar.empty());
    CHECK(rendered.grammar.lazy);
    CHECK(!rendered.grammar.trigger_patterns.empty() || !rendered.grammar.trigger_words.empty());
}

void test_render_without_tools() {
    const llamad::ChatFormat format   = make_format();
    const llamad::RenderedChat rendered = format.render({user("hi")}, {}, "");

    CHECK(rendered.grammar.grammar.empty());
    CHECK(!rendered.grammar.lazy);
    CHECK(rendered.grammar.trigger_patterns.empty());
    CHECK(rendered.grammar.trigger_words.empty());

    CHECK(contains(rendered.prompt, "<|im_start|>user\nhi<|im_end|>"));
    // The assistant turn is opened and left open.
    const std::string opener = "<|im_start|>assistant\n";
    CHECK(rendered.prompt.size() >= opener.size() &&
          rendered.prompt.compare(rendered.prompt.size() - opener.size(), opener.size(), opener) == 0);
}

void test_render_tool_result_history() {
    llamad::ToolCall call;
    call.id             = "call_1";
    call.name           = "get_weather";
    call.arguments_json = R"({"city":"Paris"})";

    llamad::ChatMessage assistant;
    assistant.role       = "assistant";
    assistant.tool_calls = {call};

    llamad::ChatMessage result;
    result.role         = "tool";
    result.content      = "18C and sunny";
    result.tool_call_id = "call_1";

    const llamad::ChatFormat format = make_format();
    const llamad::RenderedChat rendered =
        format.render({user("What is the weather in Paris?"), assistant, result}, {weather_tool()}, "");

    CHECK(contains(rendered.prompt, "<tool_call>"));
    CHECK(contains(rendered.prompt, "<tool_response>"));
    CHECK(contains(rendered.prompt, "18C and sunny"));
}

void test_stream_single_tool_call() {
    const llamad::ChatFormat format   = make_format();
    const llamad::RenderedChat rendered =
        format.render({user("What is the weather in Paris?")}, {weather_tool()}, "");

    llamad::ChatFormat::Stream stream = format.stream(rendered);

    const std::string generated = "Let me check.\n<tool_call>\n"
                                  R"({"name": "get_weather", "arguments": {"city": "Paris"}})"
                                  "\n</tool_call>";

    const std::string streamed = push_bytewise(stream, generated);

    // Content before the call must be delivered by push(), not held back until finish().
    CHECK(contains(streamed, "Let me check."));

    const llamad::ChatFormat::Stream::Final final = stream.finish();
    const std::string                       all   = streamed + final.content_tail;

    CHECK(contains(all, "Let me check."));
    CHECK(!contains(all, "<tool_call"));
    CHECK(!contains(all, "get_weather"));
    // Not even a one-character fragment of the markup may leak.
    CHECK(!contains(all, "<"));

    CHECK_EQ(final.tool_calls.size(), size_t(1));
    if (final.tool_calls.size() == 1) {
        CHECK_EQ(final.tool_calls[0].name, std::string("get_weather"));
        CHECK_EQ(normalize_json(final.tool_calls[0].arguments_json), std::string(R"({"city":"Paris"})"));
        CHECK_EQ(final.tool_calls[0].id.size(), size_t(32));
    }
}

void test_stream_two_tool_calls() {
    const llamad::ChatFormat format   = make_format();
    const llamad::RenderedChat rendered =
        format.render({user("Weather in Paris and Berlin?")}, {weather_tool()}, "");

    llamad::ChatFormat::Stream stream = format.stream(rendered);

    const std::string generated = "<tool_call>\n"
                                  R"({"name": "get_weather", "arguments": {"city": "Paris"}})"
                                  "\n</tool_call>\n<tool_call>\n"
                                  R"({"name": "get_weather", "arguments": {"city": "Berlin"}})"
                                  "\n</tool_call>";

    const std::string streamed = push_bytewise(stream, generated);
    const llamad::ChatFormat::Stream::Final final = stream.finish();

    CHECK(!contains(streamed + final.content_tail, "<"));

    CHECK_EQ(final.tool_calls.size(), size_t(2));
    if (final.tool_calls.size() == 2) {
        CHECK_EQ(final.tool_calls[0].name, std::string("get_weather"));
        CHECK_EQ(final.tool_calls[1].name, std::string("get_weather"));
        CHECK_EQ(normalize_json(final.tool_calls[0].arguments_json), std::string(R"({"city":"Paris"})"));
        CHECK_EQ(normalize_json(final.tool_calls[1].arguments_json), std::string(R"({"city":"Berlin"})"));
        CHECK(!final.tool_calls[0].id.empty());
        CHECK(!final.tool_calls[1].id.empty());
        CHECK(final.tool_calls[0].id != final.tool_calls[1].id);
    }
}

void test_stream_plain_text() {
    const llamad::ChatFormat format = make_format();

    // With tools offered but not used, and with none offered at all: both must pass text through.
    for (const std::vector<llamad::Tool> & tools : {std::vector<llamad::Tool>{weather_tool()},
                                                    std::vector<llamad::Tool>{}}) {
        const llamad::RenderedChat rendered = format.render({user("Say something nice.")}, tools, "");
        llamad::ChatFormat::Stream stream   = format.stream(rendered);

        // A euro sign and a two-code-point emoji, fed one byte at a time. The engine's StreamFilter
        // never hands ChatFormat a split code point, but the parser copes with one anyway, so the
        // stricter feed is what gets tested.
        const std::string input    = "Hello \xE2\x82\xAC" "10 \xF0\x9F\x8C\xA7\xEF\xB8\x8F today.";
        const std::string streamed = push_bytewise(stream, input);

        const llamad::ChatFormat::Stream::Final final = stream.finish();
        CHECK_EQ(final.tool_calls.size(), size_t(0));
        CHECK_EQ(streamed + final.content_tail, input);
    }
}

void test_stream_malformed_tool_call() {
    const llamad::ChatFormat format   = make_format();
    const llamad::RenderedChat rendered = format.render({user("Weather in Paris?")}, {weather_tool()}, "");

    // Truncated argument fields and JSON objects cannot form executable calls, even when
    // common's lenient mapper closes an unfinished object for display.
    for (const std::string & generated : {
             "Checking.\n<tool_call>\n" R"({"name": "get_weather", "argum)",
             "Checking.\n<tool_call>\n" R"({"name":"get_weather","arguments":{"city":"Paris")"}) {
        for (size_t chunk_size : {size_t(0), size_t(1)}) {
            const Run run = run_stream(format, rendered, generated, chunk_size);
            CHECK_EQ(run.calls.size(), size_t(0));
            CHECK(contains(run.streamed, "Checking."));
            CHECK(!contains(run.content(), "<tool_call"));
            CHECK(!contains(run.content(), "get_weather"));
            CHECK(!contains(run.content(), "{"));
            CHECK(!contains(run.content(), "<"));
        }
    }
}

void test_stream_argument_objects() {
    const llamad::ChatFormat format = make_format();
    const llamad::RenderedChat rendered = format.render({user("Weather?")}, {weather_tool()}, "");

    struct Case {
        const char * arguments;
        bool         accepted;
    };
    const Case cases[] = {
        {"{}", true}, {R"({"city":"Paris"})", true},
        {"[]", false}, {"[1]", false}, {R"("x")", false}, {"12", false},
        {"true", false}, {"false", false}, {"null", false},
    };
    for (const Case & test_case : cases) {
        const std::string call = "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":" +
                                 std::string(test_case.arguments) + "}\n</tool_call>";
        for (const std::string & calls : {call, "<tool_call>\n"
                                              R"({"name":"get_weather","arguments":{"city":"Berlin"}})"
                                              "\n</tool_call>\n" + call}) {
            const std::string generated = "Checking.\n" + calls;
            const Run reference = run_stream(format, rendered, generated, /*chunk_size*/ 0);
            const size_t expected_calls = test_case.accepted ? (calls == call ? 1 : 2) : 0;
            CHECK_EQ(reference.calls.size(), expected_calls);
            CHECK(contains(reference.content(), "Checking."));
            CHECK(!contains(reference.content(), "<"));
            CHECK(!contains(reference.content(), "get_weather"));
            const Run bytewise = run_stream(format, rendered, generated, /*chunk_size*/ 1);
            CHECK_EQ(bytewise.content(), reference.content());
            CHECK_EQ(bytewise.calls.size(), reference.calls.size());
            for (size_t i = 0; i < bytewise.calls.size() && i < reference.calls.size(); ++i) {
                CHECK_EQ(normalize_json(bytewise.calls[i].arguments_json),
                         normalize_json(reference.calls[i].arguments_json));
            }
        }
    }
}

void test_stream_daemon_ids() {
    const llamad::ChatFormat format(read_file(MISTRAL_TEMPLATE_PATH), /*bos*/ "<s>", /*eos*/ "</s>");
    llamad::ChatMessage assistant;
    assistant.role = "assistant";
    assistant.tool_calls = {{"duplicate", "get_weather", R"({"city":"Paris"})"}};
    llamad::ChatMessage result;
    result.role = "tool";
    result.tool_call_id = "duplicate";
    result.content = "Sunny";
    llamad::ChatMessage reference = result;
    reference.tool_call_id = "resultref";
    const std::vector<llamad::ChatMessage> history = {user("Weather?"), assistant, result, reference};
    const llamad::RenderedChat rendered = format.render(history, {weather_tool()}, "");
    const std::string generated = "[TOOL_CALLS]["
                                 R"({"name":"get_weather","arguments":{"city":"Berlin"},"id":"duplicate"},)"
                                 R"({"name":"get_weather","arguments":{"city":"Rome"},"id":"duplicate"})"
                                 "]";
    for (size_t chunk_size : {size_t(0), size_t(1)}) {
        const Run run = run_stream(format, rendered, generated, chunk_size);
        CHECK_EQ(run.content(), std::string());
        CHECK_EQ(run.calls.size(), size_t(2));
        if (run.calls.size() != 2) {
            continue;
        }
        CHECK(run.calls[0].id != run.calls[1].id);
        auto replay = history;
        assistant.tool_calls = run.calls;
        replay.push_back(assistant);
        for (const llamad::ToolCall & call : run.calls) {
            CHECK_EQ(call.id.size(), size_t(9));
            CHECK(call.id != "duplicate");
            CHECK(call.id != "resultref");
            for (char c : call.id) {
                CHECK((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'));
            }
            result.tool_call_id = call.id;
            replay.push_back(result);
        }
        const llamad::RenderedChat replayed = format.render(replay, {weather_tool()}, "");
        for (const llamad::ToolCall & call : run.calls) {
            CHECK(contains(replayed.prompt, "\"id\": \"" + call.id + "\""));
            CHECK(contains(replayed.prompt, "\"call_id\": \"" + call.id + "\""));
        }
    }
}

// The invariant every chunking must hold: what the caller sees is the content of one non-partial
// parse of the whole output, no matter how the bytes were split. The reference is a Stream fed the
// whole string at once, so the test needs no llama.cpp types of its own.
void test_chunking_invariant() {
    const llamad::ChatFormat format = make_format();

    struct Case {
        const char * text;
        size_t       expected_calls;
        bool         has_markup;   // the output contains a real tool call, so none of it may leak
    };

    const Case cases[] = {
        {"Let me check.\n<tool_call>\n" R"({"name": "get_weather", "arguments": {"city": "Paris"}})"
         "\n</tool_call>", 1, true},
        {"<tool_call>\n" R"({"name": "get_weather", "arguments": {"city": "Paris"}})"
         "\n</tool_call>\n<tool_call>\n" R"({"name": "get_weather", "arguments": {"city": "Berlin"}})"
         "\n</tool_call>", 2, true},
        // Cut off mid-call: the call is dropped, so nothing of it may surface either.
        {"Checking.\n<tool_call>\n" R"({"name": "get_weather", "argum)", 0, true},
        // Markup that is not a tool call is ordinary prose and must come through untouched.
        {"<tool_call>junk that is not json\nand then more prose", 0, false},
        {"if a < b and <tool is a word", 0, false},
        {"a lone < and a bare <tool_call at the very end", 0, false},
        {"Hello \xE2\x82\xAC" "10 \xF0\x9F\x8C\xA7\xEF\xB8\x8F today.", 0, false},
    };

    const size_t chunk_sizes[] = {0, 1, 2, 3, 7};

    for (const std::vector<llamad::Tool> & tools : {std::vector<llamad::Tool>{weather_tool()},
                                                    std::vector<llamad::Tool>{}}) {
        const llamad::RenderedChat rendered = format.render({user("go")}, tools, "");

        for (const Case & test_case : cases) {
            const std::string text = test_case.text;

            // Without tools nothing is ever a tool call, so the whole text is content.
            const bool tools_offered = !tools.empty();
            const Run  reference     = run_stream(format, rendered, text, /*chunk_size*/ 0);

            if (!tools_offered) {
                CHECK_EQ(reference.content(), text);
                CHECK_EQ(reference.calls.size(), size_t(0));
            } else {
                CHECK_EQ(reference.calls.size(), test_case.expected_calls);
                if (test_case.has_markup) {
                    CHECK(!contains(reference.content(), "<tool_call"));
                } else {
                    CHECK_EQ(reference.content(), text);
                }
            }

            for (size_t chunk_size : chunk_sizes) {
                const Run run = run_stream(format, rendered, text, chunk_size);
                CHECK_EQ(run.content(), reference.content());
                CHECK_EQ(run.calls.size(), reference.calls.size());
                for (size_t i = 0; i < run.calls.size() && i < reference.calls.size(); ++i) {
                    CHECK_EQ(run.calls[i].name, reference.calls[i].name);
                    CHECK_EQ(normalize_json(run.calls[i].arguments_json),
                             normalize_json(reference.calls[i].arguments_json));
                }
            }
        }
    }
}

void test_render_with_response_schema() {
    const llamad::ChatFormat   format   = make_format();
    const llamad::RenderedChat rendered = format.render({user("Is this spam?")}, {}, spam_schema());

    // The schema becomes a plain grammar the sampler is held to from the first token, so there is
    // nothing to trigger on.
    CHECK(!rendered.grammar.grammar.empty());
    CHECK(!rendered.grammar.lazy);
    CHECK(rendered.grammar.trigger_patterns.empty());
    CHECK(rendered.grammar.trigger_words.empty());

    CHECK(contains(rendered.prompt, "Is this spam?"));
    // The schema is in the prompt as well as in the grammar: the grammar fixes the shape, the
    // prompt is the only place the schema's descriptions can reach the model.
    CHECK(contains(rendered.prompt, "Reply with a single JSON object that matches this JSON Schema:"));
    CHECK(contains(rendered.prompt, spam_schema()));

    // The grammar's root opens with the assistant turn the prompt already ends with, so the engine
    // is given that text to advance the grammar past before the first token is sampled.
    CHECK_EQ(rendered.grammar.prefill, std::string("<|im_start|>assistant\n"));
    CHECK(contains(rendered.grammar.grammar, "root ::= \"<|im_start|>assistant"));
}

// A model may wrap the object in a ```json fence; the fence is markup, so only the JSON is content.
void test_stream_response_schema_fenced() {
    const llamad::ChatFormat   format   = make_format();
    const llamad::RenderedChat rendered = format.render({user("Is this spam?")}, {}, spam_schema());

    const std::string json      = R"({"spam":true,"reasons":["link farm"]})";
    const std::string generated = "```json\n" + json + "\n```";

    for (size_t chunk_size : {size_t(0), size_t(1)}) {
        const Run run = run_stream(format, rendered, generated, chunk_size);
        CHECK_EQ(normalize_json(run.content()), json);
        CHECK(!contains(run.content(), "`"));
        CHECK_EQ(run.calls.size(), size_t(0));
    }
}

void test_stream_response_schema_raw() {
    const llamad::ChatFormat   format   = make_format();
    const llamad::RenderedChat rendered = format.render({user("Is this spam?")}, {}, spam_schema());

    const std::string json = R"({"spam":false,"reasons":["known sender"]})";

    for (size_t chunk_size : {size_t(0), size_t(1)}) {
        const Run run = run_stream(format, rendered, json, chunk_size);
        CHECK_EQ(normalize_json(run.content()), json);
        CHECK_EQ(run.calls.size(), size_t(0));
    }
}

void test_render_with_response_schema_thinking() {
    const llamad::ChatFormat   format   = make_thinking_format();
    const llamad::RenderedChat rendered = format.render({user("Is this spam?")}, {}, spam_schema());

    CHECK(!rendered.grammar.grammar.empty());
    CHECK(!rendered.grammar.lazy);
    CHECK(rendered.grammar.trigger_patterns.empty());
    CHECK(rendered.grammar.trigger_words.empty());

    // A schema turn turns thinking off, so the template writes a closed, empty think block and the
    // model starts on the JSON. The grammar's root describes that same opening, which is what lets
    // the engine advance the grammar past the prefill.
    CHECK_EQ(rendered.grammar.prefill, std::string("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    CHECK(contains(rendered.grammar.grammar, "root ::= \"<|im_start|>assistant\\n\" (\"<think>\""));
}

// The grammar and the parser are generated from one PEG description, and the parser is run over
// generation_prompt + output, so a stream that parses says the grammar's root admits the prefill
// ahead of the JSON. finish()'s raw fallback emits unparsed output verbatim and would mask a
// rejected parse, hence the assertion that the whole object arrives through push().
void test_stream_response_schema_thinking() {
    const llamad::ChatFormat   format   = make_thinking_format();
    const llamad::RenderedChat rendered = format.render({user("Is this spam?")}, {}, spam_schema());

    const std::string json = R"({"spam":true,"reasons":["link farm"]})";

    for (size_t chunk_size : {size_t(0), size_t(1)}) {
        const Run run = run_stream(format, rendered, json, chunk_size);
        CHECK_EQ(normalize_json(run.streamed), json);
        CHECK(run.tail.empty());
        CHECK(!contains(run.content(), "think"));
        CHECK_EQ(run.calls.size(), size_t(0));
    }
}

// The grammar is built from the schema's shape alone, so two schemas that differ only in a
// description compile to the same grammar. Only the prompt can carry that difference.
void test_render_schema_descriptions_reach_the_prompt() {
    const llamad::ChatFormat format = make_format();

    const std::string sentiment =
        R"({"type":"object","properties":{"answer":{"type":"string",)"
        R"("description":"the message's sentiment, one of positive, neutral or negative"}},)"
        R"("required":["answer"]})";
    const std::string language =
        R"({"type":"object","properties":{"answer":{"type":"string",)"
        R"("description":"the ISO 639-1 code of the language the message is written in"}},)"
        R"("required":["answer"]})";

    const llamad::RenderedChat a = format.render({user("Bonjour!")}, {}, sentiment);
    const llamad::RenderedChat b = format.render({user("Bonjour!")}, {}, language);

    CHECK(contains(a.prompt, "one of positive, neutral or negative"));
    CHECK(contains(b.prompt, "the ISO 639-1 code of the language"));
    CHECK(a.prompt != b.prompt);
}

// The instruction joins an existing system turn instead of adding a second one, so the model sees
// one system section however the history was built.
void test_render_schema_with_system_message() {
    const llamad::ChatFormat format = make_format();

    llamad::ChatMessage system;
    system.role    = "system";
    system.content = "You are a terse spam filter.";

    const llamad::RenderedChat rendered = format.render({system, user("Is this spam?")}, {}, spam_schema());

    CHECK(contains(rendered.prompt, "You are a terse spam filter."));
    CHECK(contains(rendered.prompt, "Reply with a single JSON object that matches this JSON Schema:"));

    size_t sections = 0;
    for (size_t pos = rendered.prompt.find("<|im_start|>system"); pos != std::string::npos;
         pos        = rendered.prompt.find("<|im_start|>system", pos + 1)) {
        ++sections;
    }
    CHECK_EQ(sections, size_t(1));
}

void test_render_schema_with_tools_rejected() {
    const llamad::ChatFormat format = make_format();

    bool threw = false;
    try {
        format.render({user("Is this spam?")}, {weather_tool()}, spam_schema());
    } catch (const llamad::ChatFormatError &) {
        threw = true;
    }
    CHECK(threw);
}

void test_render_schema_not_an_object_rejected() {
    const llamad::ChatFormat format = make_format();

    for (const char * schema : {"[1,2]", "not json"}) {
        bool threw = false;
        try {
            format.render({user("hi")}, {}, schema);
        } catch (const llamad::ChatFormatError &) {
            threw = true;
        }
        CHECK(threw);
    }
}

// `{}` is a JSON Schema that admits any JSON value, but common builds no grammar from it, so the
// reply would be unconstrained prose while the request promised JSON. It is refused rather than
// silently treated as no schema at all.
void test_render_empty_schema_object_rejected() {
    const llamad::ChatFormat format = make_format();

    bool threw = false;
    try {
        format.render({user("hi")}, {}, "{}");
    } catch (const llamad::ChatFormatError &) {
        threw = true;
    }
    CHECK(threw);
}

// A tool without arguments is offered like any other and its call parses to an empty object.
void test_tool_without_arguments() {
    const llamad::ChatFormat   format   = make_format();
    const llamad::RenderedChat rendered = format.render({user("What time is it?")}, {clock_tool()}, "");

    CHECK(contains(rendered.prompt, "get_time"));
    CHECK(contains(rendered.prompt, "Say what time it is"));
    CHECK(!rendered.grammar.grammar.empty());
    CHECK(rendered.grammar.lazy);
    // The arguments rule admits exactly the empty object; the model has nothing else to write.
    CHECK(contains(rendered.grammar.grammar, R"(tool-get-time-schema ::= "{" space "}")"));

    const std::string generated = "<tool_call>\n" R"({"name": "get_time", "arguments": {}})" "\n</tool_call>";
    for (size_t chunk_size : {size_t(0), size_t(1)}) {
        const Run run = run_stream(format, rendered, generated, chunk_size);
        CHECK_EQ(run.content(), std::string());
        CHECK_EQ(run.calls.size(), size_t(1));
        if (run.calls.size() == 1) {
            CHECK_EQ(run.calls[0].name, std::string("get_time"));
            CHECK_EQ(run.calls[0].arguments_json, std::string("{}"));
        }
    }

    // Tagged calls carry no argument text at all, which the wire still spells as an object.
    const llamad::ChatFormat tagged_format = make_thinking_format();
    const llamad::RenderedChat tagged = tagged_format.render({user("What time is it?")}, {clock_tool()}, "");
    for (size_t chunk_size : {size_t(0), size_t(1)}) {
        const Run run = run_stream(tagged_format, tagged,
                                   "</think>\n\n<tool_call>\n<function=get_time>\n</function>\n</tool_call>", chunk_size);
        CHECK_EQ(run.calls.size(), size_t(1));
        if (run.calls.size() == 1) {
            CHECK_EQ(run.calls[0].name, std::string("get_time"));
            CHECK_EQ(run.calls[0].arguments_json, std::string("{}"));
        }
        CHECK(!contains(run.content(), "<tool_call"));
    }
}

void test_invalid_tool_schema() {
    const llamad::ChatFormat format = make_format();

    llamad::Tool broken           = weather_tool();
    broken.parameters_json_schema = R"({"type":"object",)";   // truncated

    bool threw = false;
    try {
        format.render({user("hi")}, {broken}, "");
    } catch (const llamad::ChatFormatError &) {
        threw = true;
    }
    CHECK(threw);
}

}  // namespace

int main() {
    test_render_with_tool();
    test_render_without_tools();
    test_render_tool_result_history();
    test_stream_single_tool_call();
    test_stream_two_tool_calls();
    test_stream_plain_text();
    test_stream_malformed_tool_call();
    test_stream_argument_objects();
    test_stream_daemon_ids();
    test_chunking_invariant();
    test_tool_without_arguments();
    test_invalid_tool_schema();
    test_render_with_response_schema();
    test_stream_response_schema_fenced();
    test_stream_response_schema_raw();
    test_render_with_response_schema_thinking();
    test_stream_response_schema_thinking();
    test_render_schema_descriptions_reach_the_prompt();
    test_render_schema_with_system_message();
    test_render_schema_with_tools_rejected();
    test_render_schema_not_an_object_rejected();
    test_render_empty_schema_object_rejected();

    return tests::report();
}
