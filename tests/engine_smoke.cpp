/** @file
 * @brief In-process engine and chat smoke CLI requiring a local GGUF model.
 *
 * Manual smoke test for llamad::Engine and the chat layer, in-process and without the daemon.
 * The two arguments are the model and the prompt; --help lists the options.
 *
 *   engine_smoke model.gguf --temp 0 --max-tokens 16 "Count to three"
 *   engine_smoke model.gguf --chat --demo-tool "What time is it in Paris?"
 *   engine_smoke model.gguf --grammar-file digits.gbnf "Pick a number"
 *   engine_smoke embedding-model.gguf --embed "a cat" --embed "a kitten" --embed "tax law"
 */

#include "chat_format.h"
#include "engine.h"
#include "engine_flags.h"
#include "flags.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/// Options specific to this executable; the context and offload flags are EngineFlags.
struct Options {
    bool                     chat      = false;  ///< Wrap the prompt as a single user message via the chat template.
    bool                     demo_tool = false;  ///< With --chat: offer the model one get_current_time tool.
    std::string              grammar_file;       ///< Constrain generation with this GBNF file.
    std::optional<float>     temp;               ///< Sampling temperature.
    std::optional<uint32_t>  seed;               ///< Sampling seed.
    std::optional<int32_t>   top_k;              ///< Top-k.
    std::optional<float>     top_p;              ///< Top-p.
    std::optional<float>     min_p;              ///< Min-p.
    std::optional<int32_t>   max_tokens;         ///< Stop after N generated tokens.
    std::vector<std::string> stop;               ///< Stop strings.
    long                     cancel_after = -1;  ///< Return false from the chunk callback after N chunks.
    long                     repeat       = 1;   ///< Run generate() N times in the same process.
    std::vector<std::string> embed;              ///< Texts to embed in one embed() call instead of generating.
    bool                     help = false;       ///< Print usage and exit.
    std::vector<std::string> positional;         ///< The model, then the prompt.
};

/// Print usage and every flag to stderr.
void print_usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s <model.gguf> [options] <prompt>\n"
                 "       %s <model.gguf> [options] --embed TEXT [--embed TEXT ...]\n"
                 "       %s --list-devices\n"
                 "\n"
                 "  --chat              wrap the prompt as a single user message via the chat template\n"
                 "  --demo-tool         with --chat: offer the model one get_current_time tool\n"
                 "  --grammar-file PATH constrain generation with this GBNF file\n"
                 "                      (not lazy; overrides --chat's grammar)\n"
                 "  --temp F            sampling temperature (<= 0 means greedy)\n"
                 "  --seed N            sampling seed\n"
                 "  --top-k N           top-k\n"
                 "  --top-p F           top-p\n"
                 "  --min-p F           min-p\n"
                 "  --max-tokens N      stop after N generated tokens (< 0 = until the context is full)\n"
                 "  --stop STR          stop string (repeatable)\n"
                 "  --cancel-after N    return false from the chunk callback after N chunks\n"
                 "  --repeat N          run generate() N times in the same process\n"
                 "  --embed TEXT        embed TEXT with an embedding model instead of generating (repeatable);\n"
                 "                      prints each vector's first values and its cosine with the first TEXT\n"
                 "%s"
                 "  --help              show this message\n",
                 argv0, argv0, argv0, llamad::kEngineFlagsHelp);
}

/// Fill `options` and `engine_flags` from the command line. Throws cli::FlagError.
void parse_command_line(int argc, char ** argv, Options & options, llamad::EngineFlags & engine_flags) {
    for (llamad::cli::Args args(argc, argv); args.next();) {
        if (!args.is_flag()) {
            options.positional.push_back(args.current());
        } else if (args.is("--chat")) {
            options.chat = true;
        } else if (args.is("--demo-tool")) {
            options.demo_tool = true;
        } else if (args.is("--grammar-file")) {
            options.grammar_file = args.value();
        } else if (args.is("--temp")) {
            options.temp = args.number<float>();
        } else if (args.is("--seed")) {
            options.seed = args.number<uint32_t>();
        } else if (args.is("--top-k")) {
            options.top_k = args.number<int32_t>();
        } else if (args.is("--top-p")) {
            options.top_p = args.number<float>();
        } else if (args.is("--min-p")) {
            options.min_p = args.number<float>();
        } else if (args.is("--max-tokens")) {
            options.max_tokens = args.number<int32_t>();
        } else if (args.is("--stop")) {
            options.stop.push_back(args.value());
        } else if (args.is("--cancel-after")) {
            options.cancel_after = args.number<long>();
        } else if (args.is("--repeat")) {
            options.repeat = args.number<long>();
        } else if (args.is("--embed")) {
            options.embed.push_back(args.value());
        } else if (args.is("--help")) {
            options.help = true;
        } else if (!llamad::parse_engine_flag(args, engine_flags)) {
            throw args.unknown();
        }
    }
}

/// Reads a whole file. Throws if it cannot be opened.
std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open '" + path + "'");
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

/// The one tool --demo-tool offers, so tool calling can be exercised without the daemon.
llamad::Tool demo_tool() {
    llamad::Tool tool;
    tool.name                   = "get_current_time";
    tool.description            = "Get the current time in a given timezone";
    tool.parameters_json_schema = R"({"type":"object","properties":{"timezone":{"type":"string",)"
                                  R"("description":"IANA timezone, e.g. Europe/Paris"}},)"
                                  R"("required":["timezone"]})";
    return tool;
}

/// A sampling flag that was not given leaves the engine's own default in place.
template <typename T>
void set_if_given(const std::optional<T> & flag, T & field) {
    if (flag) {
        field = *flag;
    }
}

/// Spell engine finish reasons in smoke-test output.
const char * reason_name(llamad::FinishReason reason) {
    switch (reason) {
        case llamad::FinishReason::Eog:       return "Eog";
        case llamad::FinishReason::Length:    return "Length";
        case llamad::FinishReason::Stop:      return "Stop";
        case llamad::FinishReason::Cancelled: return "Cancelled";
    }
    return "?";
}

/// One line per vector: its first values, its norm (1 for a unit vector) and its cosine with the
/// first input's, which for unit vectors is just their dot product.
void print_embeddings(const llamad::EmbedResult & result) {
    const std::vector<float> & first = result.embeddings.front().values;
    for (size_t i = 0; i < result.embeddings.size(); ++i) {
        const std::vector<float> & values = result.embeddings[i].values;
        double dot = 0.0;
        double norm = 0.0;
        for (size_t k = 0; k < values.size(); ++k) {
            dot  += static_cast<double>(values[k]) * first[k];
            norm += static_cast<double>(values[k]) * values[k];
        }
        std::printf("embedding %zu: norm %.6f  cosine with 0: %.4f  values:", i, std::sqrt(norm), dot);
        for (size_t k = 0; k < std::min<size_t>(values.size(), 4); ++k) {
            std::printf(" %.6f", values[k]);
        }
        std::printf(" ...\n");
    }
}

}  // namespace

/// Run the executable.
/// @return Zero on success, two for invalid command-line usage, or one for a runtime failure.
int main(int argc, char ** argv) {
    Options                options;
    llamad::EngineFlags    engine_flags;
    llamad::EngineConfig   config;
    llamad::SamplingParams params;

    try {
        parse_command_line(argc, argv, options, engine_flags);
        if (options.help) {
            print_usage(argv[0]);
            return 0;
        }
        if (options.demo_tool && !options.chat) {
            throw llamad::cli::FlagError("--demo-tool only makes sense with --chat");
        }
        config = llamad::to_config(engine_flags);
    } catch (const llamad::cli::FlagError & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        print_usage(argv[0]);
        return 2;
    }

    // Listing devices needs no model.
    if (engine_flags.list_devices) {
        llamad::print_device_table(stdout);
        return 0;
    }

    const size_t n_positional = options.embed.empty() ? 2 : 1;
    const std::vector<std::string> & positional = options.positional;
    if (positional.size() != n_positional) {
        print_usage(argv[0]);
        return 2;
    }

    config.model_path        = positional[0];
    const std::string prompt = options.embed.empty() ? positional[1] : std::string();

    set_if_given(options.temp, params.temperature);
    set_if_given(options.top_k, params.top_k);
    set_if_given(options.top_p, params.top_p);
    set_if_given(options.min_p, params.min_p);
    set_if_given(options.max_tokens, params.max_tokens);
    params.seed = options.seed;
    params.stop = options.stop;

    try {
        llamad::Engine engine(config);

        const llamad::ModelInfo info = engine.info();
        std::fprintf(stderr,
                     "model: %s\nparams: %llu  size: %.2f MiB  n_ctx: %u  n_ctx_train: %u  chat_template: %s\n",
                     info.description.c_str(), (unsigned long long) info.n_params,
                     (double) info.size_bytes / (1024.0 * 1024.0), info.n_ctx, info.n_ctx_train,
                     info.has_chat_template ? "yes" : "no");

        if (!options.embed.empty()) {
            const llamad::EmbedResult result = engine.embed(options.embed, /*keep_going*/ {});
            std::fprintf(stderr, "n_embd: %u  inputs: %zu  input_tokens: %d\n", info.n_embd,
                         result.embeddings.size(), result.input_tokens);
            print_embeddings(result);
            return 0;
        }

        // In chat mode the whole request comes out of the chat layer: the prompt, the tool-call
        // grammar and the tokens whose text the parser needs to see.
        std::unique_ptr<llamad::ChatFormat> format;
        llamad::RenderedChat                rendered;

        std::string text = prompt;
        if (options.chat) {
            const llamad::ChatTemplateInfo tmpl = engine.chat_template();
            if (tmpl.source.empty()) {
                throw std::runtime_error("the model has no built-in chat template");
            }
            format = std::make_unique<llamad::ChatFormat>(tmpl.source, tmpl.bos_token, tmpl.eos_token);

            std::vector<llamad::Tool> tools;
            if (options.demo_tool) {
                tools.push_back(demo_tool());
            }

            rendered = format->render({{"user", prompt, {}, {}}}, tools, /*response_json_schema*/ "");

            text                    = rendered.prompt;
            params.grammar          = rendered.grammar;
            params.preserved_tokens = rendered.preserved_tokens;
            params.stop.insert(params.stop.end(), rendered.additional_stops.begin(),
                               rendered.additional_stops.end());
        }

        // A hand-written grammar replaces whatever the chat layer came up with, so that a
        // constraint can be tried out on its own.
        if (!options.grammar_file.empty()) {
            // A fresh GrammarSpec is not lazy and has no prefill: a hand-written grammar describes
            // the output alone, so it must not be advanced past the prompt's generation prefix the
            // way the chat layer's grammars are.
            params.grammar         = llamad::GrammarSpec{};
            params.grammar.grammar = read_file(options.grammar_file);
        }

        std::fprintf(stderr, "prompt tokens: %zu\n", engine.tokenize(text, true, true).size());

        for (long run = 0; run < options.repeat; ++run) {
            if (options.repeat > 1) {
                std::fprintf(stderr, "--- run %ld/%ld ---\n", run + 1, options.repeat);
            }

            // The parser is what withholds tool-call markup, so only what it returns is printed.
            std::optional<llamad::ChatFormat::Stream> stream;
            if (format) {
                stream.emplace(format->stream(rendered));
            }

            long chunks = 0;
            auto on_chunk = [&](const std::string & piece) -> bool {
                ++chunks;
                const std::string visible = stream ? stream->push(piece) : piece;
                if (!visible.empty()) {
                    std::fwrite(visible.data(), 1, visible.size(), stdout);
                    std::fflush(stdout);
                }
                return !(options.cancel_after >= 0 && chunks >= options.cancel_after);
            };

            const llamad::GenerateResult result = engine.generate(text, params, on_chunk);

            std::fflush(stdout);
            std::fprintf(stderr,
                         "\nfinish: %s  prompt_tokens: %d  cached_prompt_tokens: %d  completion_tokens: %d  "
                         "prompt_ms: %.1f  completion_ms: %.1f  chunks: %ld\n",
                         reason_name(result.reason), result.stats.prompt_tokens,
                         result.stats.cached_prompt_tokens, result.stats.completion_tokens, result.stats.prompt_ms,
                         result.stats.completion_ms, chunks);

            if (stream) {
                const llamad::ChatFormat::Stream::Final final = stream->finish();
                if (!final.content_tail.empty()) {
                    std::fprintf(stderr, "content_tail: %s\n", final.content_tail.c_str());
                }
                for (const llamad::ToolCall & call : final.tool_calls) {
                    std::fprintf(stderr, "tool_call: %s %s  (id %s)\n", call.name.c_str(),
                                 call.arguments_json.c_str(), call.id.c_str());
                }
                std::fprintf(stderr, "tool_calls: %zu\n", final.tool_calls.size());
            }
        }
    } catch (const llamad::ChatFormatError & e) {
        std::fprintf(stderr, "chat format error: %s\n", e.what());
        return 1;
    } catch (const llamad::EngineError & e) {
        std::fprintf(stderr, "engine error: %s\n", e.what());
        return 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    return 0;
}
