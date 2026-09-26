/** @file
 * @brief Prefill and decode throughput of the engine, in-process, requiring a local GGUF model.
 *
 * Runs a fixed set of scenarios through Engine::generate, the path every Generate and Chat
 * request takes, and prints the median tokens per second of each as one tab-separated line:
 *
 *   engine_bench model.gguf [--reps 5] [--ngl 99]
 *
 * Prefill (pp) scenarios decode a prompt of about N tokens and generate nothing. Decode (tg)
 * scenarios generate up to 128 tokens: greedy, with the default sampler chain, after a 2048-token
 * prompt, and under a response-schema grammar from the chat layer. Every run's prompt starts with
 * its own run number, so the engine's prompt cache can only ever reuse the first token or two and
 * a prefill run is always a cold one.
 */

#include "chat_format.h"
#include "engine.h"
#include "engine_flags.h"
#include "flags.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

namespace {

using llamad::cli::help;

/// Options specific to this executable; shared flags are composed separately.
struct Options {
    [[=help{"N", "measured runs per scenario, after one warm-up run (default 5)"}]]
    int32_t reps = 5;  ///< Measured runs per scenario.

    [[=help{"NAME", "run only the scenarios whose name starts with NAME (repeatable)"}]]
    std::vector<std::string> only;  ///< Scenario name prefixes to run; empty runs every scenario.
};

/// Varied English, so the tokenizer sees ordinary text rather than one repeated token.
constexpr const char * kFiller[] = {
    "The harbour town woke slowly, fishing boats knocking against the wooden pier. ",
    "A baker on the corner pulled the first loaves from an oven older than the street. ",
    "Gulls argued over scraps while the tide drew back across the grey shingle. ",
    "In the library a clerk catalogued maps of coastlines that no longer existed. ",
    "Rain arrived at noon, thin and persistent, and the market stalls folded their awnings. ",
    "Two children raced paper boats down the gutter toward the harbour wall. ",
    "The lighthouse keeper wrote each ship's name in a ledger bound in cracked leather. ",
    "By evening the wind turned east and carried the smell of salt and diesel inland. ",
};

/// A prompt of `n_tokens` tokens, or one or two more where no word lands on it exactly, opening
/// with `run` so no two runs share more than their first token or two. Going over matters: a
/// prompt one token longer than a micro-batch costs a whole extra graph evaluation.
std::string filler_prompt(const llamad::Engine & engine, int run, size_t n_tokens, const std::string & tail) {
    const auto length = [&](const std::string & text) {
        return engine.tokenize(text + tail, /*add_special*/ true, /*parse_special*/ true).size();
    };

    // Whole sentences while there is room for one, then single words to land on the count.
    std::string text = "Document " + std::to_string(run) + ".\n";
    for (size_t i = 0; length(text) + 32 < n_tokens; ++i) {
        text += kFiller[i % std::size(kFiller)];
    }
    while (length(text) < n_tokens) {
        text += " sea";
    }
    return text + tail;
}

/// One measured quantity: the prompt tokens decoded or the tokens generated, and how long it took.
struct Sample {
    int32_t tokens = 0;  ///< Prompt tokens decoded, or tokens generated.
    double  ms     = 0;  ///< Wall-clock milliseconds they took.

    /// Throughput in tokens per second; zero when nothing was timed.
    double per_second() const { return ms > 0 ? tokens * 1000.0 / ms : 0; }
};

/// A named scenario: builds run `run`'s prompt and parameters, and says what it measures.
struct Scenario {
    std::string                         name;     ///< Printed name, matched by --only.
    bool                                prefill;  ///< Measures the prompt, not generation.
    std::function<std::string(int run)> prompt;   ///< The prompt for run number `run`.
    llamad::SamplingParams              params;   ///< Sampling, grammar and token budget.
};

/// Runs the scenario once and returns what it measures: the prompt or the generation.
Sample run_once(llamad::Engine & engine, const Scenario & scenario, int run) {
    const llamad::GenerateResult result = engine.generate(scenario.prompt(run), scenario.params, /*on_chunk*/ {});
    if (scenario.prefill) {
        return {result.stats.prompt_tokens - result.stats.cached_prompt_tokens, result.stats.prompt_ms};
    }
    return {result.stats.completion_tokens, result.stats.completion_ms};
}

/// Whether --only asks for this scenario; no --only asks for all of them.
bool selected(const Options & options, const std::string & name) {
    if (options.only.empty()) {
        return true;
    }
    return std::any_of(options.only.begin(), options.only.end(),
                       [&](const std::string & prefix) { return name.starts_with(prefix); });
}

/// Print usage and reflected flag descriptions to stderr.
void print_usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s <model.gguf> [options]\n\n", argv0);
    llamad::cli::print_flags(stderr, Options{}, llamad::EngineFlags{}, llamad::cli::HelpFlag{});
}

}  // namespace

/// Run the executable.
/// @return Zero on success, two for invalid command-line usage, or one for a runtime failure.
int main(int argc, char ** argv) {
    Options                  options;
    llamad::EngineFlags      engine_flags;
    llamad::cli::HelpFlag    help_flag;
    llamad::EngineConfig     config;
    std::vector<std::string> positional;

    try {
        positional = llamad::cli::parse_flags(argc, argv, options, engine_flags, help_flag);
        if (help_flag.help) {
            print_usage(argv[0]);
            return 0;
        }
        if (options.reps < 1) {
            throw llamad::cli::FlagError("--reps needs a positive integer");
        }
        config = llamad::to_config(engine_flags);
    } catch (const llamad::cli::FlagError & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        print_usage(argv[0]);
        return 2;
    }
    if (positional.size() != 1) {
        print_usage(argv[0]);
        return 2;
    }
    config.model_path = positional[0];

    try {
        llamad::Engine engine(config);

        const llamad::ModelInfo info = engine.info();
        std::fprintf(stderr, "model: %s  n_ctx: %u\n", info.description.c_str(), info.n_ctx);

        llamad::SamplingParams greedy;
        greedy.temperature = 0.0f;
        greedy.max_tokens  = 128;

        llamad::SamplingParams sampled;  // the daemon's defaults
        sampled.seed       = 42;
        sampled.max_tokens = 128;

        llamad::SamplingParams prompt_only;
        prompt_only.max_tokens = 0;

        const std::string story = "Write a long story about a lighthouse keeper, in plain prose:\n";

        std::vector<Scenario> scenarios;
        for (const size_t n : {128, 512, 2048}) {
            if (n >= info.n_ctx) {
                continue;
            }
            scenarios.push_back({"pp" + std::to_string(n), true,
                                 [&, n](int run) { return filler_prompt(engine, run, n, ""); }, prompt_only});
        }
        scenarios.push_back({"tg128-greedy", false,
                             [&](int run) { return "Story " + std::to_string(run) + ". " + story; }, greedy});
        scenarios.push_back({"tg128-sampled", false,
                             [&](int run) { return "Story " + std::to_string(run) + ". " + story; }, sampled});
        if (2048 + 128 < info.n_ctx) {
            scenarios.push_back({"tg128-greedy@2048", false,
                                 [&](int run) { return filler_prompt(engine, run, 2048, story); }, greedy});
        }

        // The chat layer's grammar for a response schema, as a Chat request with one would get.
        // It outlives the scenarios, whose prompts it renders.
        std::optional<llamad::ChatFormat> format;
        const llamad::ChatTemplateInfo    tmpl = engine.chat_template();
        if (!tmpl.source.empty()) {
            format.emplace(tmpl.source, tmpl.bos_token, tmpl.eos_token);
            const std::string schema =
                R"({"type":"object","properties":{"towns":{"type":"array","items":{"type":"object",)"
                R"("properties":{"name":{"type":"string"},"population":{"type":"integer"},)"
                R"("description":{"type":"string"}},"required":["name","population","description"]}}},)"
                R"("required":["towns"]})";
            llamad::SamplingParams json = greedy;
            json.grammar                = format->render({{"user", "x", {}, {}}}, {}, schema).grammar;
            scenarios.push_back({"tg128-json", false,
                                 [&, schema](int run) {
                                     const std::string ask = "Request " + std::to_string(run) +
                                                             ": describe ten imaginary coastal towns in detail.";
                                     return format->render({{"user", ask, {}, {}}}, {}, schema).prompt;
                                 },
                                 json});
        }

        std::printf("scenario\ttokens\ttok_per_s\tmin\tmax\n");
        int run = 0;
        for (const Scenario & scenario : scenarios) {
            if (!selected(options, scenario.name)) {
                continue;
            }
            run_once(engine, scenario, run++);  // warm-up

            std::vector<Sample> samples;
            for (int32_t rep = 0; rep < options.reps; ++rep) {
                samples.push_back(run_once(engine, scenario, run++));
            }
            std::sort(samples.begin(), samples.end(),
                      [](const Sample & a, const Sample & b) { return a.per_second() < b.per_second(); });
            const Sample & median = samples[samples.size() / 2];
            std::printf("%s\t%d\t%.1f\t%.1f\t%.1f\n", scenario.name.c_str(), median.tokens, median.per_second(),
                        samples.front().per_second(), samples.back().per_second());
            std::fflush(stdout);
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
