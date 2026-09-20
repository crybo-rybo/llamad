// Manual smoke test for llamad::Engine. No third-party dependencies.
//
//   engine_smoke <model.gguf> [options] <prompt>
//
// Options:
//   --chat              wrap the prompt as a single user message via the chat template
//   --temp T            sampling temperature (<= 0 means greedy)
//   --seed N            sampling seed
//   --top-k N           top-k
//   --top-p P           top-p
//   --min-p P           min-p
//   --max-tokens N      stop after N generated tokens (< 0 = until the context is full)
//   --stop STR          stop string (repeatable)
//   --cancel-after N    return false from the chunk callback after N chunks
//   --repeat N          run generate() N times in the same process
//   --ctx N             context size
//   --threads N         thread count (0 = let llama.cpp decide)
//   --ngl N             number of layers to offload to the GPU

#include "engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

void print_usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s <model.gguf> [--chat] [--temp T] [--seed N] [--top-k N] [--top-p P]\n"
                 "          [--min-p P] [--max-tokens N] [--stop STR]... [--cancel-after N]\n"
                 "          [--repeat N] [--ctx N] [--threads N] [--ngl N] <prompt>\n",
                 argv0);
}

const char * reason_name(llamad::FinishReason reason) {
    switch (reason) {
        case llamad::FinishReason::Eog:       return "Eog";
        case llamad::FinishReason::Length:    return "Length";
        case llamad::FinishReason::Stop:      return "Stop";
        case llamad::FinishReason::Cancelled: return "Cancelled";
    }
    return "?";
}

}  // namespace

int main(int argc, char ** argv) {
    llamad::EngineConfig    config;
    llamad::SamplingParams  params;
    std::vector<std::string> positional;

    bool    chat         = false;
    long    cancel_after = -1;
    long    repeat       = 1;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];

            auto next = [&](const char * name) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error(std::string("missing value for ") + name);
                }
                return argv[++i];
            };

            if (arg == "--chat") {
                chat = true;
            } else if (arg == "--temp") {
                params.temperature = std::stof(next("--temp"));
            } else if (arg == "--seed") {
                params.seed = static_cast<uint32_t>(std::stoul(next("--seed")));
            } else if (arg == "--top-k") {
                params.top_k = std::stoi(next("--top-k"));
            } else if (arg == "--top-p") {
                params.top_p = std::stof(next("--top-p"));
            } else if (arg == "--min-p") {
                params.min_p = std::stof(next("--min-p"));
            } else if (arg == "--max-tokens") {
                params.max_tokens = std::stoi(next("--max-tokens"));
            } else if (arg == "--stop") {
                params.stop.push_back(next("--stop"));
            } else if (arg == "--cancel-after") {
                cancel_after = std::stol(next("--cancel-after"));
            } else if (arg == "--repeat") {
                repeat = std::stol(next("--repeat"));
            } else if (arg == "--ctx") {
                config.n_ctx = static_cast<uint32_t>(std::stoul(next("--ctx")));
            } else if (arg == "--threads") {
                config.n_threads = std::stoi(next("--threads"));
            } else if (arg == "--ngl") {
                config.n_gpu_layers = std::stoi(next("--ngl"));
            } else if (arg == "-h" || arg == "--help") {
                print_usage(argv[0]);
                return 0;
            } else if (arg.rfind("--", 0) == 0) {
                throw std::runtime_error("unknown option: " + arg);
            } else {
                positional.push_back(arg);
            }
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        print_usage(argv[0]);
        return 2;
    }

    if (positional.size() != 2) {
        print_usage(argv[0]);
        return 2;
    }

    config.model_path        = positional[0];
    const std::string prompt = positional[1];

    try {
        llamad::Engine engine(config);

        const llamad::ModelInfo info = engine.info();
        std::fprintf(stderr,
                     "model: %s\nparams: %llu  size: %.2f MiB  n_ctx: %u  n_ctx_train: %u  chat_template: %s\n",
                     info.description.c_str(), (unsigned long long) info.n_params,
                     (double) info.size_bytes / (1024.0 * 1024.0), info.n_ctx, info.n_ctx_train,
                     info.has_chat_template ? "yes" : "no");

        std::string text = prompt;
        if (chat) {
            text = engine.apply_chat_template({{"user", prompt}});
        }
        std::fprintf(stderr, "prompt tokens: %zu\n", engine.tokenize(text, true, true).size());

        for (long run = 0; run < repeat; ++run) {
            if (repeat > 1) {
                std::fprintf(stderr, "--- run %ld/%ld ---\n", run + 1, repeat);
            }

            long chunks = 0;
            auto on_chunk = [&](const std::string & piece) -> bool {
                ++chunks;
                std::fwrite(piece.data(), 1, piece.size(), stdout);
                std::fflush(stdout);
                return !(cancel_after >= 0 && chunks >= cancel_after);
            };

            const llamad::GenerateResult result = engine.generate(text, params, on_chunk);

            std::fflush(stdout);
            std::fprintf(stderr,
                         "\nfinish: %s  prompt_tokens: %d  completion_tokens: %d  "
                         "prompt_ms: %.1f  completion_ms: %.1f  chunks: %ld\n",
                         reason_name(result.reason), result.stats.prompt_tokens,
                         result.stats.completion_tokens, result.stats.prompt_ms,
                         result.stats.completion_ms, chunks);
        }
    } catch (const llamad::EngineError & e) {
        std::fprintf(stderr, "engine error: %s\n", e.what());
        return 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    return 0;
}
