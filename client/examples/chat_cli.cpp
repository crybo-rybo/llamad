/** @file
 * @brief Interactive chat client with annotated time-tool and typed-reply examples.
 *
 * llamad-chat: a multi-turn chat REPL against a running llamad daemon, or with --embed a
 * one-shot embedding against a daemon serving an embedding model.
 * It uses only <llamad/client.h>: no gRPC or protobuf headers anywhere.
 */

#include <cctype>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <unistd.h>

#include "flags.h"
#include "llamad/client.h"

namespace {

// SIGINT is blocked in every thread and taken with sigwait by a thread of its own, where stopping
// a call is an ordinary function call rather than something a signal handler may not do. The
// stop cancels the turn even while no text is arriving: a long prompt, a tool round. At the
// prompt there is nothing to stop, so it exits the process instead.
/// Guards g_turn_stop, which the SIGINT thread reads while turns begin and end.
std::mutex         g_turn_mutex;
/// The running turn's stop source, or null at the prompt.
std::stop_source * g_turn_stop = nullptr;

/// Stop the running turn on each SIGINT; with none running, exit.
void watch_sigint(sigset_t signals) {
    while (true) {
        int received = 0;
        if (::sigwait(&signals, &received) != 0) {
            continue;
        }
        std::lock_guard<std::mutex> lock(g_turn_mutex);
        if (g_turn_stop != nullptr) {
            g_turn_stop->request_stop();
            continue;
        }
        std::fputc('\n', stdout);
        std::fflush(stdout);
        std::_Exit(130);  // not exit(): the main thread is still running
    }
}

/// A turn SIGINT can stop, for as long as the object lives.
struct StoppableTurn {
    std::stop_source stop;  ///< Requested by the SIGINT thread.

    /// Make this turn the one SIGINT stops.
    StoppableTurn() {
        std::lock_guard<std::mutex> lock(g_turn_mutex);
        g_turn_stop = &stop;
    }
    /// Return SIGINT to exiting the process.
    ~StoppableTurn() {
        std::lock_guard<std::mutex> lock(g_turn_mutex);
        g_turn_stop = nullptr;
    }
};

/// Spell the client finish reason for terminal statistics.
const char * reason_name(llamad::client::FinishReason reason) {
    switch (reason) {
        case llamad::client::FinishReason::Eog:       return "eog";
        case llamad::client::FinishReason::Length:    return "length";
        case llamad::client::FinishReason::Stop:      return "stop";
        case llamad::client::FinishReason::Cancelled: return "cancelled";
        case llamad::client::FinishReason::ToolCalls: return "tool_calls";
    }
    return "unspecified";
}

using llamad::cli::help;

/// Options specific to this executable; shared flags are composed separately.
struct Options {
    [[=help{"PATH", "daemon socket (default: the client library default)"}]]
    std::optional<std::string> socket;  ///< Daemon socket (default: the client library default).

    [[=help{"TEXT", "system prompt for the conversation"}]]
    std::string system;  ///< System prompt for the conversation.

    [[=help{"sampling temperature (0 = greedy)"}]]
    std::optional<float> temp;  ///< Sampling temperature (0 = greedy).

    [[=help{"sampling seed"}]]
    std::optional<uint32_t> seed;  ///< Sampling seed.

    [[=help{"cap on generated tokens per turn"}]]
    std::optional<int32_t> max_tokens;  ///< Cap on generated tokens per turn.

    [[=help{"PROMPT", "run a single non-interactive turn and exit"}]]
    std::optional<std::string> once;  ///< Run a single non-interactive turn and exit.

    [[=help{"offer the built-in get_current_time tool and run the\n"
            "execute-and-resend loop for any call the model makes"}]]
    bool demo_tools = false;  ///< Offer the built-in get_current_time tool and run the  execute-and-resend loop for any call the model makes.

    [[=help{"ask the model to fill a fixed struct (a spam verdict) and print\n"
            "its fields; one turn, then exit"}]]
    bool demo_json = false;  ///< Ask the model for a fixed struct and print the fields; one turn, then exit.

    [[=help{"TEXT", "embed TEXT with the daemon's embedding model (repeatable),\n"
                    "print each vector's first values and its cosine with the\n"
                    "first TEXT, and exit"}]]
    std::vector<std::string> embed;  ///< Texts to embed in one request instead of chatting.

    // Hidden, for testing cancellation: cancel the turn after N chunks.
    std::optional<long> cancel_after;  ///< Hidden test flag: cancel after this many delivered chunks.
};

/// Print usage and reflected flag descriptions to stderr.
void print_usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s [options]\n\n", argv0);
    llamad::cli::print_flags(stderr, Options{}, llamad::cli::HelpFlag{});
}

// --- the --demo-tools tool -------------------------------------------------------------------
//
// A tool is a C++ function: its identifier names it, its desc annotations describe it and its
// parameters, and its parameter list is the argument schema the daemon constrains the model to.
// The result struct travels back to the model as JSON.

using llamad::client::desc;

/// tzset() quietly falls back to UTC for a zone it cannot resolve; checking the zone file
/// first is what lets the result say so.
bool zone_is_known(const std::string & zone) {
    if (zone.empty() || zone.front() == '/' || zone.find("..") != std::string::npos) {
        return false;
    }
    const std::string punctuation = "_-+/";
    for (const char c : zone) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && punctuation.find(c) == std::string::npos) {
            return false;
        }
    }
    const char *      dir  = std::getenv("TZDIR");
    const std::string root = dir != nullptr && dir[0] != '\0' ? dir : "/usr/share/zoneinfo";
    return ::access((root + "/" + zone).c_str(), R_OK) == 0;
}

/// Structured result of the demonstration time tool.
struct CurrentTime {
    std::string                timezone;  ///< Resolved IANA timezone, or UTC for an unknown zone.
    std::string                time;      ///< Formatted local timestamp.
    std::optional<std::string> note;      ///< Explanation of a fallback zone, omitted on success.
};

// Local time in an IANA zone, via the C library: TZ is set for the call and restored after.
/// Read local time in an IANA zone, restoring the process TZ setting before returning.
/// Unknown zones produce UTC plus an explanatory note. This example mutates process
/// environment state and is intended for sequential tool execution.
[[=desc{"Get the current date and time in a given IANA timezone."}]]
CurrentTime get_current_time([[=desc{"IANA timezone, e.g. Europe/Paris"}]] std::string timezone) {
    std::optional<std::string> note;
    if (!zone_is_known(timezone)) {
        note     = timezone.empty() ? "no timezone given, answering in UTC"
                                    : "unknown timezone '" + timezone + "', answering in UTC";
        timezone = "UTC";
    }

    const char *      previous = std::getenv("TZ");
    const bool        had_tz   = previous != nullptr;
    const std::string saved    = had_tz ? previous : "";

    ::setenv("TZ", timezone.c_str(), 1);
    ::tzset();

    const std::time_t now = std::time(nullptr);
    std::tm           local{};
    char              stamp[64] = "";
    if (::localtime_r(&now, &local) != nullptr) {
        std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S %Z", &local);
    }

    if (had_tz) {
        ::setenv("TZ", saved.c_str(), 1);
    } else {
        ::unsetenv("TZ");
    }
    ::tzset();

    std::fprintf(stderr, "[tool] get_current_time(%s) -> %s\n", timezone.c_str(), stamp);
    return {timezone, stamp, note};
}

// Rounds of tool calls per turn. A model that keeps asking is looping; stop instead.
/// Maximum model requests per terminal turn, bounding repeated tool calls.
const int kMaxToolRounds = 8;

/// Print the one-line generation summary every turn ends with.
void print_stats(const llamad::client::GenerateResult & result) {
    std::fprintf(stderr,
                 "[stats] finish=%s prompt_tokens=%d cached_prompt_tokens=%d completion_tokens=%d "
                 "prompt_ms=%.1f completion_ms=%.1f\n",
                 reason_name(result.reason),
                 result.stats.prompt_tokens,
                 result.stats.cached_prompt_tokens,
                 result.stats.completion_tokens,
                 result.stats.prompt_ms,
                 result.stats.completion_ms);
}

/// One line per vector: its first values and its cosine with the first input's, which for the
/// unit vectors the daemon returns is just their dot product.
void print_embeddings(const llamad::client::EmbedResult & result) {
    const std::vector<float> & first = result.embeddings.front().values;
    for (size_t i = 0; i < result.embeddings.size(); ++i) {
        const std::vector<float> & values = result.embeddings[i].values;
        double cosine = 0.0;
        for (size_t k = 0; k < values.size() && k < first.size(); ++k) {
            cosine += static_cast<double>(values[k]) * first[k];
        }
        std::printf("embedding %zu: cosine with 0: %.4f  values:", i, cosine);
        for (size_t k = 0; k < values.size() && k < 4; ++k) {
            std::printf(" %.6f", values[k]);
        }
        std::printf(" ...\n");
    }
    std::fflush(stdout);
    std::fprintf(stderr, "[stats] inputs=%zu n_embd=%zu input_tokens=%d\n", result.embeddings.size(),
                 first.size(), result.input_tokens);
}

/// Runs one turn: streams the reply to stdout while the client appends every assistant and
/// "tool" turn it takes to the history. Returns false if the turn was cancelled.
bool run_turn(llamad::client::Client & client,
              std::vector<llamad::client::ChatMessage> & history,
              const llamad::client::ToolSet & tools,
              const llamad::client::SamplingParams & sampling,
              long cancel_after) {
    // The user turn this reply belongs to, so a cancelled empty reply can drop it along with
    // whatever tool traffic the turn had already accumulated.
    const size_t user_turn = history.empty() ? 0 : history.size() - 1;

    bool produced_text = false;
    long chunks        = 0;

    llamad::client::GenerateResult result{};
    {
        StoppableTurn turn;
        result = client.chat(history, tools, sampling, [&](const std::string & text) {
            produced_text = true;
            std::fwrite(text.data(), 1, text.size(), stdout);
            std::fflush(stdout);
            ++chunks;
            return cancel_after <= 0 || chunks < cancel_after;
        }, kMaxToolRounds, {.stop = turn.stop.get_token()});
    }

    std::fputc('\n', stdout);
    std::fflush(stdout);

    const bool cancelled = result.reason == llamad::client::FinishReason::Cancelled;
    if (cancelled) {
        std::fprintf(stderr, "[cancelled]\n");
    }
    print_stats(result);

    // Tool calls coming back from the loop mean it ran out of rounds with the model still asking.
    if (result.reason == llamad::client::FinishReason::ToolCalls) {
        std::fprintf(stderr, "[tool] giving up after %d rounds of tool calls\n", kMaxToolRounds);
    }

    if (cancelled && !produced_text) {
        history.resize(user_turn);  // drop the user turn that produced nothing, tool traffic included
    }
    return !cancelled;
}

// --- the --demo-json struct ------------------------------------------------------------------
//
// A typed turn asks for a struct rather than prose: the struct's schema travels with the request,
// the daemon holds the model to it, and the JSON that streams back is parsed into the struct.

/// How sure the model says its verdict is.
enum class Confidence {
    low,     ///< Little confidence in the verdict.
    medium,  ///< Moderate confidence in the verdict.
    high,    ///< Strong confidence in the verdict.
};

/// Structured reply of the --demo-json turn.
struct Verdict {
    [[=desc{"whether the message is spam"}]]
    bool                     spam;        ///< Whether the message is spam.
    [[=desc{"short reasons for the verdict, most important first"}]]
    std::vector<std::string> reasons;     ///< Reasons for the verdict, most important first.
    [[=desc{"how sure the verdict is"}]]
    Confidence               confidence;  ///< Reported certainty of the verdict.
};

/// The message the demonstration turn asks about.
const char kDemoMessage[] =
    "Congratulations! You have WON a free cruise. Click the link now to claim your prize!";

/// Spell the reported confidence for terminal output.
const char * confidence_name(Confidence confidence) {
    switch (confidence) {
        case Confidence::low:    return "low";
        case Confidence::medium: return "medium";
        case Confidence::high:   return "high";
    }
    return "unknown";
}

/// Runs the single typed turn: the JSON streams to stdout as any reply does, then its fields
/// are printed one per line.
void run_json_demo(llamad::client::Client & client,
                   std::vector<llamad::client::ChatMessage> & history,
                   const llamad::client::SamplingParams & sampling) {
    history.push_back({"user", std::string("Is this message spam?\n\n") + kDemoMessage});

    const llamad::client::Typed<Verdict> reply =
        client.chat<Verdict>(history, sampling, [](const std::string & text) {
            std::fwrite(text.data(), 1, text.size(), stdout);
            std::fflush(stdout);
            return true;
        });

    std::fputc('\n', stdout);
    std::fflush(stdout);

    if (reply.value) {
        std::fprintf(stderr, "[demo-json] spam: %s\n", reply.value->spam ? "true" : "false");
        for (const std::string & reason : reply.value->reasons) {
            std::fprintf(stderr, "[demo-json] reason: %s\n", reason.c_str());
        }
        std::fprintf(stderr, "[demo-json] confidence: %s\n", confidence_name(reply.value->confidence));
    } else {
        std::fprintf(stderr, "[demo-json] no value: finish=%s\n", reason_name(reply.result.reason));
    }
    print_stats(reply.result);
}

}  // namespace

/// Run the executable.
/// @return Zero on success, two for invalid command-line usage, or one for a runtime failure.
int main(int argc, char ** argv) {
    Options               options;
    llamad::cli::HelpFlag help_flag;

    try {
        const std::vector<std::string> positional = llamad::cli::parse_flags(argc, argv, options, help_flag);
        if (!positional.empty()) {
            throw llamad::cli::FlagError("unknown argument '" + positional.front() + "'");
        }
        if (help_flag.help) {
            print_usage(argv[0]);
            return 0;
        }
        if (options.cancel_after && *options.cancel_after < 1) {
            throw llamad::cli::FlagError("--cancel-after needs a positive integer");
        }
        if (options.demo_json && (options.once || options.demo_tools)) {
            throw llamad::cli::FlagError("--demo-json cannot be combined with --once or --demo-tools");
        }
        if (!options.embed.empty() && (options.once || options.demo_tools || options.demo_json)) {
            throw llamad::cli::FlagError("--embed cannot be combined with --once, --demo-tools or --demo-json");
        }
    } catch (const llamad::cli::FlagError & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        print_usage(argv[0]);
        return 2;
    }

    llamad::client::SamplingParams sampling;
    sampling.temperature = options.temp;
    sampling.seed        = options.seed;
    sampling.max_tokens  = options.max_tokens;

    const std::string socket_path = options.socket.value_or(llamad::client::Client::default_socket_path());

    // Blocked before any other thread exists, gRPC's included, so every thread inherits the mask
    // and only watch_sigint ever sees the signal. A shell starts a background job with SIGINT
    // ignored, and an ignored signal never reaches sigwait, so the default action is restored;
    // blocked, it stays pending for sigwait rather than ending the process.
    std::signal(SIGINT, SIG_DFL);
    sigset_t sigint;
    sigemptyset(&sigint);
    sigaddset(&sigint, SIGINT);
    pthread_sigmask(SIG_BLOCK, &sigint, nullptr);
    std::thread(watch_sigint, sigint).detach();

    std::vector<llamad::client::ChatMessage> history;
    if (!options.system.empty()) {
        history.push_back({"system", options.system});
    }

    llamad::client::ToolSet tools;
    if (options.demo_tools) {
        tools.add<^^get_current_time>();
    }

    try {
        llamad::client::Client client(socket_path);

        if (!options.embed.empty()) {
            print_embeddings(client.embed(options.embed));
            return 0;
        }

        if (options.demo_json) {
            run_json_demo(client, history, sampling);
            return 0;
        }

        if (options.once) {
            history.push_back({"user", *options.once});
            run_turn(client, history, tools, sampling, options.cancel_after.value_or(0));
            return 0;
        }

        const llamad::client::ModelInfo info = client.get_model_info();
        std::fprintf(stderr, "[llamad-chat] %s on unix:%s\n", info.description.c_str(), socket_path.c_str());
        std::fprintf(stderr, "[llamad-chat] Ctrl-C cancels a reply; Ctrl-D at the prompt quits.\n");
        if (options.demo_tools) {
            std::fprintf(stderr, "[llamad-chat] tools offered: %s\n",
                         tools.definitions().front().name.c_str());
        }

        std::string line;
        while (true) {
            std::fputs("> ", stdout);
            std::fflush(stdout);
            if (!std::getline(std::cin, line)) {
                std::fputc('\n', stdout);
                break;  // EOF, or Ctrl-C handled in the signal handler
            }
            if (line.empty()) {
                continue;
            }
            history.push_back({"user", line});
            run_turn(client, history, tools, sampling, options.cancel_after.value_or(0));
        }
        return 0;
    } catch (const llamad::client::RpcError & e) {
        std::fprintf(stderr, "error: rpc failed (code %d): %s\n", e.code, e.what());
        return 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
