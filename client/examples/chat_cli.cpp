/** @file
 * @brief Interactive chat client and an annotated time-tool example.
 *
 * llamad-chat: a multi-turn chat REPL against a running llamad daemon.
 * It uses only <llamad/client.h>: no gRPC or protobuf headers anywhere.
 */

#include <atomic>
#include <cctype>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <unistd.h>

#include "flags.h"
#include "llamad/client.h"

namespace {

// Set by the SIGINT handler while a generation is running; the chunk callback
// turns it into a cancel. At the prompt there is nothing to cancel, so the
// handler exits the process instead (via _exit, which is async-signal-safe).
/// Whether SIGINT should cancel a generation instead of exiting the prompt.
std::atomic<bool> g_generating{false};
/// Cancellation flag read by the synchronous chunk callback.
std::atomic<bool> g_interrupted{false};

/// Request cancellation during generation; otherwise exit using signal-safe operations.
void on_sigint(int) {
    if (g_generating.load()) {
        g_interrupted.store(true);
        return;
    }
    const char msg[] = "\n";
    ssize_t ignored = ::write(STDOUT_FILENO, msg, sizeof(msg) - 1);
    (void) ignored;
    ::_exit(130);
}

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
    std::string socket;  ///< Daemon socket (default: the client library default).

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

    g_interrupted.store(false);
    g_generating.store(true);

    llamad::client::GenerateResult result{};
    try {
        result = client.chat(history, tools, sampling, [&](const std::string & text) {
            produced_text = true;
            std::fwrite(text.data(), 1, text.size(), stdout);
            std::fflush(stdout);
            ++chunks;
            if (cancel_after > 0 && chunks >= cancel_after) {
                return false;
            }
            return !g_interrupted.load();
        }, kMaxToolRounds);
    } catch (...) {
        g_generating.store(false);
        throw;
    }
    g_generating.store(false);

    std::fputc('\n', stdout);
    std::fflush(stdout);

    const bool cancelled = result.reason == llamad::client::FinishReason::Cancelled;
    if (cancelled) {
        std::fprintf(stderr, "[cancelled]\n");
    }
    std::fprintf(stderr,
                 "[stats] finish=%s prompt_tokens=%d completion_tokens=%d prompt_ms=%.1f completion_ms=%.1f\n",
                 reason_name(result.reason),
                 result.stats.prompt_tokens,
                 result.stats.completion_tokens,
                 result.stats.prompt_ms,
                 result.stats.completion_ms);

    // Tool calls coming back from the loop mean it ran out of rounds with the model still asking.
    if (result.reason == llamad::client::FinishReason::ToolCalls) {
        std::fprintf(stderr, "[tool] giving up after %d rounds of tool calls\n", kMaxToolRounds);
    }

    if (cancelled && !produced_text) {
        history.resize(user_turn);  // drop the user turn that produced nothing, tool traffic included
    }
    return !cancelled;
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
    } catch (const llamad::cli::FlagError & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        print_usage(argv[0]);
        return 2;
    }

    llamad::client::SamplingParams sampling;
    sampling.temperature = options.temp;
    sampling.seed        = options.seed;
    sampling.max_tokens  = options.max_tokens;

    const std::string socket_path =
        options.socket.empty() ? llamad::client::Client::default_socket_path() : options.socket;

    struct sigaction sa {};
    sa.sa_handler = on_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;  // no SA_RESTART: interrupt the blocking read at the prompt
    sigaction(SIGINT, &sa, nullptr);

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
