// llamad-chat: a multi-turn chat REPL against a running llamad daemon.
// It uses only <llamad/client.h>: no gRPC or protobuf headers anywhere.

#include <atomic>
#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <unistd.h>

#include "llamad/client.h"

namespace {

// Set by the SIGINT handler while a generation is running; the chunk callback
// turns it into a cancel. At the prompt there is nothing to cancel, so the
// handler exits the process instead (via _exit, which is async-signal-safe).
std::atomic<bool> g_generating{false};
std::atomic<bool> g_interrupted{false};

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

void print_usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s [options]\n"
                 "\n"
                 "  --socket PATH       daemon socket (default: the client library default)\n"
                 "  --system TEXT       system prompt for the conversation\n"
                 "  --temp F            sampling temperature (0 = greedy)\n"
                 "  --seed N            sampling seed\n"
                 "  --max-tokens N      cap on generated tokens per turn\n"
                 "  --once PROMPT       run a single non-interactive turn and exit\n"
                 "  --demo-tools        offer the built-in get_current_time tool and run the\n"
                 "                      execute-and-resend loop for any call the model makes\n"
                 "  --help              show this message\n",
                 argv0);
}

bool parse_long(const char * text, long * out) {
    char * end = nullptr;
    errno      = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    *out = value;
    return true;
}

bool parse_float(const char * text, float * out) {
    char * end = nullptr;
    errno      = 0;
    const float value = std::strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    *out = value;
    return true;
}

// --- the --demo-tools tool -------------------------------------------------------------------
//
// A tool is a C++ function: its identifier names it, its desc annotations describe it and its
// parameters, and its parameter list is the argument schema the daemon constrains the model to.
// The result struct travels back to the model as JSON.

using llamad::client::desc;

// tzset() quietly falls back to UTC for a zone it cannot resolve; checking the zone file
// first is what lets the result say so.
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

struct CurrentTime {
    std::string                timezone;
    std::string                time;
    std::optional<std::string> note;  // why the answer is not in the zone that was asked for
};

// Local time in an IANA zone, via the C library: TZ is set for the call and restored after.
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
const int kMaxToolRounds = 8;

// Runs one turn: streams the reply to stdout while the client appends every assistant and
// "tool" turn it takes to the history. Returns false if the turn was cancelled.
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

int main(int argc, char ** argv) {
    std::string socket_path;
    std::string system_prompt;
    std::string once_prompt;
    bool        have_once    = false;
    bool        demo_tools   = false;
    long        cancel_after = 0;  // hidden --cancel-after N, for testing cancellation

    llamad::client::SamplingParams sampling;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--socket") {
            socket_path = next("--socket");
        } else if (arg == "--system") {
            system_prompt = next("--system");
        } else if (arg == "--once") {
            once_prompt = next("--once");
            have_once   = true;
        } else if (arg == "--demo-tools") {
            demo_tools = true;
        } else if (arg == "--temp") {
            float value = 0;
            if (!parse_float(next("--temp"), &value)) {
                std::fprintf(stderr, "error: --temp needs a number\n");
                return 2;
            }
            sampling.temperature = value;
        } else if (arg == "--seed") {
            long value = 0;
            if (!parse_long(next("--seed"), &value) || value < 0) {
                std::fprintf(stderr, "error: --seed needs a non-negative integer\n");
                return 2;
            }
            sampling.seed = static_cast<uint32_t>(value);
        } else if (arg == "--max-tokens") {
            long value = 0;
            if (!parse_long(next("--max-tokens"), &value)) {
                std::fprintf(stderr, "error: --max-tokens needs an integer\n");
                return 2;
            }
            sampling.max_tokens = static_cast<int32_t>(value);
        } else if (arg == "--cancel-after") {
            if (!parse_long(next("--cancel-after"), &cancel_after) || cancel_after < 1) {
                std::fprintf(stderr, "error: --cancel-after needs a positive integer\n");
                return 2;
            }
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    if (socket_path.empty()) {
        socket_path = llamad::client::Client::default_socket_path();
    }

    struct sigaction sa {};
    sa.sa_handler = on_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;  // no SA_RESTART: interrupt the blocking read at the prompt
    sigaction(SIGINT, &sa, nullptr);

    std::vector<llamad::client::ChatMessage> history;
    if (!system_prompt.empty()) {
        history.push_back({"system", system_prompt});
    }

    llamad::client::ToolSet tools;
    if (demo_tools) {
        tools.add<^^get_current_time>();
    }

    try {
        llamad::client::Client client(socket_path);

        if (have_once) {
            history.push_back({"user", once_prompt});
            run_turn(client, history, tools, sampling, cancel_after);
            return 0;
        }

        const llamad::client::ModelInfo info = client.get_model_info();
        std::fprintf(stderr, "[llamad-chat] %s on unix:%s\n", info.description.c_str(), socket_path.c_str());
        std::fprintf(stderr, "[llamad-chat] Ctrl-C cancels a reply; Ctrl-D at the prompt quits.\n");
        if (demo_tools) {
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
            run_turn(client, history, tools, sampling, cancel_after);
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
