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
// The client library has no JSON dependency and must not gain one, so the demo reads its one
// string argument by hand and writes its result by hand. Enough for a worked example, not a
// JSON parser: a real client would use one.

// Finds "key": "value" and unescapes \" and \\ in the value.
bool json_string_field(const std::string & json, const std::string & key, std::string * out) {
    const std::string needle = "\"" + key + "\"";
    for (size_t at = json.find(needle); at != std::string::npos; at = json.find(needle, at + 1)) {
        size_t i = at + needle.size();
        while (i < json.size() && std::isspace(static_cast<unsigned char>(json[i]))) { ++i; }
        if (i >= json.size() || json[i] != ':') { continue; }
        ++i;
        while (i < json.size() && std::isspace(static_cast<unsigned char>(json[i]))) { ++i; }
        if (i >= json.size() || json[i] != '"') { continue; }

        std::string value;
        for (++i; i < json.size() && json[i] != '"'; ++i) {
            if (json[i] == '\\' && i + 1 < json.size()) {
                ++i;  // only \" and \\ can appear in a timezone name
            }
            value += json[i];
        }
        if (i >= json.size()) { return false; }  // unterminated string
        *out = value;
        return true;
    }
    return false;
}

std::string json_escape(const std::string & text) {
    std::string out;
    for (const char c : text) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char escape[7];
                    std::snprintf(escape, sizeof(escape), "\\u%04x", static_cast<unsigned char>(c));
                    out += escape;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

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

// Local time in an IANA zone, via the C library: TZ is set for the call and restored after.
std::string get_current_time(const std::string & arguments_json) {
    std::string zone;
    json_string_field(arguments_json, "timezone", &zone);

    std::string note;
    if (!zone_is_known(zone)) {
        note = zone.empty() ? "no timezone given, answering in UTC" : "unknown timezone '" + zone + "', answering in UTC";
        zone = "UTC";
    }

    const char *      previous = std::getenv("TZ");
    const bool        had_tz   = previous != nullptr;
    const std::string saved    = had_tz ? previous : "";

    ::setenv("TZ", zone.c_str(), 1);
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

    std::string result = "{\"timezone\":\"" + json_escape(zone) + "\",\"time\":\"" + json_escape(stamp) + "\"";
    if (!note.empty()) {
        result += ",\"note\":\"" + json_escape(note) + "\"";
    }
    return result + "}";
}

llamad::client::Tool demo_tool() {
    return {"get_current_time",
            "Get the current date and time in a given IANA timezone.",
            "{\"type\":\"object\",\"properties\":{\"timezone\":{\"type\":\"string\","
            "\"description\":\"IANA timezone, e.g. Europe/Paris\"}},\"required\":[\"timezone\"]}"};
}

// A tool result is always a string for the model: an unknown name is reported, not fatal.
std::string run_tool(const llamad::client::ToolCall & call) {
    if (call.name == "get_current_time") {
        return get_current_time(call.arguments_json);
    }
    return "{\"error\":\"no such tool: " + json_escape(call.name) + "\"}";
}

// Rounds of tool calls per turn. A model that keeps asking is looping; stop instead.
const long kMaxToolRounds = 8;

// Runs one turn: streams the reply to stdout and appends it to the history. While the model
// asks for tools, each round appends its tool_calls turn plus one "tool" turn per call and
// generates again. Returns false if the turn was cancelled.
bool run_turn(llamad::client::Client & client,
              std::vector<llamad::client::ChatMessage> & history,
              const std::vector<llamad::client::Tool> & tools,
              const llamad::client::SamplingParams & sampling,
              long cancel_after) {
    // The user turn this reply belongs to, so a cancelled empty reply can drop it along with
    // whatever tool traffic the turn had already accumulated.
    const size_t user_turn = history.empty() ? 0 : history.size() - 1;

    for (long round = 0; round < kMaxToolRounds; ++round) {
        std::string reply;
        long        chunks = 0;

        g_interrupted.store(false);
        g_generating.store(true);

        llamad::client::GenerateResult result{};
        try {
            result = client.chat(history, tools, sampling, [&](const std::string & text) {
                reply += text;
                std::fwrite(text.data(), 1, text.size(), stdout);
                std::fflush(stdout);
                ++chunks;
                if (cancel_after > 0 && chunks >= cancel_after) {
                    return false;
                }
                return !g_interrupted.load();
            });
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

        if (result.reason == llamad::client::FinishReason::ToolCalls && !result.tool_calls.empty()) {
            history.push_back({"assistant", reply, result.tool_calls, ""});
            for (const llamad::client::ToolCall & call : result.tool_calls) {
                const std::string output = run_tool(call);
                std::fprintf(stderr, "[tool] %s(%s) -> %s\n",
                             call.name.c_str(), call.arguments_json.c_str(), output.c_str());
                history.push_back({"tool", output, {}, call.id});
            }
            continue;
        }

        if (!reply.empty()) {
            history.push_back({"assistant", reply});
        } else if (cancelled) {
            history.resize(user_turn);  // drop the user turn that produced nothing
        }
        return !cancelled;
    }

    std::fprintf(stderr, "[tool] giving up after %ld rounds of tool calls\n", kMaxToolRounds);
    return true;
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

    std::vector<llamad::client::Tool> tools;
    if (demo_tools) {
        tools.push_back(demo_tool());
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
            std::fprintf(stderr, "[llamad-chat] tools offered: %s\n", tools.front().name.c_str());
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
