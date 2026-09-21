// The tool loop in Client::chat, against a scripted fake daemon on a private Unix socket.
// No model and no inference: the fake replays canned rounds, so what is under test is the
// history the loop builds, the requests it sends, its summed stats and where it stops.

#include "llamad/client.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include <grpcpp/grpcpp.h>

#include "llamad/v1/convert.h"
#include "llamad/v1/llamad.grpc.pb.h"

namespace {

namespace client = llamad::client;
namespace v1     = llamad::v1;

using llamad::client::desc;

int failures = 0;
int checks   = 0;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        ++checks;                                                                         \
        if (!(cond)) {                                                                    \
            ++failures;                                                                   \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                                 \
    } while (0)

#define CHECK_EQ(a, b)                                                                           \
    do {                                                                                         \
        ++checks;                                                                                \
        const std::string lhs_ = (a);                                                            \
        const std::string rhs_ = (b);                                                            \
        if (lhs_ != rhs_) {                                                                      \
            ++failures;                                                                          \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s == %s\n", __FILE__, __LINE__, #a, #b); \
            std::fprintf(stderr, "    left:  '%s'\n", lhs_.c_str());                             \
            std::fprintf(stderr, "    right: '%s'\n", rhs_.c_str());                             \
        }                                                                                        \
    } while (0)

// Tools run in the calling thread, so this records what ran, in order, without a lock.
std::string tool_log;

[[=desc{"Shout a word."}]]
std::string shout([[=desc{"The word to shout."}]] std::string word) {
    tool_log += "shout:" + word + ";";
    return word + "!";
}

[[=desc{"Add two numbers."}]]
int add([[=desc{"First addend."}]] int a, [[=desc{"Second addend."}]] int b) {
    tool_log += "add:" + std::to_string(a) + "+" + std::to_string(b) + ";";
    return a + b;
}

client::ToolSet make_tools() {
    client::ToolSet tools;
    tools.add<^^shout>();
    tools.add<^^add>();
    return tools;
}

// One round of the stream shape the daemon promises: zero or more text chunks, then exactly one
// final chunk carrying the reason, the stats and — for a tool round — the calls.
struct Round {
    std::vector<std::string>      text;
    v1::FinishReason              reason = v1::FINISH_REASON_EOG;
    std::vector<client::ToolCall> tool_calls;
    client::GenerateStats         stats;
};

class FakeDaemon final : public v1::Llama::Service {
public:
    explicit FakeDaemon(std::vector<Round> script) : script_(std::move(script)) {}

    grpc::Status Chat(grpc::ServerContext *,
                      const v1::ChatRequest * request,
                      grpc::ServerWriter<v1::GenerateChunk> * writer) override {
        Round round;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::vector<client::ChatMessage> messages;
            for (const v1::ChatMessage & m : request->messages()) {
                messages.push_back(llamad::wire::from_proto<client::ChatMessage>(m));
            }
            requests_.push_back(std::move(messages));
            // A script shorter than the loop is long replays its last round, which is how a fake
            // that never stops asking for a tool is written.
            round = script_[std::min(requests_.size() - 1, script_.size() - 1)];
        }

        for (const std::string & text : round.text) {
            v1::GenerateChunk chunk;
            chunk.set_text(text);
            writer->Write(chunk);
        }

        v1::GenerateChunk final_chunk;
        final_chunk.set_finish_reason(round.reason);
        llamad::wire::to_proto(round.stats, final_chunk.mutable_stats());
        for (const client::ToolCall & call : round.tool_calls) {
            llamad::wire::to_proto(call, final_chunk.add_tool_calls());
        }
        writer->Write(final_chunk);
        return grpc::Status::OK;
    }

    // Chat runs on a gRPC server thread, so the assertions read a copy taken under the lock.
    std::vector<std::vector<client::ChatMessage>> requests() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return requests_;
    }

private:
    mutable std::mutex                            mutex_;
    std::vector<Round>                            script_;
    std::vector<std::vector<client::ChatMessage>> requests_;
};

// A socket of this test's own, gone again when the harness is.
struct TempSocket {
    std::string dir;
    std::string path;

    TempSocket() {
        char pattern[] = "/tmp/llamad-client-chat-XXXXXX";
        const char * made = mkdtemp(pattern);
        if (made == nullptr) {
            std::perror("mkdtemp");
            std::abort();
        }
        dir  = made;
        path = dir + "/llamad.sock";
    }

    ~TempSocket() {
        ::unlink(path.c_str());
        ::rmdir(dir.c_str());
    }

    TempSocket(const TempSocket &)             = delete;
    TempSocket & operator=(const TempSocket &) = delete;
};

std::unique_ptr<grpc::Server> start_server(const std::string & socket_path, FakeDaemon * daemon) {
    grpc::ServerBuilder builder;
    builder.AddListeningPort("unix:" + socket_path, grpc::InsecureServerCredentials());
    builder.RegisterService(daemon);
    return builder.BuildAndStart();
}

struct Harness {
    TempSocket                    socket;
    FakeDaemon                    daemon;
    std::unique_ptr<grpc::Server> server;
    client::Client                client;

    explicit Harness(std::vector<Round> script)
        : daemon(std::move(script)), server(start_server(socket.path, &daemon)), client(socket.path) {
        CHECK(server != nullptr);
        tool_log.clear();
    }

    ~Harness() {
        server->Shutdown();
        server->Wait();
    }
};

const client::SamplingParams params;

Round tool_round(std::vector<client::ToolCall> calls) {
    Round round;
    round.reason     = v1::FINISH_REASON_TOOL_CALLS;
    round.tool_calls = std::move(calls);
    return round;
}

void test_one_tool_round_then_an_answer() {
    Round asking  = tool_round({{"call_0", "shout", R"({"word":"Oslo"})"}});
    asking.text   = {"Let me look."};
    Round answer;
    answer.text = {"It said ", "Oslo!"};

    Harness harness({asking, answer});

    std::vector<client::ChatMessage> history = {{"user", "shout Oslo"}};
    std::string                      streamed;
    const client::GenerateResult     result =
        harness.client.chat(history, make_tools(), params, [&](const std::string & text) {
            streamed += text;
            return true;
        });

    CHECK(result.reason == client::FinishReason::Eog);
    CHECK(result.tool_calls.empty());
    CHECK_EQ(streamed, "Let me look.It said Oslo!");
    CHECK_EQ(tool_log, "shout:Oslo;");

    CHECK(history.size() == 4);
    CHECK_EQ(history[1].role, "assistant");
    CHECK_EQ(history[1].content, "Let me look.");
    CHECK(history[1].tool_calls.size() == 1);
    CHECK_EQ(history[1].tool_calls[0].name, "shout");
    CHECK_EQ(history[2].role, "tool");
    CHECK_EQ(history[2].content, "Oslo!");
    CHECK_EQ(history[2].tool_call_id, "call_0");
    CHECK_EQ(history[3].role, "assistant");
    CHECK_EQ(history[3].content, "It said Oslo!");
    CHECK(history[3].tool_calls.empty());

    // The second request carries the round the loop just ran, so the model sees its own call.
    const std::vector<std::vector<client::ChatMessage>> requests = harness.daemon.requests();
    CHECK(requests.size() == 2);
    CHECK(requests[0].size() == 1);
    CHECK(requests[1].size() == 3);
    CHECK_EQ(requests[1][1].role, "assistant");
    CHECK(requests[1][1].tool_calls.size() == 1);
    CHECK_EQ(requests[1][1].tool_calls[0].id, "call_0");
    CHECK_EQ(requests[1][2].role, "tool");
    CHECK_EQ(requests[1][2].content, "Oslo!");
    CHECK_EQ(requests[1][2].tool_call_id, "call_0");
}

void test_two_calls_in_one_round() {
    const Round asking = tool_round({{"call_0", "add", R"({"a":2,"b":3})"},
                                     {"call_1", "shout", R"({"word":"hi"})"}});
    Round answer;
    answer.text = {"done"};

    Harness harness({asking, answer});

    std::vector<client::ChatMessage> history = {{"user", "add and shout"}};
    harness.client.chat(history, make_tools(), params, nullptr);

    CHECK_EQ(tool_log, "add:2+3;shout:hi;");
    CHECK(history.size() == 5);
    CHECK_EQ(history[1].role, "assistant");
    CHECK_EQ(history[1].content, "");
    CHECK(history[1].tool_calls.size() == 2);
    CHECK_EQ(history[2].content, "5");  // a tool returning anything but a string answers in JSON
    CHECK_EQ(history[2].tool_call_id, "call_0");
    CHECK_EQ(history[3].content, "hi!");
    CHECK_EQ(history[3].tool_call_id, "call_1");
    CHECK_EQ(history[4].role, "assistant");
    CHECK_EQ(history[4].content, "done");
}

void test_stats_are_summed_over_rounds() {
    Round asking        = tool_round({{"call_0", "shout", R"({"word":"Oslo"})"}});
    asking.stats        = {10, 5, 1.5, 2.5};
    Round answer;
    answer.text  = {"ok"};
    answer.stats = {20, 7, 3.0, 4.0};

    Harness harness({asking, answer});

    std::vector<client::ChatMessage> history = {{"user", "shout Oslo"}};
    const client::GenerateResult result = harness.client.chat(history, make_tools(), params, nullptr);

    CHECK(result.stats.prompt_tokens == 30);
    CHECK(result.stats.completion_tokens == 12);
    CHECK(result.stats.prompt_ms == 4.5);
    CHECK(result.stats.completion_ms == 6.5);
}

// A model that keeps asking spends the budget and stops, with the last round's results in history.
void test_round_limit_stops_the_loop() {
    Harness harness({tool_round({{"call_0", "shout", R"({"word":"again"})"}})});

    std::vector<client::ChatMessage> history = {{"user", "keep going"}};
    const client::GenerateResult     result =
        harness.client.chat(history, make_tools(), params, nullptr, /*max_rounds*/ 2);

    CHECK(result.reason == client::FinishReason::ToolCalls);
    CHECK(result.tool_calls.size() == 1);
    CHECK(harness.daemon.requests().size() == 2);
    CHECK_EQ(tool_log, "shout:again;shout:again;");

    CHECK(history.size() == 5);
    CHECK_EQ(history.back().role, "tool");
    CHECK_EQ(history.back().content, "again!");
    CHECK_EQ(history.back().tool_call_id, "call_0");
}

// Cancelling mid-round ends the loop there: the tool the model asked for never runs.
void test_cancellation_stops_the_loop() {
    Round asking = tool_round({{"call_0", "shout", R"({"word":"never"})"}});
    asking.text  = {"thinking"};
    asking.stats = {10, 5, 1.5, 2.5};
    Round answer;
    answer.text = {"unreachable"};

    Harness harness({asking, answer});

    std::vector<client::ChatMessage> history = {{"user", "go"}};
    const client::GenerateResult     result =
        harness.client.chat(history, make_tools(), params, [](const std::string &) { return false; });

    CHECK(result.reason == client::FinishReason::Cancelled);
    CHECK(harness.daemon.requests().size() == 1);

    // The fake sends its final chunk whether or not anyone is still reading, and a cancelled
    // stream delivers none of it: no calls beside a reason that is not ToolCalls, and no stats.
    CHECK(result.tool_calls.empty());
    CHECK(result.stats.prompt_tokens == 0 && result.stats.completion_tokens == 0);
    CHECK_EQ(tool_log, "");

    CHECK(history.size() == 2);
    CHECK_EQ(history[1].role, "assistant");
    CHECK_EQ(history[1].content, "thinking");
    CHECK(history[1].tool_calls.empty());
}

void test_max_rounds_must_be_positive() {
    Harness harness({Round{}});

    std::vector<client::ChatMessage> history = {{"user", "hello"}};
    bool                             threw   = false;
    try {
        harness.client.chat(history, make_tools(), params, nullptr, /*max_rounds*/ 0);
    } catch (const std::invalid_argument &) {
        threw = true;
    }

    CHECK(threw);
    CHECK(harness.daemon.requests().empty());
    CHECK(history.size() == 1);
}

}  // namespace

int main() {
    test_one_tool_round_then_an_answer();
    test_two_calls_in_one_round();
    test_stats_are_summed_over_rounds();
    test_round_limit_stops_the_loop();
    test_cancellation_stops_the_loop();
    test_max_rounds_must_be_positive();

    std::fprintf(stderr, "%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
