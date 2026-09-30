/** @file
 * @brief Model-backed cancellation, cache reuse and raw transport activity checks.
 *
 * For CPU, run against a private daemon using the same model with
 * --ctx 8192 --threads 1 --ngl 0:
 *   activity_smoke model.gguf /tmp/llamad-test.sock 0
 * For GPU offload, use daemon --ngl 99 and harness final argument 99.
 * The last argument controls only the direct engine's GPU layer count.
 */

#include "check.h"
#include "engine.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "llamad/v1/llamad.grpc.pb.h"

namespace {

namespace v1 = llamad::v1;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

// Hold the context at a text callback, so a waiter really competes with active inference.
class HeldGeneration {
public:
    explicit HeldGeneration(llamad::Engine & engine) : thread_([&] {
        llamad::SamplingParams params;
        params.temperature = 0;
        params.max_tokens = 32;
        engine.generate("Write a sentence about cats:", params, [&](const std::string &) {
            std::unique_lock lock(mutex_);
            ready_ = true;
            wake_.notify_all();
            wake_.wait(lock, [&] { return released_; });
            return false;
        });
    }) {
        std::unique_lock lock(mutex_);
        if (!wake_.wait_for(lock, 30s, [&] { return ready_; })) {
            released_ = true;
            wake_.notify_all();
            throw std::runtime_error("holder produced no text");
        }
    }

    ~HeldGeneration() {
        {
            std::lock_guard lock(mutex_);
            released_ = true;
        }
        wake_.notify_all();
        thread_.join();
    }

private:
    std::mutex mutex_;
    std::condition_variable wake_;
    bool ready_ = false;
    bool released_ = false;
    std::jthread thread_;
};

std::string long_prompt() {
    std::string prompt;
    for (int i = 0; i < 6000; ++i) {
        prompt += " potato";
    }
    return prompt;
}

void check_engine(llamad::Engine & engine, const std::string & prompt) {
    llamad::SamplingParams params;
    params.temperature = 0;
    params.max_tokens = 1;
    int chunks = 0;
    const auto started = Clock::now();
    const auto partial = engine.generate(prompt, params, [&](const std::string &) {
        ++chunks;
        return true;
    }, [&] { return Clock::now() - started < 100ms; });
    CHECK(partial.reason == llamad::FinishReason::Cancelled);
    CHECK(partial.stats.completion_tokens == 0);
    CHECK(chunks == 0);
    std::fprintf(stderr, "partial prefill: prompt=%d time=%.0fms wall=%.0fms\n",
                 partial.stats.prompt_tokens, partial.stats.prompt_ms,
                 std::chrono::duration<double, std::milli>(Clock::now() - started).count());

    // A pre-admission cancellation must leave that successful partial batch intact.
    const auto cancelled = engine.generate("unrelated", params, nullptr, [] { return false; });
    CHECK(cancelled.reason == llamad::FinishReason::Cancelled);
    CHECK(cancelled.stats.prompt_tokens == 0);
    CHECK(cancelled.stats.completion_tokens == 0);
    const auto resumed = engine.generate(prompt, params, nullptr);
    CHECK(resumed.reason != llamad::FinishReason::Cancelled);
    CHECK(resumed.stats.cached_prompt_tokens > 0);
    CHECK(resumed.stats.cached_prompt_tokens < resumed.stats.prompt_tokens - 1);
    std::fprintf(stderr, "resumed cache: %d/%d tokens\n", resumed.stats.cached_prompt_tokens,
                 resumed.stats.prompt_tokens);

    const auto shorter = engine.generate(prompt.substr(0, 5000), params, nullptr);
    CHECK(shorter.reason != llamad::FinishReason::Cancelled);
    CHECK(shorter.stats.cached_prompt_tokens == shorter.stats.prompt_tokens - 1);
    const auto unrelated = engine.generate("Write a word about rivers:", params, nullptr);
    CHECK(unrelated.reason != llamad::FinishReason::Cancelled);
    CHECK(unrelated.stats.cached_prompt_tokens == 0);

    HeldGeneration holder(engine);
    const auto queued_started = Clock::now();
    const auto queued = engine.generate(prompt, params, nullptr, [&] {
        return Clock::now() - queued_started < 100ms;
    });
    const auto queued_ms = std::chrono::duration<double, std::milli>(Clock::now() - queued_started).count();
    CHECK(queued.reason == llamad::FinishReason::Cancelled);
    CHECK(queued.stats.prompt_tokens == 0);
    CHECK(queued.stats.completion_tokens == 0);
    CHECK(queued_ms < 500);
    std::fprintf(stderr, "cancelled waiter: %.0fms, no prompt work\n", queued_ms);
}

struct ReadResult {
    int activities = 0;
    int finals = 0;
    int texts = 0;
    double max_gap_ms = 0;
    v1::GenerateChunk final;
    grpc::Status status;
    bool valid = true;
};

ReadResult read(v1::Llama::Stub & stub, const std::string & prompt, int cancel_after_activity = -1,
                int deadline_ms = 120000, bool cancel_after_text = false) {
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(deadline_ms));
    v1::GenerateRequest request;
    request.set_prompt(prompt);
    request.mutable_sampling()->set_temperature(0);
    request.mutable_sampling()->set_max_tokens(8);
    auto reader = stub.Generate(&context, request);
    ReadResult result;
    auto last = Clock::now();
    v1::GenerateChunk chunk;
    while (reader->Read(&chunk)) {
        const auto received = Clock::now();
        result.max_gap_ms = std::max(result.max_gap_ms,
            std::chrono::duration<double, std::milli>(received - last).count());
        last = received;
        if (chunk.finish_reason() != v1::FINISH_REASON_UNSPECIFIED) {
            ++result.finals;
            result.final = chunk;
        } else if (chunk.text().empty()) {
            result.valid &= !chunk.has_stats() && chunk.tool_calls().empty();
            ++result.activities;
        } else {
            ++result.texts;
        }
        result.valid &= result.finals <= 1;
        // Neither text nor activity can arrive after the accepted final.
        if (result.finals > 0) {
            result.valid &= chunk.finish_reason() != v1::FINISH_REASON_UNSPECIFIED;
        }
        if ((cancel_after_activity >= 0 && result.activities >= cancel_after_activity) ||
            (cancel_after_text && result.texts > 0)) {
            context.TryCancel();
        }
    }
    result.status = reader->Finish();
    return result;
}

void check_transport(const std::string & socket, const std::string & prompt) {
    const auto channel = grpc::CreateChannel("unix:" + socket, grpc::InsecureChannelCredentials());
    auto stub = v1::Llama::NewStub(channel);

    // A second request must remain observable while waiting behind uncached prefill.
    // Stop the first upon its first activity frame, before any user-visible text.
    ReadResult partial;
    std::jthread prefilling([&] { partial = read(*stub, prompt, 1); });
    std::this_thread::sleep_for(200ms);
    const auto expired = read(*stub, "Queue a different prompt", -1, 1200);
    prefilling.join();
    CHECK(partial.valid);
    CHECK(partial.status.error_code() == grpc::StatusCode::CANCELLED);
    CHECK(partial.activities == 1);
    CHECK(partial.texts == 0);
    CHECK(partial.finals == 0);
    CHECK(partial.max_gap_ms < 1500);
    CHECK(expired.valid);
    CHECK(expired.status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED);
    CHECK(expired.activities >= 1);
    CHECK(expired.texts == 0);
    CHECK(expired.finals == 0);
    CHECK(expired.max_gap_ms < 1500);

    const auto resumed = read(*stub, prompt);
    CHECK(resumed.valid);
    CHECK(resumed.status.ok());
    CHECK(resumed.finals == 1);
    CHECK(resumed.final.stats().cached_prompt_tokens() > 0);
    CHECK(resumed.final.stats().cached_prompt_tokens() < resumed.final.stats().prompt_tokens() - 1);
    CHECK(resumed.max_gap_ms < 1500);
    std::fprintf(stderr, "raw prefill: %d activity frames, maximum gap %.0fms, cached %d/%d tokens\n",
                 resumed.activities, resumed.max_gap_ms, resumed.final.stats().cached_prompt_tokens(),
                 resumed.final.stats().prompt_tokens());

    const auto shorter = read(*stub, prompt.substr(0, 5000));
    CHECK(shorter.status.ok());
    CHECK(shorter.finals == 1);
    CHECK(shorter.final.stats().cached_prompt_tokens() == shorter.final.stats().prompt_tokens() - 1);
    const auto unrelated = read(*stub, "Write a word about rivers:");
    CHECK(unrelated.status.ok());
    CHECK(unrelated.finals == 1);
    CHECK(unrelated.final.stats().cached_prompt_tokens() == 0);

    const auto cancelled_text = read(*stub, "Write a sentence about cats:", -1, 120000, true);
    CHECK(cancelled_text.status.error_code() == grpc::StatusCode::CANCELLED);
    CHECK(cancelled_text.texts > 0);
    CHECK(cancelled_text.finals == 0);

    // A refusal throws after the timer starts; its lifetime must end with that RPC.
    const auto invalid = read(*stub, "");
    CHECK(invalid.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
    CHECK(invalid.finals == 0);
    const auto after_error = read(*stub, "Say hello:");
    CHECK(after_error.status.ok());
    CHECK(after_error.finals == 1);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <private-daemon-socket> <gpu-layers>\n", argv[0]);
        return 2;
    }
    try {
        llamad::EngineConfig config;
        config.model_path = argv[1];
        config.n_gpu_layers = std::stoi(argv[3]);
        config.n_ctx = 8192;
        config.n_threads = 1;
        llamad::Engine engine(config);
        const auto prompt = long_prompt();
        check_engine(engine, prompt);
        check_transport(argv[2], prompt);
        return tests::report();
    } catch (const std::exception & error) {
        std::fprintf(stderr, "activity smoke failed: %s\n", error.what());
        return 1;
    }
}
