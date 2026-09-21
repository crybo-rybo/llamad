/** @file
 * @brief Compile-time and runtime checks that wire messages and plain structs mirror exactly.
 *
 * The structs in engine.h, chat_format.h and client.h mirror messages in llamad.proto field for
 * field. This is the one place all four are visible at once, so it is where that is checked: a
 * field added, removed or renamed on either side fails the build here rather than at a call site.
 */

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include "chat_format.h"
#include "engine.h"
#include "llamad/client.h"
#include "llamad/v1/convert.h"
#include "llamad/v1/llamad.pb.h"

namespace {

namespace client = llamad::client;
namespace v1     = llamad::v1;
namespace wire   = llamad::wire;

static_assert(wire::mirrors<client::ModelInfo, v1::ModelInfo>());
static_assert(wire::mirrors<client::SamplingParams, v1::SamplingParams>());
static_assert(wire::mirrors<client::GenerateStats, v1::GenerateStats>());
static_assert(wire::mirrors<client::Tool, v1::Tool>());
static_assert(wire::mirrors<client::ToolCall, v1::ToolCall>());
static_assert(wire::mirrors<client::ChatMessage, v1::ChatMessage>());

static_assert(wire::mirrors<llamad::ModelInfo, v1::ModelInfo>());
static_assert(wire::mirrors<llamad::GenerateStats, v1::GenerateStats>());
static_assert(wire::mirrors<llamad::Tool, v1::Tool>());
static_assert(wire::mirrors<llamad::ToolCall, v1::ToolCall>());
static_assert(wire::mirrors<llamad::ChatMessage, v1::ChatMessage>());

// And the guard itself. Each struct below differs from the message in exactly one way that the
// converters would carry out silently, or not at all; drifted() is mirrors() with the compile
// error it raises caught and reported as a value instead.
static_assert(!wire::drifted<client::ModelInfo, v1::ModelInfo>());

struct NarrowedParams {
    std::string description;
    uint32_t    n_params          = 0;  // the message says uint64
    uint64_t    size_bytes        = 0;
    uint32_t    n_ctx             = 0;
    uint32_t    n_ctx_train       = 0;
    bool        has_chat_template = false;
};
static_assert(wire::drifted<NarrowedParams, v1::ModelInfo>());

struct OptionalNCtx {
    std::string             description;
    uint64_t                n_params   = 0;
    uint64_t                size_bytes = 0;
    std::optional<uint32_t> n_ctx;  // the message's n_ctx has no presence
    uint32_t                n_ctx_train       = 0;
    bool                    has_chat_template = false;
};
static_assert(wire::drifted<OptionalNCtx, v1::ModelInfo>());

struct PlainMaxTokens {
    std::optional<float>     temperature;
    std::optional<int32_t>   top_k;
    std::optional<float>     top_p;
    std::optional<float>     min_p;
    std::optional<uint32_t>  seed;
    int32_t                  max_tokens = 0;  // the message says `optional int32`
    std::vector<std::string> stop;
};
static_assert(wire::drifted<PlainMaxTokens, v1::SamplingParams>());

struct ShortTool {
    std::string name;
    std::string description;  // no parameters_json_schema
};
static_assert(wire::drifted<ShortTool, v1::Tool>());

struct LongTool {
    std::string name;
    std::string description;
    std::string parameters_json_schema;
    std::string returns_json_schema;  // the message has no such field
};
static_assert(wire::drifted<LongTool, v1::Tool>());

// Instantiating enum_cast without a fallback demands a twin for every value of the source enum,
// so these also fail the build if either FinishReason gains a value llamad.proto does not name.
static_assert(wire::enum_cast<v1::FinishReason>(llamad::FinishReason::Cancelled) ==
              v1::FINISH_REASON_CANCELLED);
static_assert(wire::enum_cast<v1::FinishReason>(client::FinishReason::ToolCalls) ==
              v1::FINISH_REASON_TOOL_CALLS);

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

void test_chat_message_round_trip() {
    client::ChatMessage message;
    message.role         = "assistant";
    message.content      = "let me look that up";
    message.tool_call_id = "call_0";
    message.tool_calls.push_back({"call_1", "get_time", R"({"timezone":"Europe/Paris"})"});
    message.tool_calls.push_back({"call_2", "get_weather", R"({"city":"Oslo"})"});

    v1::ChatMessage proto;
    wire::to_proto(message, &proto);

    CHECK(proto.role() == "assistant");
    CHECK(proto.content() == "let me look that up");
    CHECK(proto.tool_call_id() == "call_0");
    CHECK(proto.tool_calls().size() == 2);
    CHECK(proto.tool_calls(1).name() == "get_weather");
    CHECK(proto.tool_calls(1).arguments_json() == R"({"city":"Oslo"})");

    const client::ChatMessage back = wire::from_proto<client::ChatMessage>(proto);
    CHECK(back.role == message.role);
    CHECK(back.content == message.content);
    CHECK(back.tool_call_id == message.tool_call_id);
    CHECK(back.tool_calls.size() == 2);
    CHECK(back.tool_calls[0].id == "call_1");
    CHECK(back.tool_calls[0].name == "get_time");
    CHECK(back.tool_calls[1].arguments_json == R"({"city":"Oslo"})");
}

void test_sampling_params_round_trip() {
    client::SamplingParams params;
    params.temperature = 0.25f;
    params.max_tokens  = 128;
    params.stop        = {"</done>", "\n\n"};

    v1::SamplingParams proto;
    wire::to_proto(params, &proto);

    CHECK(proto.has_temperature());
    CHECK(proto.temperature() == 0.25f);
    CHECK(proto.has_max_tokens());
    CHECK(proto.max_tokens() == 128);

    // An optional the caller left alone must stay absent on the wire, or the daemon's own
    // default would be overwritten with a zero nobody asked for.
    CHECK(!proto.has_top_k());
    CHECK(!proto.has_top_p());
    CHECK(!proto.has_min_p());
    CHECK(!proto.has_seed());

    CHECK(proto.stop().size() == 2);
    CHECK(proto.stop(0) == "</done>");

    const client::SamplingParams back = wire::from_proto<client::SamplingParams>(proto);
    CHECK(back.temperature.has_value());
    CHECK(back.max_tokens == 128);
    CHECK(!back.top_k.has_value());
    CHECK(!back.seed.has_value());
    CHECK(back.stop == params.stop);
}

void test_finish_reason_names() {
    CHECK(std::string(wire::value_name(v1::FINISH_REASON_TOOL_CALLS)) == "TOOL_CALLS");
    CHECK(std::string(wire::value_name(v1::FINISH_REASON_EOG)) == "EOG");
    CHECK(std::string(wire::value_name(v1::FINISH_REASON_UNSPECIFIED)) == "UNSPECIFIED");
    CHECK(wire::enum_cast(v1::FINISH_REASON_TOOL_CALLS, client::FinishReason::Eog) ==
          client::FinishReason::ToolCalls);
    CHECK(wire::enum_cast(v1::FINISH_REASON_UNSPECIFIED, client::FinishReason::Eog) ==
          client::FinishReason::Eog);
}

}  // namespace

int main() {
    test_chat_message_round_trip();
    test_sampling_params_round_trip();
    test_finish_reason_names();

    std::fprintf(stderr, "%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
