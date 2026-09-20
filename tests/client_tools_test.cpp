// ToolSet tests: a C++ function becomes a tool definition, and a model's ToolCall runs it.
// No daemon — this is everything client.h does on its own side of the RPC.

#include "llamad/client.h"

#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

namespace client = llamad::client;

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

enum class Scale { Metric, Imperial };

struct Forecast {
    std::string                city;
    double                     high = 0;
    std::optional<std::string> warning;
};

[[=desc{"Look up the weather in a city."}]]
std::string get_weather([[=desc{"City name, e.g. Oslo"}]] std::string city,
                        [[=desc{"Units to answer in."}]] Scale scale,
                        [[=desc{"How many days ahead."}]] std::optional<int> days) {
    return city + "/" + (scale == Scale::Metric ? "metric" : "imperial") + "/" +
           (days ? std::to_string(*days) : "today");
}

[[=desc{"Look up tomorrow's forecast."}]]
Forecast get_forecast(std::string city) {
    Forecast forecast;
    forecast.city = city;
    forecast.high = 21.5;
    if (city == "Oslo") {
        forecast.warning = "bring a coat";
    }
    return forecast;
}

std::string always_fails(std::string reason) {
    throw std::runtime_error("cannot do that: " + reason);
}

void test_definition() {
    client::ToolSet tools;
    tools.add<^^get_weather>();

    CHECK(tools.definitions().size() == 1);
    const client::Tool & tool = tools.definitions().front();
    CHECK_EQ(tool.name, "get_weather");
    CHECK_EQ(tool.description, "Look up the weather in a city.");
    CHECK_EQ(tool.parameters_json_schema,
             R"({"type":"object","properties":{)"
             R"("city":{"type":"string","description":"City name, e.g. Oslo"},)"
             R"("scale":{"type":"string","enum":["Metric","Imperial"],"description":"Units to answer in."},)"
             R"("days":{"type":"integer","description":"How many days ahead."}},)"
             R"("required":["city","scale"]})");
}

void test_call() {
    client::ToolSet tools;
    tools.add<^^get_weather>();
    tools.add<^^get_forecast>();
    tools.add<^^always_fails>();

    CHECK_EQ(tools.call({"call_0", "get_weather", R"({"city":"Oslo","scale":"Metric","days":2})"}),
             "Oslo/metric/2");

    // An absent optional is not an error; a key the tool does not know is ignored.
    CHECK_EQ(tools.call({"call_1", "get_weather", R"({"city":"Oslo","scale":"Imperial","mood":"grey"})"}),
             "Oslo/imperial/today");

    // A tool returning anything but a string answers in JSON.
    CHECK_EQ(tools.call({"call_2", "get_forecast", R"({"city":"Oslo"})"}),
             R"({"city":"Oslo","high":21.5,"warning":"bring a coat"})");
    CHECK_EQ(tools.call({"call_3", "get_forecast", R"({"city":"Rome"})"}),
             R"({"city":"Rome","high":21.5})");
}

// Everything the model can get wrong comes back as a JSON object the model can read.
void test_call_reports_failures() {
    client::ToolSet tools;
    tools.add<^^get_weather>();
    tools.add<^^always_fails>();

    const std::string unknown = tools.call({"call_0", "get_time", "{}"});
    CHECK(unknown.rfind(R"({"error":"no such tool: get_time")", 0) == 0);

    const std::string malformed = tools.call({"call_1", "get_weather", R"({"city":)"});
    CHECK(malformed.rfind(R"({"error":"json: )", 0) == 0);

    const std::string missing = tools.call({"call_2", "get_weather", R"({"scale":"Metric"})"});
    CHECK(missing.find("missing key 'city'") != std::string::npos);

    CHECK_EQ(tools.call({"call_3", "always_fails", R"({"reason":"\"reasons\""})"}),
             R"({"error":"cannot do that: \"reasons\""})");
}

}  // namespace

int main() {
    test_definition();
    test_call();
    test_call_reports_failures();

    std::fprintf(stderr, "%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
