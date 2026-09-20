// Reader, writer and schema tests for client/src/json.cpp and the reflective glue in json.h.
// No daemon and no model: this is the JSON path a tool call travels, on its own.

#include "llamad/json.h"

#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace json = llamad::client::json;

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

enum class Unit { Celsius, Fahrenheit };

struct Point {
    double x = 0;
    double y = 0;
};

struct Reading {
    std::string              place;
    Unit                     unit = Unit::Celsius;
    double                   value = 0;
    bool                     stale = false;
    std::optional<int>       samples;
    std::vector<std::string> tags;
    Point                    at;
};

struct Alarm {
    [[=desc{"What to sound the alarm about."}]] std::string reason;

    Unit unit = Unit::Celsius;

    [[=desc{"How many times to repeat it."}]] std::optional<int> repeat;
};

// Reads `text` and reports whether the reader rejected it.
template <typename T>
bool rejects(const std::string & text) {
    T value{};
    try {
        json::read(text, value);
    } catch (const json::Error &) {
        return true;
    }
    return false;
}

const char * const kReading =
    R"({"place":"a \"quoted\"\tname","unit":"Fahrenheit","value":-12.5e1,)"
    R"("stale":true,"samples":7,"tags":["a","b"],"at":{"x":1,"y":-2.25},"extra":{"skip":[1,2,{}]}})";

void test_read_reading() {
    Reading reading;
    json::read(kReading, reading);

    CHECK_EQ(reading.place, "a \"quoted\"\tname");
    CHECK(reading.unit == Unit::Fahrenheit);
    CHECK(reading.value == -125.0);
    CHECK(reading.stale);
    CHECK(reading.samples.has_value() && *reading.samples == 7);
    CHECK(reading.tags.size() == 2 && reading.tags[1] == "b");
    CHECK(reading.at.x == 1.0 && reading.at.y == -2.25);
}

// Not raw strings: GCC resolves \uXXXX inside one of those too, so the escape would never reach
// the reader.
void test_unicode_escapes() {
    std::string text;
    json::read("\"caf\\u00e9 \\u00b5m\"", text);
    CHECK_EQ(text, "caf\xc3\xa9 \xc2\xb5m");

    json::read("\"a\\ud83d\\ude00b\"", text);  // a surrogate pair is one code point
    CHECK_EQ(text, "a\xf0\x9f\x98\x80" "b");

    CHECK(rejects<std::string>("\"\\ud83d\""));   // a high surrogate with no low one
    CHECK(rejects<std::string>("\"\\udc00\""));   // a low surrogate on its own
    CHECK(rejects<std::string>("\"\\u00g0\""));
    CHECK(rejects<std::string>("\"\\q\""));
}

void test_optional() {
    Reading reading;
    json::read(R"({"place":"x","unit":"Celsius","value":0,"stale":false,"tags":[],"at":{"x":0,"y":0}})",
               reading);
    CHECK(!reading.samples.has_value());

    // An explicit null reads the same as the key not being there at all.
    reading.samples = 3;
    json::read(R"({"place":"x","unit":"Celsius","value":0,"stale":false,"samples":null,"tags":[],)"
               R"("at":{"x":0,"y":0}})",
               reading);
    CHECK(!reading.samples.has_value());
}

void test_numbers() {
    Point point;
    json::read(R"({"x":-0.5,"y":2e-3})", point);
    CHECK(point.x == -0.5);
    CHECK(point.y == 0.002);

    Alarm alarm;
    json::read(R"({"reason":"heat","unit":"Celsius","repeat":-4})", alarm);
    CHECK(alarm.repeat.has_value() && *alarm.repeat == -4);

    // An integer member will not quietly take a fraction.
    CHECK(rejects<Alarm>(R"({"reason":"heat","unit":"Celsius","repeat":1.5})"));
}

void test_rejects() {
    CHECK(rejects<Alarm>(R"({"unit":"Celsius"})"));                         // missing required key
    CHECK(rejects<Alarm>(R"({"reason":"heat","unit":"Kelvin"})"));          // no such enumerator
    CHECK(rejects<Alarm>(R"({"reason":"heat","unit":"Celsius")"));          // truncated
    CHECK(rejects<Alarm>(R"({"reason":"heat)"));                            // unterminated string
    CHECK(rejects<Alarm>(R"({"reason":"heat","unit":"Celsius"} tail)"));    // trailing text
    CHECK(rejects<Point>(""));
    CHECK(rejects<Point>("["));
    CHECK(rejects<Point>(R"({"x":,"y":0})"));
    CHECK(rejects<Point>(std::string(200, '[')));                           // nested too deeply

    Alarm alarm;
    try {
        json::read(R"({"unit":"Celsius"})", alarm);
        CHECK(false);
    } catch (const json::Error & e) {
        CHECK(e.offset == 18);
        CHECK(std::string(e.what()).find("missing key 'reason'") != std::string::npos);
    }
}

void test_write() {
    Reading reading;
    reading.place   = "line\nbreak \x01 \"quoted\"";
    reading.unit    = Unit::Fahrenheit;
    reading.value   = 1.5;
    reading.stale   = true;
    reading.tags    = {"a", "b"};
    reading.at      = {1, -2.25};

    // An unset optional is left out, and a control character is escaped.
    CHECK_EQ(json::write(reading),
             R"({"place":"line\nbreak \u0001 \"quoted\"","unit":"Fahrenheit","value":1.5,"stale":true,)"
             R"("tags":["a","b"],"at":{"x":1,"y":-2.25}})");

    reading.samples = 42;
    Reading back;
    json::read(json::write(reading), back);
    CHECK_EQ(back.place, reading.place);
    CHECK(back.unit == reading.unit);
    CHECK(back.value == reading.value);
    CHECK(back.stale == reading.stale);
    CHECK(back.samples.has_value() && *back.samples == 42);
    CHECK(back.tags == reading.tags);
    CHECK(back.at.y == reading.at.y);
}

void test_schema() {
    CHECK_EQ(json::schema<Alarm>(),
             R"({"type":"object","properties":{)"
             R"("reason":{"type":"string","description":"What to sound the alarm about."},)"
             R"("unit":{"type":"string","enum":["Celsius","Fahrenheit"]},)"
             R"("repeat":{"type":"integer","description":"How many times to repeat it."}},)"
             R"("required":["reason","unit"]})");

    CHECK_EQ(json::schema<Point>(),
             R"({"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"}},)"
             R"("required":["x","y"]})");
}

}  // namespace

int main() {
    test_read_reading();
    test_unicode_escapes();
    test_optional();
    test_numbers();
    test_rejects();
    test_write();
    test_schema();

    std::fprintf(stderr, "%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
