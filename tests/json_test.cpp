/** @file
 * @brief Checked JSON conversion, enum names, optional presence and generated schemas.
 *
 * The client's reflection adapter: field mapping, conversion policies and tool schemas.
 * JSON syntax and escaping belong to nlohmann's tests. No daemon or model is needed here.
 */

#include "check.h"
#include "llamad/json.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace json = llamad::client::json;

using llamad::client::desc;

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

// Reads `text` and reports whether the reader rejected it. json::Error is the only thing read()
// throws, so anything else escapes and fails the test.
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
    R"({"place":"Oslo","unit":"Fahrenheit","value":-125.0,)"
    R"("stale":true,"samples":7,"tags":["a","b"],"at":{"x":1,"y":-2.25},"extra":{"skip":[1,2,{}]}})";

void test_reflected_fields() {
    Reading reading;
    json::read(kReading, reading);

    CHECK_EQ(reading.place, "Oslo");
    CHECK(reading.unit == Unit::Fahrenheit);
    CHECK(reading.value == -125.0);
    CHECK(reading.stale);
    CHECK(reading.samples.has_value() && *reading.samples == 7);
    CHECK(reading.tags.size() == 2 && reading.tags[1] == "b");
    CHECK(reading.at.x == 1.0 && reading.at.y == -2.25);
}

void test_optional() {
    Reading reading;
    json::read(R"({"place":"x","unit":"Celsius","value":0,"stale":false,"tags":[],"at":{"x":0,"y":0}})",
               reading);
    CHECK(!reading.samples.has_value());

    reading.samples = 3;
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

void test_checked_numbers() {
    // nlohmann permits narrowing in get<T>(); the tool adapter must reject it before invocation.
    CHECK(rejects<int32_t>("2147483648"));
    CHECK(rejects<int32_t>("-2147483649"));
    CHECK(rejects<int32_t>("4294967296"));
    CHECK(rejects<uint32_t>("-1"));
    CHECK(rejects<uint32_t>("4294967296"));
    CHECK(rejects<int64_t>("9223372036854775808"));
    CHECK(rejects<int64_t>("-9223372036854775809"));
    CHECK(rejects<uint64_t>("18446744073709551616"));
    CHECK(rejects<int>("1.0"));
    CHECK(rejects<int>("true"));
    CHECK(rejects<double>("true"));
    CHECK(rejects<float>("1e100"));
    CHECK(rejects<float>("-1e100"));
    CHECK(rejects<char>("256"));
    CHECK(rejects<int8_t>("128"));
    char character = 0;
    json::read("65", character);
    CHECK(character == 'A');

    uint64_t unsigned_value = 0;
    json::read("18446744073709551615", unsigned_value);
    CHECK(unsigned_value == std::numeric_limits<uint64_t>::max());
    CHECK_EQ(json::write(unsigned_value), "18446744073709551615");
    int64_t signed_value = 0;
    json::read("-9223372036854775808", signed_value);
    CHECK(signed_value == std::numeric_limits<int64_t>::min());
    json::read("9223372036854775807", signed_value);
    CHECK(signed_value == std::numeric_limits<int64_t>::max());
    float finite = 0;
    json::read(json::write(std::numeric_limits<float>::max()), finite);
    CHECK(finite == std::numeric_limits<float>::max());
}

void test_object_contract() {
    CHECK(rejects<Alarm>(R"({"unit":"Celsius"})"));                         // missing required key
    CHECK(rejects<Alarm>(R"({"reason":"heat","unit":"Kelvin"})"));          // no such enumerator
    CHECK(rejects<Alarm>(R"({"reason":7,"unit":"Celsius"})"));              // not a string
    CHECK(rejects<Alarm>(R"({"reason":"heat","unit":0})"));                 // an enum is its name
    CHECK(rejects<bool>("1"));
    CHECK(rejects<Point>("[]"));
    CHECK(rejects<std::vector<int>>("{}"));
    CHECK(rejects<Point>(R"({"x":0,"y":)"));                                // not JSON at all
    CHECK(rejects<Point>(""));

    Alarm alarm;
    try {
        json::read(R"({"unit":"Celsius"})", alarm);
        CHECK(false);
    } catch (const json::Error & e) {
        CHECK_EQ(e.what(), "json: missing key 'reason'");
    }
}

void test_reflected_result() {
    Reading reading;
    reading.place   = "Oslo";
    reading.unit    = Unit::Fahrenheit;
    reading.value   = 1.5;
    reading.stale   = true;
    reading.tags    = {"a", "b"};
    reading.at      = {1, -2.25};

    // Field names, enum names and omission of unset members are the adapter's contract.
    CHECK(nlohmann::json::parse(json::write(reading)) == nlohmann::json::parse(
        R"({"place":"Oslo","unit":"Fahrenheit","value":1.5,"stale":true,)"
        R"("tags":["a","b"],"at":{"x":1,"y":-2.25}})"));

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

    // A result is whatever the tool read from somewhere, so a byte that is not UTF-8 is replaced
    // and the rest still reaches the model.
    CHECK_EQ(json::write(std::string("caf\xe9 au lait")), "\"caf\xef\xbf\xbd au lait\"");
}

enum class Alias { First = 0, AlsoFirst = 0 };

void test_vectors_and_enums() {
    std::vector<bool> flags;
    json::read("[true,false,true]", flags);
    CHECK(flags == std::vector<bool>({true, false, true}));
    CHECK_EQ(json::write(flags), "[true,false,true]");

    std::vector<std::optional<Alias>> values;
    json::read(R"(["AlsoFirst",null,"First"])", values);
    CHECK(values.size() == 3);
    CHECK(values[0] == Alias::First && !values[1] && values[2] == Alias::First);
    CHECK_EQ(json::write(values), R"(["First",null,"First"])");

    try {
        json::write(static_cast<Alias>(42));
        CHECK(false);
    } catch (const json::Error &) {
    }
    for (const double value : {std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
        try {
            json::write(value);
            CHECK(false);
        } catch (const json::Error &) {
        }
    }
    if constexpr (std::numeric_limits<long double>::max() > std::numeric_limits<double>::max()) {
        try {
            json::write(std::numeric_limits<long double>::max());
            CHECK(false);
        } catch (const json::Error &) {
        }
    }
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

    struct Batch {
        std::vector<Point> points;
        std::optional<std::vector<bool>> flags;
    };
    const auto schema = nlohmann::json::parse(json::schema<Batch>());
    CHECK(schema["properties"]["points"]["items"] == nlohmann::json::parse(json::schema<Point>()));
    CHECK(schema["properties"]["flags"]["items"]["type"] == "boolean");
    CHECK(schema["required"] == nlohmann::json::array({"points"}));
}

}  // namespace

int main() {
    test_reflected_fields();
    test_optional();
    test_checked_numbers();
    test_object_contract();
    test_reflected_result();
    test_vectors_and_enums();
    test_schema();

    return tests::report();
}
