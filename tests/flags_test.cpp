/** @file
 * @brief Reflected flag values, errors, positional arguments and help formatting.
 *
 * Tests for src/cli/flags.h: what each member type takes from the command line, what `--help`
 * prints for it, and what is rejected. No daemon, no model, no gRPC.
 */

#include "flags.h"

#include <cstdio>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace {

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
        const auto & lhs_ = (a);                                                                 \
        const auto & rhs_ = (b);                                                                 \
        if (!(lhs_ == rhs_)) {                                                                   \
            ++failures;                                                                          \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s == %s\n", __FILE__, __LINE__, #a, #b); \
            std::fprintf(stderr, "    left:  %s\n", to_display(lhs_).c_str());                   \
            std::fprintf(stderr, "    right: %s\n", to_display(rhs_).c_str());                   \
        }                                                                                        \
    } while (0)

std::string to_display(const std::string & s) { return "'" + s + "'"; }
std::string to_display(size_t n) { return std::to_string(n); }

using llamad::cli::help;

struct Options {
    [[=help{"PATH", "GGUF model to load"}]]
    std::string model;

    [[=help{"context size"}]]
    int32_t ctx = 4096;

    [[=help{"sampling temperature"}]]
    std::optional<float> temp;

    [[=help{"sampling seed"}]]
    std::optional<uint32_t> seed;

    [[=help{"TEXT", "system prompt"}]]
    std::optional<std::string> system;

    [[=help{"STR", "stop string (repeatable)"}]]
    std::vector<std::string> stop;

    [[=help{"list the devices and exit"}]]
    bool list_devices = false;

    [[=help{"run a single non-interactive turn\n"
            "and exit"}]]
    bool once = false;

    // No help annotation: a hidden flag, parsed but left out of --help.
    std::optional<long> cancel_after;
};

struct Extra {
    [[=help{"N", "rounds"}]]
    int rounds = 1;
};

// argv as main() gets it: the program name, then the arguments.
struct CommandLine {
    explicit CommandLine(std::vector<std::string> arguments) : storage(std::move(arguments)) {
        pointers.push_back(const_cast<char *>("test"));
        for (std::string & argument : storage) {
            pointers.push_back(argument.data());
        }
    }

    int     argc() const { return static_cast<int>(pointers.size()); }
    char ** argv() { return pointers.data(); }

    std::vector<std::string> storage;
    std::vector<char *>      pointers;
};

template <typename... Structs>
std::vector<std::string> parse(std::vector<std::string> arguments, Structs &... options) {
    CommandLine line(std::move(arguments));
    return llamad::cli::parse_flags(line.argc(), line.argv(), options...);
}

// The message of the FlagError `arguments` provokes, or "" when they parse.
template <typename... Structs>
std::string error_of(std::vector<std::string> arguments, Structs &... options) {
    try {
        parse(std::move(arguments), options...);
    } catch (const llamad::cli::FlagError & e) {
        return e.what();
    }
    return "";
}

std::string capture_help() {
    std::FILE * out = std::tmpfile();
    if (out == nullptr) {
        std::fprintf(stderr, "cannot open a temporary file\n");
        std::exit(1);
    }
    llamad::cli::print_flags(out, Options{}, llamad::cli::HelpFlag{});

    std::string text;
    std::rewind(out);
    for (int c = std::fgetc(out); c != EOF; c = std::fgetc(out)) {
        text += static_cast<char>(c);
    }
    std::fclose(out);
    return text;
}

bool contains(const std::string & haystack, const std::string & needle) {
    return haystack.find(needle) != std::string::npos;
}

void test_values() {
    Options options;
    const std::vector<std::string> positional =
        parse({"--model", "m.gguf", "--ctx", "2048", "--temp", "0.5", "--seed", "7", "--system",
               "be brief", "--stop", "</s>", "--stop", "a,b", "--list-devices", "--cancel-after", "3"},
              options);

    CHECK(positional.empty());
    CHECK_EQ(options.model, std::string("m.gguf"));
    CHECK(options.ctx == 2048);
    CHECK(options.temp && *options.temp == 0.5f);
    CHECK(options.seed && *options.seed == 7u);
    CHECK(options.system && *options.system == "be brief");
    CHECK(options.list_devices);
    CHECK(options.cancel_after && *options.cancel_after == 3);

    // A repeated flag appends, and a comma is part of the value: splitting one is the owner's job.
    CHECK_EQ(options.stop.size(), size_t(2));
    CHECK_EQ(options.stop[0], std::string("</s>"));
    CHECK_EQ(options.stop[1], std::string("a,b"));
}

void test_defaults() {
    Options options;
    parse({}, options);

    CHECK(options.model.empty());
    CHECK(options.ctx == 4096);
    CHECK(!options.temp);
    CHECK(!options.seed);
    CHECK(!options.system);
    CHECK(options.stop.empty());
    CHECK(!options.list_devices);
    CHECK(!options.cancel_after);
}

void test_positionals() {
    Options options;
    const std::vector<std::string> positional =
        parse({"first", "--ctx", "8", "second", "--list-devices", "-dash", "third"}, options);

    CHECK(options.ctx == 8);
    CHECK(options.list_devices);
    CHECK_EQ(positional.size(), size_t(4));
    CHECK_EQ(positional[0], std::string("first"));
    CHECK_EQ(positional[1], std::string("second"));
    CHECK_EQ(positional[2], std::string("-dash"));
    CHECK_EQ(positional[3], std::string("third"));
}

void test_several_structs() {
    Options options;
    Extra   extra;
    llamad::cli::HelpFlag help_flag;
    parse({"--rounds", "4", "--ctx", "512", "-h"}, options, extra, help_flag);

    CHECK(extra.rounds == 4);
    CHECK(options.ctx == 512);
    CHECK(help_flag.help);

    llamad::cli::HelpFlag spelled_out;
    parse({"--help"}, spelled_out);
    CHECK(spelled_out.help);
}

void test_errors() {
    Options options;
    CHECK_EQ(error_of({"--nope"}, options), std::string("unknown argument '--nope'"));
    CHECK_EQ(error_of({"--ctx"}, options), std::string("--ctx needs a value"));
    CHECK_EQ(error_of({"--ctx", "many"}, options), std::string("--ctx needs an integer"));
    CHECK_EQ(error_of({"--ctx", "12x"}, options), std::string("--ctx needs an integer"));
    CHECK_EQ(error_of({"--ctx", "9999999999"}, options), std::string("--ctx is out of range"));
    CHECK_EQ(error_of({"--temp", "warm"}, options), std::string("--temp needs a number"));
    CHECK_EQ(error_of({"--seed", "-1"}, options), std::string("--seed needs a non-negative integer"));

    // -h is the only short option; anything else starting with one dash is an argument.
    llamad::cli::HelpFlag help_flag;
    CHECK_EQ(error_of({"-h"}, help_flag), std::string(""));
}

void test_help_text() {
    const std::string text = capture_help();

    // The placeholder comes from the annotation, or from the member's type where it gives none,
    // and the columns line up across every struct printed.
    CHECK(contains(text, "  --model PATH   GGUF model to load\n"));
    CHECK(contains(text, "  --ctx N        context size\n"));
    CHECK(contains(text, "  --temp F       sampling temperature\n"));
    CHECK(contains(text, "  --system TEXT  system prompt\n"));
    CHECK(contains(text, "  --stop STR     stop string (repeatable)\n"));
    CHECK(contains(text, "  --list-devices list the devices and exit\n"));
    CHECK(contains(text, "  --help         show this message\n"));

    // A newline in the text starts a continuation line, indented to the text column.
    CHECK(contains(text, "  --once         run a single non-interactive turn\n"
                         "                 and exit\n"));

    // A member with no help annotation is a hidden flag.
    CHECK(!contains(text, "cancel-after"));
}

}  // namespace

int main() {
    test_values();
    test_defaults();
    test_positionals();
    test_several_structs();
    test_errors();
    test_help_text();

    std::fprintf(stderr, "%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
