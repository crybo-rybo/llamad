/** @file
 * @brief The flag reader's values and errors, and the shared engine flags.
 *
 * Tests for src/flags.h and src/engine_flags.h. No daemon, no model, no gRPC.
 */

#include "check.h"
#include "engine_flags.h"
#include "flags.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace cli = llamad::cli;

// argv as main() gets it: the program name, then the arguments.
struct CommandLine {
    explicit CommandLine(std::vector<std::string> arguments) : storage(std::move(arguments)) {
        pointers.push_back(program.data());
        for (std::string & argument : storage) {
            pointers.push_back(argument.data());
        }
    }

    int     argc() const { return static_cast<int>(pointers.size()); }
    char ** argv() { return pointers.data(); }

    std::string              program = "test";
    std::vector<std::string> storage;
    std::vector<char *>      pointers;
};

// What a binary's own flags look like, read the way every binary reads them.
struct Options {
    std::string              model;
    std::optional<float>     temp;
    std::optional<uint32_t>  seed;
    std::vector<std::string> stop;
    bool                     help = false;
    std::vector<std::string> positional;
};

Options parse(std::vector<std::string> arguments, llamad::EngineFlags & engine_flags) {
    CommandLine line(std::move(arguments));
    Options     options;
    for (cli::Args args(line.argc(), line.argv()); args.next();) {
        if (!args.is_flag()) {
            options.positional.push_back(args.current());
        } else if (args.is("--model")) {
            options.model = args.value();
        } else if (args.is("--temp")) {
            options.temp = args.number<float>();
        } else if (args.is("--seed")) {
            options.seed = args.number<uint32_t>();
        } else if (args.is("--stop")) {
            options.stop.push_back(args.value());
        } else if (args.is("--help")) {
            options.help = true;
        } else if (!llamad::parse_engine_flag(args, engine_flags)) {
            throw args.unknown();
        }
    }
    return options;
}

Options parse(std::vector<std::string> arguments) {
    llamad::EngineFlags engine_flags;
    return parse(std::move(arguments), engine_flags);
}

// The message of the FlagError `arguments` provoke, or "" when they parse.
std::string error_of(std::vector<std::string> arguments) {
    try {
        parse(std::move(arguments));
    } catch (const cli::FlagError & e) {
        return e.what();
    }
    return "";
}

// The message of the FlagError to_config raises for these engine flags, or "".
std::string config_error_of(std::vector<std::string> arguments) {
    try {
        llamad::EngineFlags engine_flags;
        parse(std::move(arguments), engine_flags);
        llamad::to_config(engine_flags);
    } catch (const cli::FlagError & e) {
        return e.what();
    }
    return "";
}

void test_values() {
    llamad::EngineFlags engine_flags;
    const Options       options =
        parse({"--model", "m.gguf", "--temp", "0.5", "--seed", "7", "--stop", "</s>", "--stop", "a,b", "--ctx", "2048",
               "--threads", "3", "--list-devices"},
              engine_flags);

    CHECK(options.positional.empty());
    CHECK_EQ(options.model, "m.gguf");
    CHECK(options.temp && *options.temp == 0.5f);
    CHECK(options.seed && *options.seed == 7u);
    CHECK(engine_flags.ctx == 2048);
    CHECK(engine_flags.threads == 3);
    CHECK(engine_flags.list_devices);

    // A repeated flag appends, and a comma is part of the value: splitting one is the owner's job.
    CHECK_EQ(options.stop.size(), size_t(2));
    CHECK_EQ(options.stop[0], "</s>");
    CHECK_EQ(options.stop[1], "a,b");
}

void test_defaults() {
    llamad::EngineFlags engine_flags;
    const Options       options = parse({}, engine_flags);

    CHECK(options.model.empty());
    CHECK(!options.temp);
    CHECK(!options.seed);
    CHECK(options.stop.empty());
    CHECK(!options.help);
    CHECK(engine_flags.ctx == 4096);
    CHECK(engine_flags.ngl == 99);
    CHECK(!engine_flags.devices);
    CHECK(!engine_flags.list_devices);
}

void test_positionals() {
    const Options options = parse({"first", "--temp", "1", "second", "-dash", "third"});

    // Only a long option is a flag, so a prompt or a path that starts with one dash is not.
    CHECK_EQ(options.positional.size(), size_t(4));
    CHECK_EQ(options.positional[0], "first");
    CHECK_EQ(options.positional[1], "second");
    CHECK_EQ(options.positional[2], "-dash");
    CHECK_EQ(options.positional[3], "third");
}

void test_help() {
    CHECK(parse({"-h"}).help);
    CHECK(parse({"--help"}).help);
}

void test_errors() {
    CHECK_EQ(error_of({"--nope"}), "unknown argument '--nope'");
    CHECK_EQ(error_of({"--ctx"}), "--ctx needs a value");
    CHECK_EQ(error_of({"--ctx", "many"}), "--ctx needs an integer");
    CHECK_EQ(error_of({"--ctx", "12x"}), "--ctx needs an integer");
    CHECK_EQ(error_of({"--ctx", "9999999999"}), "--ctx is out of range");
    CHECK_EQ(error_of({"--temp", "warm"}), "--temp needs a number");
    CHECK_EQ(error_of({"--temp", ""}), "--temp needs a number");
    CHECK_EQ(error_of({"--temp", " 1"}), "--temp needs a number");
    CHECK_EQ(error_of({"--temp", "nan"}), "--temp needs a number");
    CHECK_EQ(error_of({"--temp", "1e99"}), "--temp is out of range");
    CHECK_EQ(error_of({"--seed", "-1"}), "--seed needs a non-negative integer");
}

void test_engine_config() {
    llamad::EngineFlags engine_flags;
    parse({"--ctx", "512", "--ngl", "0", "--devices", "Vulkan0,Vulkan1", "--tensor-split", "3,1.5"}, engine_flags);
    const llamad::EngineConfig config = llamad::to_config(engine_flags);

    CHECK(config.n_ctx == 512);
    CHECK(config.n_gpu_layers == 0);
    CHECK_EQ(config.devices.size(), size_t(2));
    CHECK_EQ(config.devices[1], "Vulkan1");
    CHECK_EQ(config.tensor_split.size(), size_t(2));
    CHECK(config.tensor_split[1] == 1.5f);

    const std::string split_error = "--tensor-split needs a comma-separated list of non-negative numbers, e.g. 3,1";
    CHECK_EQ(config_error_of({"--ctx", "0"}), "--ctx needs a positive integer");
    CHECK_EQ(config_error_of({"--threads", "-1"}), "--threads needs a non-negative integer");
    CHECK_EQ(config_error_of({"--devices", "a,,b"}),
             "--devices needs a comma-separated list of device names (see --list-devices)");
    CHECK_EQ(config_error_of({"--tensor-split", "3,-1"}), split_error);
    CHECK_EQ(config_error_of({"--tensor-split", "3,x"}), split_error);
    CHECK_EQ(config_error_of({"--tensor-split", ""}), split_error);
}

}  // namespace

int main() {
    test_values();
    test_defaults();
    test_positionals();
    test_help();
    test_errors();
    test_engine_config();

    return tests::report();
}
