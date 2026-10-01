/** @file
 * @brief A small command-line reader shared by the daemon and the engine CLIs.
 *
 * Each binary walks its arguments with Args and an if/else chain over the flags it takes, and
 * prints its own --help text. Only `--long-option` and `-h` are flags, so a prompt or a path that
 * starts with a dash stays an argument; a flag that takes a value reads it from the next argument.
 */

#pragma once

#include <cctype>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace llamad::cli {

/// An unknown flag, a flag missing its value, or a value the flag cannot take. The message names
/// the flag, so a binary prints it with its usage.
struct FlagError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// Reads `text` whole as a number of type T: no sign on an unsigned type, nothing left over, in
/// range. `flag` names the flag in the error.
template <typename T>
T parse_number(std::string_view flag, const std::string & text) {
    static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>);

    const auto fail = [&](const char * what) { return FlagError(std::string(flag) + what); };

    if constexpr (std::is_floating_point_v<T>) {
        // strtod rather than from_chars, whose floating-point overloads not every standard
        // library has. strtod skips leading space and reads "inf" and "nan"; none of those are a
        // value a flag means.
        if (text.empty() || std::isspace(static_cast<unsigned char>(text[0]))) {
            throw fail(" needs a number");
        }
        char * end = nullptr;
        errno      = 0;
        const double value = std::strtod(text.c_str(), &end);
        if (end != text.c_str() + text.size() || !std::isfinite(value)) {
            throw fail(" needs a number");
        }
        if (errno == ERANGE || std::abs(value) > static_cast<double>(std::numeric_limits<T>::max())) {
            throw fail(" is out of range");
        }
        return static_cast<T>(value);
    } else {
        T value{};
        const char * end    = text.data() + text.size();
        const auto   parsed = std::from_chars(text.data(), end, value);
        if (parsed.ec == std::errc::result_out_of_range) {
            throw fail(" is out of range");
        }
        if (parsed.ec != std::errc{} || parsed.ptr != end) {
            throw fail(std::is_unsigned_v<T> ? " needs a non-negative integer" : " needs an integer");
        }
        return value;
    }
}

/// Walks argv one argument at a time.
///
///     for (cli::Args args(argc, argv); args.next();) {
///         if (!args.is_flag())          positional.push_back(args.current());
///         else if (args.is("--model"))  model = args.value();
///         else if (args.is("--ctx"))    ctx   = args.number<int32_t>();
///         else                          throw args.unknown();
///     }
class Args {
public:
    /// `argv[0]` is the program and is skipped.
    Args(int argc, char ** argv) : argc_(argc), argv_(argv) {}

    /// Moves to the next argument; false when there is none.
    bool next() { return ++at_ < argc_; }

    /// The argument next() moved to.
    std::string current() const { return argv_[at_]; }

    /// Whether the current argument is a flag: a long option, or -h.
    bool is_flag() const {
        const std::string_view arg = argv_[at_];
        return arg.starts_with("--") || arg == "-h";
    }

    /// Whether the current argument is the flag `name`; -h is --help.
    bool is(std::string_view name) const {
        const std::string_view arg = argv_[at_];
        return arg == name || (arg == "-h" && name == "--help");
    }

    /// The current flag's value: the argument after it, which this consumes.
    std::string value() {
        if (at_ + 1 >= argc_) {
            throw FlagError(current() + " needs a value");
        }
        return argv_[++at_];
    }

    /// The current flag's value read as a number; see parse_number.
    template <typename T>
    T number() {
        const std::string flag = current();
        return parse_number<T>(flag, value());
    }

    /// The error for a flag no branch recognised.
    FlagError unknown() const { return FlagError("unknown argument '" + current() + "'"); }

private:
    int     argc_;   ///< Argument count, the program included.
    char ** argv_;   ///< The arguments; argv_[0] is the program.
    int     at_ = 0; ///< Index of the current argument; 0 before the first next().
};

}  // namespace llamad::cli
