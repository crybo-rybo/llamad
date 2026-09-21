#pragma once

// The command line of a binary in this repository, as a struct whose members are its flags:
// `max_tokens` is `--max-tokens`, the member's type says what the flag takes, and its help
// annotation is the line `--help` prints for it. A member with no help annotation is a hidden
// flag: parsed, but not printed.
//
// A bool flag takes no value and is set by its presence. std::string, integers and floating
// point take one, parsed strictly: the whole value, in range, or an error. std::optional<T>
// stays unset until the flag is given, which is how a member leaves a default that lives
// elsewhere alone, and std::vector<std::string> collects one element per occurrence. A
// comma-separated list is a std::string its owner splits afterwards, so a value is never split
// behind its back.
//
// Arguments that are not flags come back in order, for the binary to interpret.

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdio>
#include <meta>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

namespace llamad {
namespace cli {

// An unknown flag, a flag missing its value, or a value the member's type cannot take. The
// message names the flag, so a binary prints it with its usage.
struct FlagError : std::runtime_error {
    explicit FlagError(const std::string & message) : std::runtime_error(message) {}
};

// What `--help` says about a flag: the placeholder its value is shown as (PATH, N, ...) and what
// the flag does, or the text alone where the member's type supplies the placeholder. A newline
// in the text starts a continuation line, indented to the text column. The text is a char array
// because an annotation's value has to be of a structural type.
template <std::size_t N, std::size_t M>
struct help {
    char placeholder[N];
    char text[M];

    consteval help(const char (&text_)[M]) : placeholder{}, text{} {
        for (std::size_t i = 0; i < M; ++i) {
            text[i] = text_[i];
        }
    }

    consteval help(const char (&placeholder_)[N], const char (&text_)[M]) : placeholder{}, text{} {
        for (std::size_t i = 0; i < N; ++i) {
            placeholder[i] = placeholder_[i];
        }
        for (std::size_t i = 0; i < M; ++i) {
            text[i] = text_[i];
        }
    }
};

template <std::size_t M>
help(const char (&)[M]) -> help<1, M>;
template <std::size_t N, std::size_t M>
help(const char (&)[N], const char (&)[M]) -> help<N, M>;

// --help, worded the same way by every binary and also accepted as -h. It is a struct of its own
// so that it is the last flag printed, whatever else a binary takes.
struct HelpFlag {
    [[=help{"show this message"}]]
    bool help = false;
};

namespace detail {

template <typename T> constexpr bool is_optional                   = false;
template <typename T> constexpr bool is_optional<std::optional<T>> = true;
template <typename T> constexpr bool is_vector                     = false;
template <typename T> constexpr bool is_vector<std::vector<T>>     = true;

// What a flag's value is read into: what an optional holds, or what a vector collects.
template <typename T> struct value_type                   { using type = T; };
template <typename T> struct value_type<std::optional<T>> { using type = T; };
template <typename T> struct value_type<std::vector<T>>   { using type = T; };

// define_static_array outlives the constant evaluation that builds the list, so `template for`
// can iterate it.
consteval auto fields_of(std::meta::info type) {
    return std::define_static_array(
        std::meta::nonstatic_data_members_of(type, std::meta::access_context::unprivileged()));
}

consteval bool is_help(std::meta::info annotation) {
    const std::meta::info type = std::meta::type_of(annotation);
    return std::meta::has_template_arguments(type) && std::meta::template_of(type) == ^^cli::help;
}

// The member's help annotation, or the null reflection: a hidden flag.
consteval std::meta::info help_of(std::meta::info member) {
    for (std::meta::info annotation : std::meta::annotations_of(member)) {
        if (is_help(annotation)) {
            return annotation;
        }
    }
    return std::meta::info{};
}

// The flag a member spells: `max_tokens` is `--max-tokens`.
template <std::meta::info Member>
consteval const char * flag_name() {
    std::string out = "--";
    for (char c : std::meta::identifier_of(Member)) {
        out += c == '_' ? '-' : c;
    }
    return std::define_static_string(out);
}

template <std::meta::info Annotation>
consteval const char * help_text() {
    return std::define_static_string(std::string_view([:std::meta::constant_of(Annotation):].text));
}

// Empty where the annotation gives none and the type is asked instead.
template <std::meta::info Annotation>
consteval const char * help_placeholder() {
    return std::define_static_string(std::string_view([:std::meta::constant_of(Annotation):].placeholder));
}

template <typename T>
constexpr const char * type_placeholder() {
    using Value = typename value_type<T>::type;
    if constexpr (std::is_same_v<Value, bool>) {
        return "";  // a bool flag takes no value
    } else if constexpr (std::is_floating_point_v<Value>) {
        return "F";
    } else if constexpr (std::is_integral_v<Value>) {
        return "N";
    } else {
        return "TEXT";
    }
}

template <typename T>
constexpr const char * number_kind() {
    if constexpr (std::is_floating_point_v<T>) {
        return "a number";
    } else if constexpr (std::is_unsigned_v<T>) {
        return "a non-negative integer";
    } else {
        return "an integer";
    }
}

// "--max-tokens N", as `--help` prints it before the description.
template <std::meta::info Member>
std::string flag_label() {
    using Value = [:std::meta::type_of(Member):];

    constexpr std::meta::info annotation  = help_of(Member);
    constexpr const char *    given       = help_placeholder<annotation>();
    constexpr const char *    placeholder = given[0] != '\0' ? given : type_placeholder<Value>();

    std::string label = flag_name<Member>();
    if (placeholder[0] != '\0') {
        label += ' ';
        label += placeholder;
    }
    return label;
}

template <typename T>
void read_value(T & out, const char * flag, const std::string & text) {
    if constexpr (is_optional<T>) {
        read_value(out.emplace(), flag, text);
    } else if constexpr (is_vector<T>) {
        read_value(out.emplace_back(), flag, text);
    } else if constexpr (std::is_same_v<T, std::string>) {
        out = text;
    } else {
        const char * end = text.data() + text.size();
        T            value{};
        const std::from_chars_result parsed = std::from_chars(text.data(), end, value);
        if (parsed.ec == std::errc::result_out_of_range) {
            throw FlagError(std::string(flag) + " is out of range");
        }
        // Trailing junk is a mistake, not a value: "12x" is not 12.
        if (parsed.ec != std::errc{} || parsed.ptr != end) {
            throw FlagError(std::string(flag) + " needs " + number_kind<T>());
        }
        out = value;
    }
}

// Sets the member `name` stands for, taking its value from argv and leaving `at` on the last
// argument the flag consumed. False when this struct has no such flag.
template <typename T>
bool assign_flag(T & options, const std::string & name, int argc, char ** argv, int & at) {
    bool matched = false;
    template for (constexpr std::meta::info member : fields_of(^^T)) {
        using Member = [:std::meta::type_of(member):];
        if (!matched && name == flag_name<member>()) {
            matched = true;
            if constexpr (std::is_same_v<Member, bool>) {
                options.[:member:] = true;
            } else {
                if (at + 1 >= argc) {
                    throw FlagError(name + " needs a value");
                }
                read_value(options.[:member:], flag_name<member>(), argv[++at]);
            }
        }
    }
    return matched;
}

template <typename T>
std::size_t widest_label() {
    std::size_t widest = 0;
    template for (constexpr std::meta::info member : fields_of(^^T)) {
        constexpr std::meta::info annotation = help_of(member);
        if constexpr (annotation != std::meta::info{}) {
            widest = std::max(widest, flag_label<member>().size());
        }
    }
    return widest;
}

template <typename T>
void print_struct_flags(std::FILE * out, std::size_t column) {
    template for (constexpr std::meta::info member : fields_of(^^T)) {
        constexpr std::meta::info annotation = help_of(member);
        if constexpr (annotation != std::meta::info{}) {
            const std::string label = flag_label<member>();
            std::string       line  = "  " + label + std::string(column - 2 - label.size(), ' ');
            for (const char c : std::string_view(help_text<annotation>())) {
                line += c;
                if (c == '\n') {
                    line += std::string(column, ' ');
                }
            }
            std::fprintf(out, "%s\n", line.c_str());
        }
    }
}

}  // namespace detail

// Fills the flags of every struct it is given from the command line and returns the arguments
// that are not flags, in order. Throws FlagError on anything the structs cannot take.
template <typename... Structs>
std::vector<std::string> parse_flags(int argc, char ** argv, Structs &... options) {
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        // Only a long option is a flag, so a prompt or a path that starts with a dash stays an
        // argument. -h is the one short option, and every binary spells it --help.
        if (arg.rfind("--", 0) != 0 && arg != "-h") {
            positional.push_back(arg);
            continue;
        }
        const std::string name = arg == "-h" ? "--help" : arg;

        if (!(detail::assign_flag(options, name, argc, argv, i) || ...)) {
            throw FlagError("unknown argument '" + arg + "'");
        }
    }
    return positional;
}

// The option lines of `--help`, one per flag that has a help annotation, aligned across all of
// the structs. The structs themselves are only read for their types.
template <typename... Structs>
void print_flags(std::FILE * out, const Structs &...) {
    const std::size_t column = std::max({detail::widest_label<Structs>()...}) + 3;
    (detail::print_struct_flags<Structs>(out, column), ...);
}

}  // namespace cli
}  // namespace llamad
