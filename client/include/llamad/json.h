#pragma once

// The JSON the client needs: the arguments a model sends to a tool, the result the tool sends
// back, and the JSON Schema that describes either. It is driven by a type's members, so a struct
// is the schema and the parser at once. This is deliberately not a general JSON library — it
// reads what a grammar-constrained daemon can produce and rejects the rest.
//
// Values: std::string, bool, integers, floating point, enums (by enumerator name),
// std::optional (absent or null leaves it unset), std::vector, and nested aggregate structs.
// Unknown object keys are skipped; a key missing for a member that is not std::optional is an
// error, as is anything the type cannot take.

#include <array>
#include <charconv>
#include <cstddef>
#include <meta>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace llamad {
namespace client {

// A description of the thing it annotates: a tool function, one of its parameters, or a member
// of a struct, where it becomes the "description" of the schema property. The text is a char
// array because an annotation's value has to be of a structural type.
template <std::size_t N>
struct desc {
    char text[N];

    consteval desc(const char (&literal)[N]) {
        for (std::size_t i = 0; i < N; ++i) {
            text[i] = literal[i];
        }
    }
};

namespace json {

// Thrown by read() for input the type cannot take. `offset` is the byte the reader gave up on.
struct Error : std::runtime_error {
    Error(std::size_t offset, const std::string & message) : std::runtime_error(message), offset(offset) {}
    std::size_t offset;
};

namespace detail {

template <typename T> constexpr bool is_optional                   = false;
template <typename T> constexpr bool is_optional<std::optional<T>> = true;
template <typename T> constexpr bool is_vector                     = false;
template <typename T> constexpr bool is_vector<std::vector<T>>     = true;

// define_static_array outlives the constant evaluation that builds the list, so `template for`
// can iterate it.
consteval auto fields_of(std::meta::info type) {
    return std::define_static_array(
        std::meta::nonstatic_data_members_of(type, std::meta::access_context::unprivileged()));
}

consteval auto values_of(std::meta::info enum_type) {
    return std::define_static_array(std::meta::enumerators_of(enum_type));
}

consteval bool is_desc(std::meta::info annotation) {
    const std::meta::info type = std::meta::type_of(annotation);
    return std::meta::has_template_arguments(type) && std::meta::template_of(type) == ^^client::desc;
}

// The text of X's desc annotation, or "" when it has none. X is a member, a parameter or a
// function.
template <std::meta::info X>
consteval const char * description() {
    template for (constexpr std::meta::info annotation :
                  std::define_static_array(std::meta::annotations_of(X))) {
        if constexpr (is_desc(annotation)) {
            return std::define_static_string(std::string_view([:std::meta::constant_of(annotation):].text));
        }
    }
    // Also from define_static_string, so that the result is a constant a caller can reflect on.
    return std::define_static_string(std::string_view());
}

// A cursor over the input. Every read skips the whitespace before its value and stops on the
// character after it. Defined in json.cpp, so this header carries only the reflective glue.
class Reader {
public:
    explicit Reader(std::string_view text) : text_(text) {}

    bool        consume(char c);  // takes `c` when that is what comes next
    void        expect(char c);
    void        expect_end();     // nothing but whitespace left
    std::string read_string();
    double      read_number();
    long long   read_integer();
    bool        read_bool();
    bool        read_null();      // takes `null` and says whether that is what was next
    void        skip_value();

    [[noreturn]] void fail(const std::string & message) const;

private:
    void             skip_whitespace();
    bool             match(std::string_view word);
    unsigned         read_hex4();
    std::string_view scan_number();
    void             skip_value(std::size_t depth);

    std::string_view text_;
    std::size_t      at_ = 0;
};

// Appends `text` as a JSON string, quotes included.
void write_string(std::string & out, std::string_view text);

// Appends `"name":`, with the comma separating it from whatever is in `out` already.
void write_key(std::string & out, bool & first, std::string_view name);

template <typename T>
void read_value(Reader & in, T & out);

template <typename T>
void read_object(Reader & in, T & out) {
    std::array<bool, fields_of(^^T).size()> seen{};

    in.expect('{');
    if (!in.consume('}')) {
        do {
            const std::string key = in.read_string();
            in.expect(':');

            bool        matched = false;
            std::size_t index   = 0;
            template for (constexpr std::meta::info field : fields_of(^^T)) {
                if (!matched && key == std::define_static_string(std::meta::identifier_of(field))) {
                    read_value(in, out.[:field:]);
                    seen[index] = true;
                    matched     = true;
                }
                ++index;
            }
            if (!matched) {
                in.skip_value();
            }
        } while (in.consume(','));
        in.expect('}');
    }

    std::size_t index = 0;
    template for (constexpr std::meta::info field : fields_of(^^T)) {
        using Field = [:std::meta::type_of(field):];
        if constexpr (!is_optional<Field>) {
            if (!seen[index]) {
                in.fail(std::string("missing key '") +
                        std::define_static_string(std::meta::identifier_of(field)) + "'");
            }
        }
        ++index;
    }
}

template <typename T>
void read_value(Reader & in, T & out) {
    if constexpr (is_optional<T>) {
        // A key that is there but null reads the same as a key that is absent.
        if (in.read_null()) {
            out.reset();
        } else {
            read_value(in, out.emplace());
        }
    } else if constexpr (is_vector<T>) {
        out.clear();
        in.expect('[');
        if (!in.consume(']')) {
            do {
                read_value(in, out.emplace_back());
            } while (in.consume(','));
            in.expect(']');
        }
    } else if constexpr (std::is_same_v<T, std::string>) {
        out = in.read_string();
    } else if constexpr (std::is_same_v<T, bool>) {
        out = in.read_bool();
    } else if constexpr (std::is_enum_v<T>) {
        const std::string name    = in.read_string();
        bool              matched = false;
        template for (constexpr std::meta::info value : values_of(^^T)) {
            if (name == std::define_static_string(std::meta::identifier_of(value))) {
                out     = [:value:];
                matched = true;
            }
        }
        if (!matched) {
            in.fail("'" + name + "' is not a value of " +
                    std::define_static_string(std::meta::identifier_of(^^T)));
        }
    } else if constexpr (std::is_integral_v<T>) {
        out = static_cast<T>(in.read_integer());
    } else if constexpr (std::is_floating_point_v<T>) {
        out = static_cast<T>(in.read_number());
    } else {
        read_object(in, out);
    }
}

template <typename T>
void write_value(std::string & out, const T & value);

template <typename T>
void write_object(std::string & out, const T & value) {
    out += '{';
    bool first = true;
    template for (constexpr std::meta::info field : fields_of(^^T)) {
        using Field                 = [:std::meta::type_of(field):];
        constexpr const char * name = std::define_static_string(std::meta::identifier_of(field));
        if constexpr (is_optional<Field>) {
            // An unset member is left out rather than written as null: absent is how read()
            // spells it too.
            if (value.[:field:]) {
                write_key(out, first, name);
                write_value(out, *value.[:field:]);
            }
        } else {
            write_key(out, first, name);
            write_value(out, value.[:field:]);
        }
    }
    out += '}';
}

template <typename T>
void write_value(std::string & out, const T & value) {
    if constexpr (is_optional<T>) {
        if (value) {
            write_value(out, *value);
        } else {
            out += "null";
        }
    } else if constexpr (is_vector<T>) {
        out += '[';
        for (const auto & element : value) {
            if (&element != &value.front()) {
                out += ',';
            }
            write_value(out, element);
        }
        out += ']';
    } else if constexpr (std::is_same_v<T, std::string>) {
        write_string(out, value);
    } else if constexpr (std::is_same_v<T, bool>) {
        out += value ? "true" : "false";
    } else if constexpr (std::is_enum_v<T>) {
        template for (constexpr std::meta::info enumerator : values_of(^^T)) {
            if (value == [:enumerator:]) {
                write_string(out, std::define_static_string(std::meta::identifier_of(enumerator)));
            }
        }
    } else if constexpr (std::is_integral_v<T>) {
        out += std::to_string(value);
    } else if constexpr (std::is_floating_point_v<T>) {
        char buffer[32];
        out.append(buffer, std::to_chars(buffer, buffer + sizeof(buffer), value).ptr);
    } else {
        write_object(out, value);
    }
}

// The body of the schema object for one value of type T: everything between the braces, so the
// caller can add a "description" of its own.
template <typename T>
void write_type_schema(std::string & out);

template <typename T>
void write_object_schema(std::string & out, std::span<const char * const> descriptions) {
    out += R"("type":"object","properties":{)";

    std::string required;
    bool        first_property = true;
    bool        first_required = true;
    std::size_t index          = 0;

    template for (constexpr std::meta::info field : fields_of(^^T)) {
        using Field                 = [:std::meta::type_of(field):];
        constexpr const char * name = std::define_static_string(std::meta::identifier_of(field));

        write_key(out, first_property, name);
        out += '{';
        write_type_schema<Field>(out);
        const char * text = index < descriptions.size() ? descriptions[index] : description<field>();
        if (text[0] != '\0') {
            out += R"(,"description":)";
            write_string(out, text);
        }
        out += '}';

        if constexpr (!is_optional<Field>) {
            if (!first_required) {
                required += ',';
            }
            first_required = false;
            write_string(required, name);
        }
        ++index;
    }

    out += R"(},"required":[)" + required + ']';
}

template <typename T>
void write_type_schema(std::string & out) {
    if constexpr (is_optional<T>) {
        write_type_schema<typename T::value_type>(out);
    } else if constexpr (is_vector<T>) {
        out += R"("type":"array","items":{)";
        write_type_schema<typename T::value_type>(out);
        out += '}';
    } else if constexpr (std::is_same_v<T, std::string>) {
        out += R"("type":"string")";
    } else if constexpr (std::is_same_v<T, bool>) {
        out += R"("type":"boolean")";
    } else if constexpr (std::is_enum_v<T>) {
        out += R"("type":"string","enum":[)";
        bool first = true;
        template for (constexpr std::meta::info enumerator : values_of(^^T)) {
            if (!first) {
                out += ',';
            }
            first = false;
            write_string(out, std::define_static_string(std::meta::identifier_of(enumerator)));
        }
        out += ']';
    } else if constexpr (std::is_integral_v<T>) {
        out += R"("type":"integer")";
    } else if constexpr (std::is_floating_point_v<T>) {
        out += R"("type":"number")";
    } else {
        write_object_schema<T>(out, {});
    }
}

}  // namespace detail

// Fills `out` from one JSON value, which must be the whole of `text`.
template <typename T>
void read(std::string_view text, T & out) {
    detail::Reader in(text);
    detail::read_value(in, out);
    in.expect_end();
}

template <typename T>
std::string write(const T & value) {
    std::string out;
    detail::write_value(out, value);
    return out;
}

// A JSON Schema object for T: one property per member, its desc annotation as the description,
// and every member that is not std::optional listed in "required".
//
// `descriptions` replaces those annotations positionally. That is for a T whose members were
// synthesised from somewhere else and so carry none of their own — a tool function's parameter
// list, where the descriptions are annotations on the parameters.
template <typename T>
std::string schema(std::span<const char * const> descriptions = {}) {
    std::string out = "{";
    detail::write_object_schema<T>(out, descriptions);
    out += '}';
    return out;
}

}  // namespace json
}  // namespace client
}  // namespace llamad
