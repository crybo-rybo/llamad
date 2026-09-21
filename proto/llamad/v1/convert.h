#pragma once

// Conversion between the messages in llamad.proto and the plain structs that mirror them on
// either side of the wire (engine.h and chat_format.h in the daemon, client.h in the client).
//
// A mirror struct names its fields exactly as the message does, so the mapping is derived from
// those names at compile time instead of being written out per field. A field the message does
// not have fails to compile, which is what keeps the two from drifting apart.
//
// Nothing here names a protobuf type: the message is only ever reached through the accessors
// protoc generates for each field `x`: x(), set_x(), has_x(), add_x() and clear_x().

#include <cstddef>
#include <meta>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace llamad {
namespace wire {
namespace detail {

template <typename T> constexpr bool is_optional                   = false;
template <typename T> constexpr bool is_optional<std::optional<T>> = true;
template <typename T> constexpr bool is_vector                     = false;
template <typename T> constexpr bool is_vector<std::vector<T>>     = true;

// A `repeated` field of messages rather than of scalars: the element mirrors a message too.
template <typename T> constexpr bool is_nested_vector                 = false;
template <typename T> constexpr bool is_nested_vector<std::vector<T>> = std::is_aggregate_v<T>;

// The element of a repeated field's container, named the way from_proto reads it out.
template <typename Container> using value_type_of = typename Container::value_type;

// One of the traits above, applied to a reflected type: trait(^^is_optional, t) is
// is_optional<[:t:]>, so the checks below classify a field exactly as the converters do.
consteval bool trait(std::meta::info variable_template, std::meta::info type) {
    return std::meta::extract<bool>(std::meta::substitute(variable_template, {type}));
}

// std::optional<U> and std::vector<U> each hold one U.
consteval std::meta::info held_type(std::meta::info type) {
    return std::meta::template_arguments_of(type)[0];
}

// The three walks below only ever look at what a caller of the message could call anyway.
// define_static_array outlives the constant evaluation that builds the list, so `template for`
// can iterate it.
consteval auto fields_of(std::meta::info type) {
    return std::define_static_array(
        std::meta::nonstatic_data_members_of(type, std::meta::access_context::unprivileged()));
}

consteval auto values_of(std::meta::info enum_type) {
    return std::define_static_array(std::meta::enumerators_of(enum_type));
}

consteval auto members_of(std::meta::info type) {
    return std::meta::members_of(type, std::meta::access_context::unprivileged());
}

// The null reflection when `message` has no such accessor. Overloads differ by arity
// (stop() and stop(int)); setters that take a string are function templates.
consteval std::meta::info find_accessor(std::meta::info message, std::string_view name, size_t arity) {
    std::meta::info function_template{};
    for (std::meta::info member : members_of(message)) {
        if (!std::meta::has_identifier(member) || std::meta::identifier_of(member) != name) {
            continue;
        }
        if (std::meta::is_function_template(member)) {
            function_template = member;
        } else if (std::meta::is_function(member) && std::meta::parameters_of(member).size() == arity) {
            return member;
        }
    }
    return arity > 0 ? function_template : std::meta::info{};
}

consteval std::meta::info accessor(std::meta::info message, std::string_view prefix, std::meta::info field,
                                   size_t arity) {
    const std::string name = std::string(prefix) + std::string(std::meta::identifier_of(field));
    const std::meta::info found = find_accessor(message, name, arity);
    if (found == std::meta::info{}) {
        throw std::meta::exception(std::string(std::meta::display_string_of(message)) + " has no " + name +
                                       "(): the struct and llamad.proto have drifted apart",
                                   field);
    }
    return found;
}

// One field of a mirror struct against the message: same name, same presence, same value type.
// Throws, so the caller's constant evaluation fails with a message naming what disagrees.
consteval void check_field(std::meta::info message, std::meta::info field) {
    const std::string     name   = std::string(std::meta::identifier_of(field));
    const std::string     drift  = ": the struct and llamad.proto have drifted apart";
    const std::meta::info stored = std::meta::type_of(field);
    const std::string     what =
        std::string(std::meta::display_string_of(std::meta::parent_of(field))) + " field '" + name + "' ";

    // Same name: the getter both converters reach for has to exist.
    const std::meta::info get = accessor(message, "", field, /*arity*/ 0);

    // Same presence. A proto3 `optional` scalar is what has a has_<name>(), and std::optional is
    // what the converters pair with it: an unset one stays off the wire so the receiver's default
    // stands, where a plain field is always sent, zero and all. (protoc gives a singular message
    // field a has_<name>() as well; no struct mirrored here has one.)
    const bool optional_here  = trait(^^is_optional, stored);
    const bool optional_there = find_accessor(message, "has_" + name, /*arity*/ 0) != std::meta::info{};
    if (optional_here != optional_there) {
        throw std::meta::exception(what +
                                       (optional_here ? "is std::optional, the message has no has_" + name + "()"
                                                      : "is not std::optional, the message has has_" + name + "()") +
                                       drift,
                                   field);
    }

    // A repeated field of messages: the element types mirror one another rather than being the
    // same type, and contract_test asserts that pair on its own.
    if (trait(^^is_nested_vector, stored)) {
        return;
    }

    // Same type, compared on the value that actually crosses, so a struct field narrower than the
    // message's cannot truncate in silence.
    std::meta::info here  = stored;
    std::meta::info there = std::meta::remove_cvref(std::meta::return_type_of(get));
    if (optional_here) {
        here = held_type(stored);
    } else if (trait(^^is_vector, stored)) {
        if (!std::meta::can_substitute(^^value_type_of, {there})) {
            throw std::meta::exception(what + "is a std::vector, the message's " + name +
                                           "() is not a repeated field" + drift,
                                       field);
        }
        here  = held_type(stored);
        there = std::meta::substitute(^^value_type_of, {there});
    }
    here  = std::meta::dealias(here);
    there = std::meta::dealias(there);
    if (here != there) {
        throw std::meta::exception(what + "is " + std::string(std::meta::display_string_of(here)) +
                                       ", the message has " + std::string(std::meta::display_string_of(there)) +
                                       drift,
                                   field);
    }
}

template <std::meta::info Function, typename P, typename V>
void call(P * out, const V & value) {
    if constexpr (std::meta::is_function_template(Function)) {
        out->template [:Function:]<>(value);
    } else {
        out->[:Function:](value);
    }
}

// "FinishReason" -> "FINISH_REASON_", the prefix protobuf style puts on every value of the enum.
consteval std::string value_prefix(std::meta::info enum_type) {
    std::string out;
    for (char c : std::meta::identifier_of(enum_type)) {
        if (c >= 'A' && c <= 'Z' && !out.empty()) {
            out += '_';
        }
        out += (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c;
    }
    return out + '_';
}

// The enumerator's name without that prefix: "TOOL_CALLS" for FINISH_REASON_TOOL_CALLS.
consteval std::string_view short_name(std::meta::info enumerator) {
    const std::string_view name   = std::meta::identifier_of(enumerator);
    const std::string      prefix = value_prefix(std::meta::parent_of(enumerator));
    return name.starts_with(prefix) ? name.substr(prefix.size()) : name;
}

// Spelling differences between the two sides disappear: ToolCalls and TOOL_CALLS are "toolcalls".
consteval std::string folded_name(std::meta::info enumerator) {
    std::string out;
    for (char c : short_name(enumerator)) {
        if (c != '_') {
            out += (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
        }
    }
    return out;
}

consteval std::meta::info same_value_in(std::meta::info enum_type, std::meta::info enumerator) {
    for (std::meta::info candidate : std::meta::enumerators_of(enum_type)) {
        if (folded_name(candidate) == folded_name(enumerator)) {
            return candidate;
        }
    }
    return std::meta::info{};
}

}  // namespace detail

template <typename T, typename P>
void to_proto(const T & in, P * out) {
    template for (constexpr std::meta::info field : detail::fields_of(^^T)) {
        using Field = [:std::meta::type_of(field):];
        const Field & value = in.[:field:];

        if constexpr (detail::is_nested_vector<Field>) {
            constexpr std::meta::info add = detail::accessor(^^P, "add_", field, /*arity*/ 0);
            for (const auto & element : value) {
                to_proto(element, out->[:add:]());
            }
        } else if constexpr (detail::is_vector<Field>) {
            constexpr std::meta::info add = detail::accessor(^^P, "add_", field, /*arity*/ 1);
            for (const auto & element : value) {
                detail::call<add>(out, element);
            }
        } else if constexpr (detail::is_optional<Field>) {
            // An unset optional stays unset on the wire, so the receiver's default applies.
            constexpr std::meta::info set = detail::accessor(^^P, "set_", field, /*arity*/ 1);
            if (value) {
                detail::call<set>(out, *value);
            }
        } else {
            constexpr std::meta::info set = detail::accessor(^^P, "set_", field, /*arity*/ 1);
            detail::call<set>(out, value);
        }
    }
}

template <typename T, typename P>
T from_proto(const P & in) {
    T out;
    template for (constexpr std::meta::info field : detail::fields_of(^^T)) {
        using Field = [:std::meta::type_of(field):];
        constexpr std::meta::info get = detail::accessor(^^P, "", field, /*arity*/ 0);

        if constexpr (detail::is_nested_vector<Field>) {
            for (const auto & element : in.[:get:]()) {
                out.[:field:].push_back(from_proto<typename Field::value_type>(element));
            }
        } else if constexpr (detail::is_vector<Field>) {
            out.[:field:].assign(in.[:get:]().begin(), in.[:get:]().end());
        } else if constexpr (detail::is_optional<Field>) {
            constexpr std::meta::info has = detail::accessor(^^P, "has_", field, /*arity*/ 0);
            if (in.[:has:]()) {
                out.[:field:] = in.[:get:]();
            }
        } else {
            out.[:field:] = in.[:get:]();
        }
    }
    return out;
}

// True when T and message P have exactly the same fields, each with the same presence —
// std::optional against a proto3 `optional` — and the same value type; a compile error naming the
// field and what disagrees otherwise. to_proto and from_proto already reject a struct field the
// message lacks; this also catches a field added to the message and not to the struct, and one
// that agrees by name while narrowing or dropping the value it carries.
template <typename T, typename P>
consteval bool mirrors() {
    // Every field of the struct is a field of the message and matches it, or check_field throws.
    for (std::meta::info field : detail::fields_of(^^T)) {
        detail::check_field(^^P, field);
    }

    // And the other way round: protoc gives every field of a message its own clear_<name>().
    constexpr std::string_view clear = "clear_";
    for (std::meta::info member : detail::members_of(^^P)) {
        if (!std::meta::is_function(member) || !std::meta::has_identifier(member) ||
            !std::meta::identifier_of(member).starts_with(clear)) {
            continue;
        }
        const std::string_view name = std::meta::identifier_of(member).substr(clear.size());
        bool found = false;
        for (std::meta::info field : detail::fields_of(^^T)) {
            found = found || std::meta::identifier_of(field) == name;
        }
        if (!found) {
            throw std::meta::exception(std::string(std::meta::display_string_of(^^T)) + " has no field '" +
                                           std::string(name) +
                                           "': the struct and llamad.proto have drifted apart",
                                       member);
        }
    }
    return true;
}

// The same check as a value: true where mirrors<T, P>() is the compile error instead. It is how
// contract_test asserts that a struct deliberately out of step with its message is rejected.
template <typename T, typename P>
consteval bool drifted() {
    try {
        return !mirrors<T, P>();
    } catch (const std::meta::exception &) {
        return true;
    }
}

// Converts between an enum in llamad.proto and its plain twin by value name, ignoring the
// FINISH_REASON_ style prefix and case: FINISH_REASON_TOOL_CALLS is FinishReason::ToolCalls.
// Every value of From must exist in To, or the call does not compile.
template <typename To, typename From>
constexpr To enum_cast(From value) {
    template for (constexpr std::meta::info enumerator : detail::values_of(^^From)) {
        constexpr std::meta::info same = detail::same_value_in(^^To, enumerator);
        static_assert(same != std::meta::info{}, "enum_cast: a source value has no twin in the target");
        if (value == [:enumerator:]) {
            return [:same:];
        }
    }
    return To{};
}

// The same, for a From with values To lacks (UNSPECIFIED, or one from a later daemon).
template <typename To, typename From>
constexpr To enum_cast(From value, To fallback) {
    template for (constexpr std::meta::info enumerator : detail::values_of(^^From)) {
        constexpr std::meta::info same = detail::same_value_in(^^To, enumerator);
        if constexpr (same != std::meta::info{}) {
            if (value == [:enumerator:]) {
                return [:same:];
            }
        }
    }
    return fallback;
}

// The value's name as llamad.proto spells it, without the prefix: "TOOL_CALLS".
template <typename E>
const char * value_name(E value) {
    template for (constexpr std::meta::info enumerator : detail::values_of(^^E)) {
        if (value == [:enumerator:]) {
            return std::define_static_string(detail::short_name(enumerator));
        }
    }
    return "UNSPECIFIED";
}

}  // namespace wire
}  // namespace llamad
