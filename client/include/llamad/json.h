#pragma once

// Reflection over tool arguments, results and schemas. nlohmann handles JSON syntax and
// serialization; this adapter maps ordinary C++ types to it, with checked numeric conversions.
// Values are strings, booleans, numbers, enums, optionals, vectors and aggregate structs.
// Unknown keys are ignored, missing required members are errors, and absent or null optionals
// are unset. Enums use their names; unset optional members are omitted when writing objects.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <meta>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

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
namespace detail {

// Declaration order is also the order presented to the model in its tool schema.
using Json = nlohmann::ordered_json;

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

template <typename T>
void read_value(const Json & in, T & out) {
    if constexpr (is_optional<T>) {
        if (in.is_null()) {
            out.reset();
        } else {
            read_value(in, out.emplace());
        }
    } else if constexpr (is_vector<T>) {
        if (!in.is_array()) {
            throw std::invalid_argument("json: expected an array");
        }
        out.clear();
        out.reserve(in.size());
        for (const Json & element : in) {
            typename T::value_type value{};
            read_value(element, value);
            out.push_back(std::move(value));
        }
    } else if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, bool>) {
        in.get_to(out);
    } else if constexpr (std::is_enum_v<T>) {
        const std::string name = in.get<std::string>();
        template for (constexpr std::meta::info value : values_of(^^T)) {
            if (name == std::define_static_string(std::meta::identifier_of(value))) {
                out = [:value:];
                return;
            }
        }
        throw std::invalid_argument("json: '" + name + "' is not a value of " +
                                    std::define_static_string(std::meta::identifier_of(^^T)));
    } else if constexpr (std::is_integral_v<T>) {
        // get<T>() permits narrowing and signed/unsigned conversions without range checks.
        if (!in.is_number_integer()) {
            throw std::invalid_argument("json: expected an integer");
        }
        // in_range accepts signed/unsigned integer types, so normalize char types too.
        using Integer = std::conditional_t<std::is_signed_v<T>, std::make_signed_t<T>, std::make_unsigned_t<T>>;
        const bool fits = in.is_number_unsigned()
            ? std::in_range<Integer>(in.get<uint64_t>())
            : std::in_range<Integer>(in.get<int64_t>());
        if (!fits) {
            throw std::invalid_argument("json: integer out of range");
        }
        in.get_to(out);
    } else if constexpr (std::is_floating_point_v<T>) {
        if (!in.is_number()) {
            throw std::invalid_argument("json: expected a number");
        }
        const double value = in.get<double>();
        if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<T>::max()) {
            throw std::invalid_argument("json: number out of range");
        }
        out = static_cast<T>(value);
    } else {
        if (!in.is_object()) {
            throw std::invalid_argument("json: expected an object");
        }
        template for (constexpr std::meta::info field : fields_of(^^T)) {
            using Field = [:std::meta::type_of(field):];
            constexpr const char * name = std::define_static_string(std::meta::identifier_of(field));
            const auto it = in.find(name);
            if (it != in.end()) {
                read_value(*it, out.[:field:]);
            } else if constexpr (is_optional<Field>) {
                out.[:field:].reset();
            } else {
                throw std::invalid_argument(std::string("json: missing key '") + name + "'");
            }
        }
    }
}

template <typename T>
Json write_value(const T & value) {
    if constexpr (is_optional<T>) {
        return value ? write_value(*value) : Json(nullptr);
    } else if constexpr (is_vector<T>) {
        Json out = Json::array();
        for (const auto & element : value) {
            out.push_back(write_value(element));
        }
        return out;
    } else if constexpr (std::is_enum_v<T>) {
        template for (constexpr std::meta::info enumerator : values_of(^^T)) {
            if (value == [:enumerator:]) {
                return std::define_static_string(std::meta::identifier_of(enumerator));
            }
        }
        throw std::invalid_argument("json: unnamed enum value");
    } else if constexpr (std::is_floating_point_v<T>) {
        // nlohmann writes non-finite values as null, which would conceal an invalid tool result.
        if (!std::isfinite(value)) {
            throw std::invalid_argument("json: non-finite number");
        }
        if (std::abs(value) > std::numeric_limits<Json::number_float_t>::max()) {
            throw std::invalid_argument("json: number out of range");
        }
        return value;
    } else if constexpr (std::is_arithmetic_v<T> || std::is_same_v<T, std::string>) {
        return value;
    } else {
        Json out = Json::object();
        template for (constexpr std::meta::info field : fields_of(^^T)) {
            using Field = [:std::meta::type_of(field):];
            constexpr const char * name = std::define_static_string(std::meta::identifier_of(field));
            if constexpr (is_optional<Field>) {
                if (value.[:field:]) {
                    out[name] = write_value(*value.[:field:]);
                }
            } else {
                out[name] = write_value(value.[:field:]);
            }
        }
        return out;
    }
}

template <typename T>
Json type_schema();

template <typename T>
Json object_schema(std::span<const char * const> descriptions) {
    Json out = {{"type", "object"}, {"properties", Json::object()}, {"required", Json::array()}};
    std::size_t index = 0;
    template for (constexpr std::meta::info field : fields_of(^^T)) {
        using Field = [:std::meta::type_of(field):];
        constexpr const char * name = std::define_static_string(std::meta::identifier_of(field));
        Json property = type_schema<Field>();
        const char * text = index < descriptions.size() ? descriptions[index] : description<field>();
        if (text[0] != '\0') {
            property["description"] = text;
        }
        out["properties"][name] = std::move(property);
        if constexpr (!is_optional<Field>) {
            out["required"].push_back(name);
        }
        ++index;
    }
    return out;
}

template <typename T>
Json type_schema() {
    if constexpr (is_optional<T>) {
        return type_schema<typename T::value_type>();
    } else if constexpr (is_vector<T>) {
        return {{"type", "array"}, {"items", type_schema<typename T::value_type>()}};
    } else if constexpr (std::is_same_v<T, std::string>) {
        return {{"type", "string"}};
    } else if constexpr (std::is_same_v<T, bool>) {
        return {{"type", "boolean"}};
    } else if constexpr (std::is_enum_v<T>) {
        Json names = Json::array();
        template for (constexpr std::meta::info enumerator : values_of(^^T)) {
            names.push_back(std::define_static_string(std::meta::identifier_of(enumerator)));
        }
        return {{"type", "string"}, {"enum", std::move(names)}};
    } else if constexpr (std::is_integral_v<T>) {
        return {{"type", "integer"}};
    } else if constexpr (std::is_floating_point_v<T>) {
        return {{"type", "number"}};
    } else {
        return object_schema<T>({});
    }
}

}  // namespace detail

// Invalid JSON or incompatible values throw nlohmann exceptions or std::invalid_argument.
template <typename T>
void read(std::string_view text, T & out) {
    const auto value = detail::Json::parse(text, [](int depth, detail::Json::parse_event_t, detail::Json &) {
        if (depth > 64) {
            throw std::invalid_argument("json: nested too deeply");
        }
        return true;
    });
    detail::read_value(value, out);
}

template <typename T>
std::string write(const T & value) {
    return detail::write_value(value).dump();
}

// Non-optional members are required; descriptions override member annotations positionally for
// the aggregate synthesised from a tool function's parameter list.
template <typename T>
std::string schema(std::span<const char * const> descriptions = {}) {
    return detail::object_schema<T>(descriptions).dump();
}

}  // namespace json
}  // namespace client
}  // namespace llamad
