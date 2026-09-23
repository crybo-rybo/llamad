/** @file
 * @brief The CHECK and CHECK_EQ macros every test executable shares.
 *
 * No test framework: a failed check prints where it failed and counts, the run carries on, and
 * main() returns report(), which is non-zero if anything failed.
 */

#pragma once

#include <cstdio>
#include <string>
#include <type_traits>

namespace tests {

inline int failures = 0;
inline int checks   = 0;

// What CHECK_EQ compares: text as std::string, so two C strings compare by content and not by
// address, and numbers as themselves.
inline std::string shown(const char * text) { return text; }
inline std::string shown(const std::string & text) { return text; }

template <typename T>
    requires std::is_arithmetic_v<T>
T shown(T value) {
    return value;
}

inline std::string to_display(const std::string & text) { return "'" + text + "'"; }

template <typename T>
    requires std::is_arithmetic_v<T>
std::string to_display(T value) {
    return std::to_string(value);
}

// Prints the summary and returns the exit code for main().
inline int report() {
    std::fprintf(stderr, "%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}

}  // namespace tests

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        ++tests::checks;                                                                  \
        if (!(cond)) {                                                                    \
            ++tests::failures;                                                            \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                                 \
    } while (0)

#define CHECK_EQ(a, b)                                                                           \
    do {                                                                                         \
        ++tests::checks;                                                                         \
        const auto lhs_ = tests::shown(a);                                                       \
        const auto rhs_ = tests::shown(b);                                                       \
        if (!(lhs_ == rhs_)) {                                                                   \
            ++tests::failures;                                                                   \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s == %s\n", __FILE__, __LINE__, #a, #b); \
            std::fprintf(stderr, "    left:  %s\n", tests::to_display(lhs_).c_str());            \
            std::fprintf(stderr, "    right: %s\n", tests::to_display(rhs_).c_str());            \
        }                                                                                        \
    } while (0)
