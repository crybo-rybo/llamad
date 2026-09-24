/** @file
 * @brief The engine's prompt-cache bookkeeping, checked without a model.
 */

#include "check.h"
#include "engine.h"

#include <cstdint>
#include <vector>

namespace {

using Tokens = std::vector<int32_t>;

void test_nothing_is_kept_from_an_empty_cache() {
    CHECK_EQ(llamad::reusable_prefix({}, {1, 2, 3}), 0u);
}

void test_the_shared_prefix_is_kept() {
    CHECK_EQ(llamad::reusable_prefix({1, 2, 3, 4}, {1, 2, 9, 4}), 2u);
    CHECK_EQ(llamad::reusable_prefix({5, 2, 3}, {1, 2, 3}), 0u);
}

void test_a_prompt_that_extends_the_cache_keeps_all_of_it() {
    // The next chat turn: the previous prompt and reply, then new messages.
    const Tokens cached = {1, 2, 3};
    CHECK_EQ(llamad::reusable_prefix(cached, {1, 2, 3, 4, 5}), 3u);
}

void test_a_cache_that_runs_past_the_prompt_is_cut_back_to_it() {
    // A reply the next prompt does not repeat, or an edited earlier turn.
    CHECK_EQ(llamad::reusable_prefix({1, 2, 3, 4, 5}, {1, 2, 3, 7}), 3u);
}

void test_the_last_prompt_token_is_always_decoded() {
    // Sampling needs the logits of the prompt's last token, which only decoding it produces.
    CHECK_EQ(llamad::reusable_prefix({1, 2, 3}, {1, 2, 3}), 2u);
    CHECK_EQ(llamad::reusable_prefix({1, 2, 3, 4}, {1, 2, 3}), 2u);
    CHECK_EQ(llamad::reusable_prefix({1}, {1}), 0u);
}

}  // namespace

int main() {
    test_nothing_is_kept_from_an_empty_cache();
    test_the_shared_prefix_is_kept();
    test_a_prompt_that_extends_the_cache_keeps_all_of_it();
    test_a_cache_that_runs_past_the_prompt_is_cut_back_to_it();
    test_the_last_prompt_token_is_always_decoded();
    return tests::report();
}
