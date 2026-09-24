/** @file
 * @brief Model ownership, device selection, sampling and safe streaming through libllama.
 */

#include "engine.h"

#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <ranges>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace llamad {

namespace {

// ---------------------------------------------------------------------------
// llama.cpp / ggml logging: only warnings and errors reach stderr.
// ---------------------------------------------------------------------------

/// GGML_LOG_LEVEL_CONT means "continuation of the previous log line", so its own
/// level carries no information: remember the level of the last real line and let
/// continuations inherit it.
void log_callback(ggml_log_level level, const char * text, void * /*user_data*/) {
    // ggml logs from worker threads too, so keep the remembered level atomic.
    static std::atomic<int> last_level{GGML_LOG_LEVEL_NONE};

    ggml_log_level effective = level;
    if (level == GGML_LOG_LEVEL_CONT) {
        effective = static_cast<ggml_log_level>(last_level.load(std::memory_order_relaxed));
    } else {
        last_level.store(level, std::memory_order_relaxed);
    }

    if (effective == GGML_LOG_LEVEL_WARN || effective == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

/// Initialize backend registration and logging once across all Engine instances.
void init_llama_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(log_callback, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
    });
}

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

/// `ggml_backend_dev_type` names both the enum and a function, and the function
/// hides the type, so the type always needs its elaborated `enum` form here.
const char * device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "igpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accel";
        case GGML_BACKEND_DEVICE_TYPE_META:  return "meta";
    }
    return "unknown";
}

/// Return the backend's device identifier, or empty when unavailable.
std::string device_name(ggml_backend_dev_t dev) {
    const char * name = ggml_backend_dev_name(dev);
    return name != nullptr ? std::string(name) : std::string();
}

/// Return the backend's human-readable device description, or empty.
std::string device_description(ggml_backend_dev_t dev) {
    const char * desc = ggml_backend_dev_description(dev);
    return desc != nullptr ? std::string(desc) : std::string();
}

/// Every device the compiled-in backends registered, in backend order.
std::vector<ggml_backend_dev_t> all_devices() {
    std::vector<ggml_backend_dev_t> out;
    const size_t n = ggml_backend_dev_count();
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev != nullptr) {
            out.push_back(dev);
        }
    }
    return out;
}

/// Format the selected device identifiers for configuration diagnostics.
std::string join_device_names(const std::vector<ggml_backend_dev_t> & devices) {
    std::string out;
    for (ggml_backend_dev_t dev : devices) {
        if (!out.empty()) {
            out += ", ";
        }
        out += device_name(dev);
    }
    return out;
}

/// The devices llama.cpp offloads to when no explicit list is given: every
/// discrete GPU, or the integrated GPUs if there is no discrete one.
std::vector<ggml_backend_dev_t> default_offload_devices(const std::vector<ggml_backend_dev_t> & devices) {
    std::vector<ggml_backend_dev_t> gpus;
    std::vector<ggml_backend_dev_t> igpus;
    for (ggml_backend_dev_t dev : devices) {
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_GPU:  gpus.push_back(dev); break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU: igpus.push_back(dev); break;
            default:                            break;
        }
    }
    return gpus.empty() ? igpus : gpus;
}

/// The devices the model's layers will land on: the ones the caller named, in that order, or
/// llama.cpp's default when none were named.
std::vector<ggml_backend_dev_t> select_offload_devices(const EngineConfig & config,
                                                       const std::vector<ggml_backend_dev_t> & devices) {
    if (config.devices.empty()) {
        return default_offload_devices(devices);
    }

    std::vector<ggml_backend_dev_t> offload;
    for (const std::string & wanted : config.devices) {
        const auto it = std::find_if(devices.begin(), devices.end(), [&](ggml_backend_dev_t dev) {
            return device_name(dev) == wanted;
        });
        if (it == devices.end()) {
            throw EngineError("unknown device '" + wanted + "'; available devices are: " +
                              join_device_names(devices));
        }
        if (std::find(offload.begin(), offload.end(), *it) != offload.end()) {
            throw EngineError("device '" + wanted + "' is listed more than once");
        }
        offload.push_back(*it);
    }
    return offload;
}

/// The tensor split as the fixed-size array llama.cpp reads, llama_max_devices() long, or empty
/// when none was given. It is validated against the `n_offload` devices it will be applied to
/// before it is padded out.
std::vector<float> padded_tensor_split(const EngineConfig & config, size_t n_offload) {
    if (config.tensor_split.empty()) {
        return {};
    }

    float sum = 0.0f;
    for (size_t i = 0; i < config.tensor_split.size(); ++i) {
        // Written so that a NaN fails the test too.
        if (!(config.tensor_split[i] >= 0.0f)) {
            throw EngineError("tensor split entry " + std::to_string(i + 1) +
                              " is negative or not a number");
        }
        sum += config.tensor_split[i];
    }
    if (sum <= 0.0f) {
        throw EngineError("tensor split is all zeros: it would leave every device without work");
    }
    if (n_offload == 0) {
        throw EngineError("a tensor split was given but there is no GPU to offload to");
    }
    if (config.tensor_split.size() > n_offload) {
        throw EngineError("tensor split has " + std::to_string(config.tensor_split.size()) +
                          " entries but only " + std::to_string(n_offload) +
                          (config.devices.empty() ? " device(s) will be used for offloading"
                                                  : " device(s) were requested"));
    }
    if (config.tensor_split.size() > llama_max_devices()) {
        throw EngineError("tensor split has " + std::to_string(config.tensor_split.size()) +
                          " entries but llama.cpp supports at most " +
                          std::to_string(llama_max_devices()) + " devices");
    }

    std::vector<float> padded(llama_max_devices(), 0.0f);
    std::copy(config.tensor_split.begin(), config.tensor_split.end(), padded.begin());
    return padded;
}

/// Classify discrete and integrated GPUs as offload-capable devices.
bool is_gpu_device(ggml_backend_dev_t dev) {
    const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

/// Convert a byte count to binary gibibytes for device reporting.
double as_gib(uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

/// One line per device the model was offloaded to, read back after the model and
/// the KV cache are resident so the free figure reflects what the load actually
/// cost. llama.cpp's own INFO output is suppressed by log_callback, so this is
/// the only place the device setup becomes visible.
void report_offload_devices(const EngineConfig & config, const std::vector<ggml_backend_dev_t> & offload) {
    const bool has_gpu = std::any_of(offload.begin(), offload.end(), is_gpu_device);

    if (!has_gpu) {
        std::fprintf(stderr, "[llamad] no GPU device available: inference is CPU-only\n");
        return;
    }
    if (config.n_gpu_layers == 0) {
        std::fprintf(stderr, "[llamad] no layers offloaded (n_gpu_layers = 0): inference is CPU-only\n");
        return;
    }

    for (ggml_backend_dev_t dev : offload) {
        size_t free_bytes  = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        std::fprintf(stderr, "[llamad] device %s: %s (%.1f GiB free of %.1f GiB)\n",
                     device_name(dev).c_str(), device_description(dev).c_str(),
                     as_gib(free_bytes), as_gib(total_bytes));
    }
    std::fflush(stderr);
}

// ---------------------------------------------------------------------------
// UTF-8 / stop-string aware streaming
// ---------------------------------------------------------------------------

/// Number of trailing bytes of `s` that form an incomplete UTF-8 sequence.
/// Returns 0 when the buffer ends on a complete (or simply invalid) sequence,
/// in which case nothing needs to be held back.
size_t incomplete_utf8_tail(const std::string & s) {
    const size_t n        = s.size();
    const size_t max_look = std::min<size_t>(n, 4);

    for (size_t k = 1; k <= max_look; ++k) {
        const unsigned char c = static_cast<unsigned char>(s[n - k]);
        if ((c & 0xC0) == 0x80) {
            continue;  // continuation byte, keep scanning backwards for the lead byte
        }

        size_t need = 0;
        if ((c & 0x80) == 0x00) {
            need = 1;
        } else if ((c & 0xE0) == 0xC0) {
            need = 2;
        } else if ((c & 0xF0) == 0xE0) {
            need = 3;
        } else if ((c & 0xF8) == 0xF0) {
            need = 4;
        } else {
            return 0;  // invalid lead byte: not worth holding back, emit as-is
        }

        return k < need ? k : 0;
    }

    return 0;  // only continuation bytes (invalid): emit as-is
}

/// Length of the longest suffix of `pending` that is a proper prefix of one of
/// the stop strings. That suffix must be held back: it may yet become a match.
size_t stop_prefix_holdback(const std::string & pending, const std::vector<std::string> & stops) {
    size_t best = 0;
    for (const std::string & stop : stops) {
        if (stop.empty()) {
            continue;
        }
        const size_t max_len = std::min(pending.size(), stop.size() - 1);
        for (size_t len = max_len; len > best; --len) {
            if (pending.compare(pending.size() - len, len, stop, 0, len) == 0) {
                best = len;
                break;
            }
        }
    }
    return best;
}

/// Buffers generated text and hands it to the callback as soon as it is known to
/// be deliverable: never an incomplete UTF-8 sequence, and never bytes that could
/// still turn out to be the start of a stop string.
class StreamFilter {
public:
    /// Outcome of delivering one token piece.
    enum class Status {
        Continue,  ///< More input may be generated.
        Stopped,   ///< A stop string matched.
        Cancelled  ///< The receiver declined a chunk.
    };

    /// Borrow stop strings and callback for this generation; both must outlive the filter.
    StreamFilter(const std::vector<std::string> & stops, const ChunkCallback & on_chunk) :
        stops_(stops), on_chunk_(on_chunk) {}

    /// Buffer a token piece and emit only the prefix safe from UTF-8 and stop-string splits.
    Status push(const std::string & piece) {
        pending_ += piece;

        // A stop string can never straddle already-delivered text, because any
        // suffix that is a prefix of a stop string is held back.
        size_t match = std::string::npos;
        for (const std::string & stop : stops_) {
            if (stop.empty()) {
                continue;
            }
            const size_t p = pending_.find(stop);
            if (p != std::string::npos && p < match) {
                match = p;
            }
        }

        if (match != std::string::npos) {
            std::string head = pending_.substr(0, match);
            head.resize(head.size() - incomplete_utf8_tail(head));
            pending_.clear();
            if (!deliver(head)) {
                return Status::Cancelled;
            }
            return Status::Stopped;
        }

        const size_t hold = std::min(pending_.size(),
                                     std::max(incomplete_utf8_tail(pending_),
                                              stop_prefix_holdback(pending_, stops_)));

        const size_t safe = pending_.size() - hold;
        if (safe > 0) {
            std::string out = pending_.substr(0, safe);
            pending_.erase(0, safe);
            if (!deliver(out)) {
                return Status::Cancelled;
            }
        }

        return Status::Continue;
    }

    /// Called when generation ended on its own: a held-back stop prefix that never
    /// completed is real output and must be delivered; an incomplete UTF-8 tail is
    /// dropped because it can never become a valid character.
    void flush() {
        if (pending_.empty()) {
            return;
        }
        std::string out;
        out.swap(pending_);
        out.resize(out.size() - incomplete_utf8_tail(out));
        deliver(out);
    }

private:
    /// Deliver nonempty text, treating an empty callback as acceptance.
    bool deliver(const std::string & text) {
        return text.empty() || !on_chunk_ || on_chunk_(text);
    }

    const std::vector<std::string> & stops_;     ///< Borrowed stop strings for this generation.
    const ChunkCallback &            on_chunk_;  ///< Borrowed synchronous receiver.
    std::string                      pending_;   ///< Bytes withheld until their UTF-8 and stop-prefix status is determined.
};

// ---------------------------------------------------------------------------
// Grammar constraint
// ---------------------------------------------------------------------------

/// Escapes the characters std::regex treats as syntax, so that a literal word can be used as a
/// grammar trigger pattern. Same character set as common/common.cpp's regex_escape.
std::string regex_escape(const std::string & text) {
    static const std::string special = ".^$|()*+?[]{}\\";

    std::string out;
    out.reserve(text.size());
    for (const char c : text) {
        if (special.find(c) != std::string::npos) {
            out += '\\';
        }
        out += c;
    }
    return out;
}

/// The string-valued parts of SamplingParams that only mean something once they have been
/// looked up in the vocabulary.
struct ResolvedTokens {
    std::unordered_set<llama_token> preserved;         ///< tokens whose special text is emitted
    std::vector<llama_token>        trigger_tokens;    ///< lazy grammar: trigger on this token
    std::vector<std::string>        trigger_patterns;  ///< lazy grammar: trigger on this regex
    std::vector<llama_token>        prefill;           ///< non-lazy grammar: tokens it consumes before sampling
};

/// RAII wrapper so the sampler chain and its grammar are freed on every path, exceptions included.
class SamplerChain {
public:
    /// Allocate and populate a sampler chain; release it on construction failure.
    SamplerChain(const llama_vocab * vocab, const SamplingParams & params, const ResolvedTokens & resolved) :
        n_vocab_(llama_vocab_n_tokens(vocab)),
        n_selectable_(params.temperature <= 0.0f ? 1
                      : params.top_k > 0         ? std::min(params.top_k, n_vocab_)
                                                 : n_vocab_) {
        llama_sampler_chain_params cparams = llama_sampler_chain_default_params();
        cparams.no_perf                    = true;

        chain_ = llama_sampler_chain_init(cparams);
        if (chain_ == nullptr) {
            throw EngineError("failed to create the sampler chain");
        }

        // A constructor that throws gets no destructor call, so the samplers are freed by hand.
        try {
            build(vocab, params, resolved);
        } catch (...) {
            free_samplers();
            throw;
        }
    }

    /// Free the owned chain, every sampler added to it, and the grammar.
    ~SamplerChain() { free_samplers(); }

    /// Copying is disabled because the samplers have a single owner.
    SamplerChain(const SamplerChain &)             = delete;
    /// Copying is disabled because the samplers have a single owner.
    SamplerChain & operator=(const SamplerChain &) = delete;

    /// Choose the next token from the logits of the last decoded position and accept it into
    /// every sampler, which advances the grammar and fires its lazy triggers.
    llama_token sample(llama_context * ctx) {
        llama_token token = choose(ctx, n_selectable_, /*apply_grammar*/ false);

        if (grammar_ != nullptr) {
            // A lazy grammar that has not triggered leaves the logit alone, so this check passes.
            llama_token_data       single       = { token, /*logit*/ 1.0f, /*p*/ 0.0f };
            llama_token_data_array single_array = { &single, /*size*/ 1, /*selected*/ -1, /*sorted*/ false };
            llama_sampler_apply(grammar_, &single_array);
            if (single.logit == -INFINITY) {
                // The grammar may refuse every token the chain alone could select, so the retry
                // starts from the whole vocabulary.
                token = choose(ctx, n_vocab_, /*apply_grammar*/ true);
            }
            llama_sampler_accept(grammar_, token);
        }

        llama_sampler_accept(chain_, token);
        return token;
    }

private:
    /// Build the probability filters; the grammar, if any, is kept apart from them.
    void build(const llama_vocab * vocab, const SamplingParams & params, const ResolvedTokens & resolved) {
        // The grammar stays outside the chain so it runs only when it has to: applying it to the
        // whole vocabulary costs more than the rest of a token's sampling put together, and the
        // token the chain chooses is usually one the grammar allows. sample() checks that token
        // alone and reruns the chain behind the grammar only when it is refused, as
        // common/sampling.cpp does. Greedy decoding picks the same token either way: the most
        // likely one the grammar allows.
        //
        // The chain below can only ever select among the highest logits: greedy the single highest,
        // top-k the k highest. sample() therefore hands it only those, n_selectable_ of them, since
        // filling and scanning a candidate for every token of a ~150k vocabulary is a measurable
        // share of decode time. The chain's own top-k still sorts them, so top-p, min-p, temp and
        // dist see the candidates they would over the whole vocabulary.
        if (!params.grammar.grammar.empty()) {
            add_grammar(vocab, params, resolved);
        }

        if (params.temperature <= 0.0f) {
            // Greedy: every other sampler is irrelevant.
            llama_sampler_chain_add(chain_, llama_sampler_init_greedy());
            return;
        }

        if (params.top_k > 0) {
            llama_sampler_chain_add(chain_, llama_sampler_init_top_k(params.top_k));
        }
        if (params.top_p < 1.0f) {
            llama_sampler_chain_add(chain_, llama_sampler_init_top_p(params.top_p, 1));
        }
        if (params.min_p > 0.0f) {
            llama_sampler_chain_add(chain_, llama_sampler_init_min_p(params.min_p, 1));
        }
        llama_sampler_chain_add(chain_, llama_sampler_init_temp(params.temperature));
        llama_sampler_chain_add(chain_,
                                llama_sampler_init_dist(params.seed ? *params.seed : LLAMA_DEFAULT_SEED));
    }

    /// Create the eager or triggered GBNF sampler; reject grammars that cannot be parsed.
    void add_grammar(const llama_vocab * vocab, const SamplingParams & params, const ResolvedTokens & resolved) {
        std::vector<const char *> patterns;
        patterns.reserve(resolved.trigger_patterns.size());
        for (const std::string & pattern : resolved.trigger_patterns) {
            patterns.push_back(pattern.c_str());
        }

        // A non-empty grammar string that does not parse (bad syntax, no root rule, left recursion)
        // makes both of these return null; they never hand back a sampler with no grammar in it.
        grammar_ =
            params.grammar.lazy
                ? llama_sampler_init_grammar_lazy_patterns(vocab, params.grammar.grammar.c_str(), /*grammar_root*/ "root",
                                                           patterns.data(), patterns.size(),
                                                           resolved.trigger_tokens.data(),
                                                           resolved.trigger_tokens.size())
                : llama_sampler_init_grammar(vocab, params.grammar.grammar.c_str(), /*grammar_root*/ "root");
        if (grammar_ == nullptr) {
            throw EngineError("failed to parse the grammar");
        }

        // The chat layer's grammars describe the whole assistant turn, whose opening the template
        // has already written into the prompt. Advancing the grammar past that text is what makes
        // the model continue the turn instead of being constrained to repeat its opening.
        try {
            for (const llama_token token : resolved.prefill) {
                llama_sampler_accept(grammar_, token);
            }
        } catch (const std::exception & e) {
            throw EngineError(std::string("the grammar does not accept the prompt's final text: ") + e.what());
        }
    }

    /// Run the chain, behind the grammar when asked, over the `count` highest logits and return its
    /// choice. A count below the vocabulary size is only correct without the grammar.
    llama_token choose(llama_context * ctx, int32_t count, bool apply_grammar) {
        const float * logits = llama_get_logits_ith(ctx, /*i*/ -1);
        const auto    vocab  = std::views::iota(0, n_vocab_) | std::views::transform([logits](llama_token id) {
            return llama_token_data{ id, logits[id], /*p*/ 0.0f };
        });

        candidates_.resize(count);
        if (count == n_vocab_) {
            std::ranges::copy(vocab, candidates_.begin());
        } else if (count == 1) {
            // max_element keeps the first of equal maxima, as llama.cpp's greedy sampler does.
            candidates_[0] = *std::ranges::max_element(vocab, {}, &llama_token_data::logit);
        } else {
            std::ranges::partial_sort_copy(vocab, candidates_, std::ranges::greater{}, &llama_token_data::logit,
                                           &llama_token_data::logit);
        }

        llama_token_data_array array = { candidates_.data(), candidates_.size(), /*selected*/ -1, /*sorted*/ false };
        if (apply_grammar) {
            llama_sampler_apply(grammar_, &array);
        }
        llama_sampler_apply(chain_, &array);
        if (array.selected < 0 || array.selected >= static_cast<int64_t>(array.size)) {
            throw EngineError("the sampler chain selected no token");
        }
        return array.data[array.selected].id;
    }

    /// Free whichever samplers exist; the constructor's failure path calls this too.
    void free_samplers() {
        if (grammar_ != nullptr) {
            llama_sampler_free(grammar_);
            grammar_ = nullptr;
        }
        if (chain_ != nullptr) {
            llama_sampler_free(chain_);
            chain_ = nullptr;
        }
    }

    const int32_t                 n_vocab_;            ///< Candidates built when the chain keeps every token, or the grammar retries.
    const int32_t                 n_selectable_;       ///< How many of the highest logits the chain can select from.
    llama_sampler *               chain_   = nullptr;  ///< Owned probability filters and final selection.
    llama_sampler *               grammar_ = nullptr;  ///< Owned grammar, or null when the request has none.
    std::vector<llama_token_data> candidates_;         ///< Reused per token to avoid a vocabulary-sized allocation.
};

/// The vector scaled to unit length, as llama.cpp's own tools normalise a pooled embedding; a
/// zero vector stays zero.
std::vector<float> l2_normalized(const float * values, int32_t n) {
    double sum = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        sum += static_cast<double>(values[i]) * values[i];
    }
    const double scale = sum > 0.0 ? 1.0 / std::sqrt(sum) : 0.0;

    std::vector<float> out(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        out[static_cast<size_t>(i)] = static_cast<float>(values[i] * scale);
    }
    return out;
}

/// Measure elapsed steady-clock time in milliseconds.
double ms_since(const std::chrono::steady_clock::time_point & t0) {
    const auto dt = std::chrono::steady_clock::now() - t0;
    return std::chrono::duration<double, std::milli>(dt).count();
}

}  // namespace

// ---------------------------------------------------------------------------
// Engine::Impl
// ---------------------------------------------------------------------------

/// Own the model and context and serialize access to generation state.
struct Engine::Impl {
    llama_model *       model = nullptr;  ///< Owned model, released after its context.
    llama_context *     ctx   = nullptr;  ///< Owned generation context.
    const llama_vocab * vocab = nullptr;  ///< Vocabulary borrowed from model.

    uint32_t n_ctx     = 0;  ///< context size actually in use
    uint32_t n_ctx_seq = 0;  ///< per-sequence capacity (what a single generation may use)
    uint32_t n_batch   = 0;  ///< largest batch llama_decode accepts

    bool     embeddings       = false;  ///< an embedding model: the context pools and outputs embeddings
    uint32_t max_embed_tokens = 0;      ///< most tokens one embed() input may have

    std::mutex context_mutex;  ///< Serializes requests sharing the context and KV cache.

    /// The tokens sequence 0 of the KV cache holds, in position order. In a generative model only
    /// decode() and clear_cache() change the cache after construction, and both keep this in step
    /// with it. An embedding model never generates, so embed() leaves it unused.
    std::vector<llama_token> cached;

    /// Release the context before the model it borrows.
    ~Impl() {
        if (ctx != nullptr) {
            llama_free(ctx);
        }
        if (model != nullptr) {
            llama_model_free(model);
        }
    }

    /// Empty the KV cache and its record.
    void clear_cache() {
        llama_memory_clear(llama_get_memory(ctx), /*data*/ true);
        cached.clear();
    }

    /// Trims the KV cache to the part of it `prompt` starts with and returns that part's length.
    /// Falls back to an empty cache when the memory refuses to drop part of a sequence (recurrent
    /// and hybrid models can) or no longer holds position 0 (a recurrent state, or a
    /// sliding-window cache that has let old tokens go), since it cannot vouch for the prefix then.
    size_t keep_prefix(const std::vector<llama_token> & prompt) {
        const size_t   keep = reusable_prefix(cached, prompt);
        llama_memory_t mem  = llama_get_memory(ctx);
        if (keep == 0 || !llama_memory_seq_rm(mem, 0, static_cast<llama_pos>(keep), -1) ||
            llama_memory_seq_pos_min(mem, 0) != 0) {
            clear_cache();
            return 0;
        }
        cached.resize(keep);
        return keep;
    }

    /// Decodes `n` tokens after the cached ones and records them. A failed decode can leave part
    /// of the batch in the cache, so it empties the cache instead: a miss on the next request is
    /// cheap, a record that disagrees with the cache would silently corrupt it.
    int32_t decode(llama_token * tokens, size_t n) {
        int32_t ret = 0;
        try {
            ret = llama_decode(ctx, llama_batch_get_one(tokens, static_cast<int32_t>(n)));
        } catch (...) {
            clear_cache();
            throw;
        }
        if (ret != 0) {
            clear_cache();
            return ret;
        }
        cached.insert(cached.end(), tokens, tokens + n);
        return 0;
    }

    /// Query the vocabulary for required capacity, then tokenize into an owned buffer.
    std::vector<int32_t> tokenize(const std::string & text, bool add_special, bool parse_special) const {
        // llama_tokenize returns -(number of tokens) when the buffer is too small.
        int32_t n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0,
                                   add_special, parse_special);
        if (n == INT32_MIN) {
            throw EngineError("tokenization overflowed int32");
        }
        if (n < 0) {
            n = -n;
        }

        static_assert(std::is_same<llama_token, int32_t>::value, "llama_token must be int32_t");

        std::vector<int32_t> tokens(static_cast<size_t>(n));
        const int32_t written = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                                               tokens.data(), n, add_special, parse_special);
        if (written < 0) {
            throw EngineError("failed to tokenize the input text");
        }
        tokens.resize(static_cast<size_t>(written));
        return tokens;
    }

    /// `special` renders the text of a control/added token instead of nothing; it is on only for the
    /// tokens the caller asked to have preserved.
    std::string token_to_piece(llama_token token, bool special) const {
        char          buf[256];
        const int32_t n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, special);
        if (n >= 0) {
            return std::string(buf, static_cast<size_t>(n));
        }

        std::vector<char> big(static_cast<size_t>(-n));
        const int32_t     n2 = llama_token_to_piece(vocab, token, big.data(),
                                                    static_cast<int32_t>(big.size()), 0, special);
        if (n2 < 0) {
            throw EngineError("failed to convert a token to text");
        }
        return std::string(big.data(), static_cast<size_t>(n2));
    }

    /// Preserved tokens and grammar triggers are given as text; only the vocabulary can turn them
    /// into token ids. Mirrors what llama-server does with the same two request fields.
    ResolvedTokens resolve(const SamplingParams & params) const {
        ResolvedTokens out;

        // A string that is not exactly one token cannot name a token, so it is simply ignored.
        for (const std::string & text : params.preserved_tokens) {
            const std::vector<int32_t> ids = tokenize(text, /*add_special*/ false, /*parse_special*/ true);
            if (ids.size() == 1) {
                out.preserved.insert(ids[0]);
            }
        }

        if (!params.grammar.grammar.empty() && !params.grammar.lazy && !params.grammar.prefill.empty()) {
            out.prefill = tokenize(params.grammar.prefill, /*add_special*/ false, /*parse_special*/ true);
            // Some tokenizers put a space in front of the first piece. That space is not in the
            // prompt, so feeding it would send the grammar down the wrong branch.
            const std::string first = out.prefill.empty() ? "" : token_to_piece(out.prefill[0], /*special*/ true);
            if (!first.empty() && std::isspace(static_cast<unsigned char>(first[0])) &&
                !std::isspace(static_cast<unsigned char>(params.grammar.prefill[0]))) {
                out.prefill.erase(out.prefill.begin());
            }
        }

        if (params.grammar.grammar.empty() || !params.grammar.lazy) {
            return out;   // triggers only mean something for a lazy grammar
        }

        out.trigger_patterns = params.grammar.trigger_patterns;
        for (const std::string & word : params.grammar.trigger_words) {
            const std::vector<int32_t> ids = tokenize(word, /*add_special*/ false, /*parse_special*/ true);
            // A word that is one token and is preserved becomes a token trigger, the cheap exact
            // form; llama-server insists on that pairing and rejects the request otherwise.
            // Everything else matches as text, which is safe even for a special token: the
            // grammar's trigger buffer is fed pieces rendered with special = true regardless.
            if (ids.size() == 1 && out.preserved.count(ids[0]) != 0) {
                out.trigger_tokens.push_back(ids[0]);
            } else {
                out.trigger_patterns.push_back(regex_escape(word));
            }
        }

        return out;
    }
};

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

size_t reusable_prefix(const std::vector<int32_t> & cached, const std::vector<int32_t> & prompt) {
    if (prompt.empty()) {
        return 0;
    }
    const size_t limit = std::min(cached.size(), prompt.size() - 1);
    size_t       n     = 0;
    while (n < limit && cached[n] == prompt[n]) {
        ++n;
    }
    return n;
}

std::vector<DeviceInfo> Engine::list_devices() {
    init_llama_once();

    std::vector<DeviceInfo> out;
    for (ggml_backend_dev_t dev : all_devices()) {
        DeviceInfo info;
        info.name        = device_name(dev);
        info.description = device_description(dev);
        info.type        = device_type_name(ggml_backend_dev_type(dev));

        size_t free_bytes  = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        info.memory_free  = free_bytes;
        info.memory_total = total_bytes;

        out.push_back(std::move(info));
    }
    return out;
}

Engine::Engine(const EngineConfig & config) : impl_(new Impl()) {
    init_llama_once();

    if (config.model_path.empty()) {
        throw EngineError("no model path was given");
    }

    const std::vector<ggml_backend_dev_t> offload = select_offload_devices(config, all_devices());

    // The NULL-terminated device array llama.cpp wants (only when the caller named devices) and
    // the padded tensor split must both stay alive until llama_model_load_from_file has returned.
    std::vector<ggml_backend_dev_t> device_arg;
    if (!config.devices.empty()) {
        device_arg = offload;
        device_arg.push_back(nullptr);
    }
    const std::vector<float> tensor_split = padded_tensor_split(config, offload.size());

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = config.n_gpu_layers;
    if (!device_arg.empty()) {
        mparams.devices = device_arg.data();
    }
    if (!tensor_split.empty()) {
        mparams.tensor_split = tensor_split.data();
    }

    impl_->model = llama_model_load_from_file(config.model_path.c_str(), mparams);
    if (impl_->model == nullptr) {
        throw EngineError("failed to load the model from '" + config.model_path +
                          "' (missing file, not a supported GGUF, or not enough memory on the "
                          "offload devices; see the errors above)");
    }

    impl_->vocab = llama_model_get_vocab(impl_->model);
    if (impl_->vocab == nullptr) {
        throw EngineError("model '" + config.model_path + "' has no vocabulary");
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = config.n_ctx;
    cparams.no_perf              = true;
    // Generation is memory-bound, so more threads than about half the hardware threads
    // stops helping (and on hybrid CPUs starts to hurt).
    const int32_t n_threads = config.n_threads > 0
                                  ? config.n_threads
                                  : std::max<int32_t>(1, static_cast<int32_t>(std::thread::hardware_concurrency() / 2));
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;

    impl_->ctx = llama_init_from_model(impl_->model, cparams);
    if (impl_->ctx == nullptr) {
        throw EngineError("failed to create a llama context for '" + config.model_path + "'");
    }

    // The context may not honour the requested values verbatim: query them back.
    impl_->n_ctx     = llama_n_ctx(impl_->ctx);
    impl_->n_ctx_seq = llama_n_ctx_seq(impl_->ctx);
    impl_->n_batch   = llama_n_batch(impl_->ctx);

    if (impl_->n_ctx_seq == 0 || impl_->n_batch == 0) {
        throw EngineError("llama context reports a zero context or batch size");
    }

    // The context takes the pooling the GGUF declares, and pooling is what makes an embedding
    // model. A reranker pools as well, but into class scores rather than a vector, which nothing
    // here serves, so it is refused rather than loaded half-usable.
    const enum llama_pooling_type pooling = llama_pooling_type(impl_->ctx);
    if (pooling == LLAMA_POOLING_TYPE_RANK) {
        throw EngineError("'" + config.model_path + "' is a reranking model, which llamad does not serve");
    }
    impl_->embeddings = pooling != LLAMA_POOLING_TYPE_NONE;
    if (impl_->embeddings) {
        llama_set_embeddings(impl_->ctx, true);

        // Pooling sees only the micro-batch it runs in, and a non-causal encoder cannot split a
        // sequence across micro-batches at all, so an input has to fit in one. Nor may it be
        // longer than the model was trained on: a BERT-style encoder has no position embedding
        // past that.
        impl_->max_embed_tokens = std::min(llama_n_ubatch(impl_->ctx), impl_->n_ctx_seq);
        const int32_t n_ctx_train = llama_model_n_ctx_train(impl_->model);
        if (n_ctx_train > 0) {
            impl_->max_embed_tokens = std::min(impl_->max_embed_tokens, static_cast<uint32_t>(n_ctx_train));
        }
    }

    // GPU backends build their compute pipelines on first use, which costs the first
    // request a second or more. Pay that here instead: a small batch and a single
    // token exercise the prompt and the generation paths. Failure is not an error.
    llama_token warm = llama_vocab_bos(impl_->vocab);
    if (warm == LLAMA_TOKEN_NULL) {
        warm = llama_vocab_eos(impl_->vocab);
    }
    if (warm != LLAMA_TOKEN_NULL) {
        std::vector<llama_token> batch(std::min<uint32_t>(32, impl_->n_batch), warm);
        llama_decode(impl_->ctx, llama_batch_get_one(batch.data(), static_cast<int32_t>(batch.size())));
        llama_decode(impl_->ctx, llama_batch_get_one(batch.data(), 1));
        llama_synchronize(impl_->ctx);
        llama_memory_clear(llama_get_memory(impl_->ctx), /*data*/ true);
    }

    report_offload_devices(config, offload);
}

Engine::~Engine() = default;

ModelInfo Engine::info() const {
    ModelInfo out;

    char          buf[256];
    const int32_t n = llama_model_desc(impl_->model, buf, sizeof(buf));
    if (n >= 0 && static_cast<size_t>(n) < sizeof(buf)) {
        out.description = std::string(buf, static_cast<size_t>(n));
    } else if (n > 0) {
        // Truncated: llama_model_desc uses snprintf, so n is the length needed.
        std::vector<char> big(static_cast<size_t>(n) + 1);
        const int32_t     n2 = llama_model_desc(impl_->model, big.data(), big.size());
        if (n2 > 0) {
            out.description = std::string(big.data(), std::min<size_t>(static_cast<size_t>(n2), big.size() - 1));
        }
    }

    out.n_params          = llama_model_n_params(impl_->model);
    out.size_bytes        = llama_model_size(impl_->model);
    out.n_ctx             = impl_->n_ctx;
    out.n_ctx_train       = static_cast<uint32_t>(std::max(0, llama_model_n_ctx_train(impl_->model)));
    out.has_chat_template = llama_model_chat_template(impl_->model, nullptr) != nullptr;
    out.serves_embeddings = impl_->embeddings;
    out.n_embd            = impl_->embeddings ? static_cast<uint32_t>(std::max(0, llama_model_n_embd_out(impl_->model)))
                                              : 0;

    return out;
}

std::vector<int32_t> Engine::tokenize(const std::string & text, bool add_special, bool parse_special) const {
    return impl_->tokenize(text, add_special, parse_special);
}

ChatTemplateInfo Engine::chat_template() const {
    ChatTemplateInfo out;

    const char * source = llama_model_chat_template(impl_->model, /*name*/ nullptr);
    if (source != nullptr) {
        out.source = source;
    }

    // A Jinja template refers to the BOS/EOS tokens by their text, so they are rendered with their
    // special text here (this is how common_chat_templates_init derives them from a model). A vocab
    // without the token contributes an empty string.
    const auto token_text = [&](llama_token token) -> std::string {
        return token == LLAMA_TOKEN_NULL ? std::string() : impl_->token_to_piece(token, /*special*/ true);
    };

    out.bos_token = token_text(llama_vocab_bos(impl_->vocab));
    out.eos_token = token_text(llama_vocab_eos(impl_->vocab));

    return out;
}

GenerateResult Engine::generate(const std::string & prompt, const SamplingParams & params,
                                const ChunkCallback & on_chunk) {
    if (impl_->embeddings) {
        throw EngineError("the model is an embedding model: it cannot generate text");
    }

    std::lock_guard<std::mutex> lock(impl_->context_mutex);

    std::vector<int32_t> tokens = impl_->tokenize(prompt, /*add_special*/ true, /*parse_special*/ true);
    if (tokens.empty()) {
        throw EngineError("the prompt tokenized to zero tokens");
    }
    if (tokens.size() >= impl_->n_ctx_seq) {
        throw EngineError("prompt is too long: " + std::to_string(tokens.size()) +
                          " tokens for a context of " + std::to_string(impl_->n_ctx_seq));
    }

    GenerateResult result;
    result.reason              = FinishReason::Length;
    result.stats.prompt_tokens = static_cast<int32_t>(tokens.size());

    // The preserved tokens decide which pieces are rendered with their special text; the triggers
    // are only used by a lazy grammar.
    const ResolvedTokens resolved = impl_->resolve(params);

    // The sampler chain is built before any decoding so a bad configuration (an unparsable grammar,
    // say) fails fast, and is freed by RAII on every path.
    SamplerChain sampler(impl_->vocab, params, resolved);

    // --- prompt ---------------------------------------------------------
    // Everything above leaves the cache alone, so a request refused there costs the next one
    // nothing.
    const size_t kept                 = impl_->keep_prefix(tokens);
    result.stats.cached_prompt_tokens = static_cast<int32_t>(kept);

    const auto t_prompt = std::chrono::steady_clock::now();
    for (size_t off = kept; off < tokens.size();) {
        const size_t  n_chunk = std::min<size_t>(impl_->n_batch, tokens.size() - off);
        const int32_t ret     = impl_->decode(tokens.data() + off, n_chunk);
        if (ret != 0) {
            throw EngineError("llama_decode failed on the prompt (code " + std::to_string(ret) + ")");
        }

        off += n_chunk;
    }
    // GPU backends return from llama_decode before the work is done; without this the
    // prompt would look free and its cost would be billed to the first generated token.
    llama_synchronize(impl_->ctx);
    result.stats.prompt_ms = ms_since(t_prompt);

    // --- generation -----------------------------------------------------
    StreamFilter filter(params.stop, on_chunk);

    const auto t_completion = std::chrono::steady_clock::now();
    bool       flush_tail   = true;

    while (true) {
        if (params.max_tokens >= 0 && result.stats.completion_tokens >= params.max_tokens) {
            result.reason = FinishReason::Length;
            break;
        }

        const llama_token token = sampler.sample(impl_->ctx);

        if (llama_vocab_is_eog(impl_->vocab, token)) {
            result.reason = FinishReason::Eog;
            break;
        }

        result.stats.completion_tokens++;

        const StreamFilter::Status status =
            filter.push(impl_->token_to_piece(token, resolved.preserved.count(token) != 0));
        if (status == StreamFilter::Status::Stopped) {
            result.reason = FinishReason::Stop;
            flush_tail    = false;
            break;
        }
        if (status == StreamFilter::Status::Cancelled) {
            result.reason = FinishReason::Cancelled;
            flush_tail    = false;
            break;
        }

        if (params.max_tokens >= 0 && result.stats.completion_tokens >= params.max_tokens) {
            result.reason = FinishReason::Length;
            break;
        }

        // Room for one more token in the context?
        if (impl_->cached.size() + 1 > impl_->n_ctx_seq) {
            result.reason = FinishReason::Length;
            break;
        }

        // Only a token the loop goes on from is decoded: the one that ends it (EOG, a stop, a
        // cancel, the budget) never reaches the cache.
        llama_token   next = token;
        const int32_t ret  = impl_->decode(&next, 1);
        if (ret != 0) {
            if (ret > 0) {
                // No KV slot: the context is full for practical purposes.
                result.reason = FinishReason::Length;
                break;
            }
            throw EngineError("llama_decode failed while generating (code " + std::to_string(ret) + ")");
        }
    }

    // A stop prefix that never completed is genuine output and must still be
    // delivered; an incomplete UTF-8 tail is dropped.
    if (flush_tail) {
        filter.flush();
    }

    result.stats.completion_ms = ms_since(t_completion);
    return result;
}

bool Engine::serves_embeddings() const {
    return impl_->embeddings;
}

EmbedResult Engine::embed(const std::vector<std::string> & inputs, const std::function<bool()> & keep_going) {
    if (!impl_->embeddings) {
        throw EngineError("the model is not an embedding model: its GGUF declares no pooling type");
    }

    // Every input is measured before any is decoded, so an over-long one fails the whole request
    // up front rather than after the others have been paid for.
    std::vector<std::vector<int32_t>> tokenized;
    tokenized.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::vector<int32_t> tokens = impl_->tokenize(inputs[i], /*add_special*/ true, /*parse_special*/ true);
        if (tokens.empty()) {
            throw EngineError("input " + std::to_string(i) + " tokenized to zero tokens");
        }
        if (tokens.size() > impl_->max_embed_tokens) {
            throw EngineError("input " + std::to_string(i) + " is too long: " + std::to_string(tokens.size()) +
                              " tokens, and one input may have at most " +
                              std::to_string(impl_->max_embed_tokens));
        }
        tokenized.push_back(std::move(tokens));
    }

    std::lock_guard<std::mutex> lock(impl_->context_mutex);

    const int32_t n_embd = llama_model_n_embd_out(impl_->model);

    EmbedResult result;
    result.embeddings.reserve(tokenized.size());
    for (std::vector<int32_t> & tokens : tokenized) {
        if (keep_going && !keep_going()) {
            break;
        }

        // One input per decode, as sequence 0 from position 0. An encoder-only model has no
        // memory, and clearing none is a no-op.
        llama_memory_clear(llama_get_memory(impl_->ctx), /*data*/ true);

        const int32_t ret =
            llama_decode(impl_->ctx, llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size())));
        if (ret != 0) {
            throw std::runtime_error("llama_decode failed on an embedding input (code " + std::to_string(ret) + ")");
        }

        const float * pooled = llama_get_embeddings_seq(impl_->ctx, /*seq_id*/ 0);
        if (pooled == nullptr) {
            throw std::runtime_error("the model produced no pooled embedding");
        }

        result.embeddings.push_back({l2_normalized(pooled, n_embd)});
        result.input_tokens += static_cast<int32_t>(tokens.size());
    }
    return result;
}

}  // namespace llamad
