#include "engine.h"

#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace llamad {

namespace {

// ---------------------------------------------------------------------------
// llama.cpp / ggml logging: only warnings and errors reach stderr.
// ---------------------------------------------------------------------------

// GGML_LOG_LEVEL_CONT means "continuation of the previous log line", so its own
// level carries no information: remember the level of the last real line and let
// continuations inherit it.
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

void init_llama_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(log_callback, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
    });
}

// ---------------------------------------------------------------------------
// UTF-8 / stop-string aware streaming
// ---------------------------------------------------------------------------

// Number of trailing bytes of `s` that form an incomplete UTF-8 sequence.
// Returns 0 when the buffer ends on a complete (or simply invalid) sequence,
// in which case nothing needs to be held back.
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

// Length of the longest suffix of `pending` that is a proper prefix of one of
// the stop strings. That suffix must be held back: it may yet become a match.
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

// Buffers generated text and hands it to the callback as soon as it is known to
// be deliverable: never an incomplete UTF-8 sequence, and never bytes that could
// still turn out to be the start of a stop string.
class StreamFilter {
public:
    enum class Status { Continue, Stopped, Cancelled };

    StreamFilter(const std::vector<std::string> & stops, const ChunkCallback & on_chunk) :
        stops_(stops), on_chunk_(on_chunk) {}

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

    // Called when generation ended on its own: a held-back stop prefix that never
    // completed is real output and must be delivered; an incomplete UTF-8 tail is
    // dropped because it can never become a valid character.
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
    bool deliver(const std::string & text) {
        if (text.empty()) {
            return true;
        }
        if (!on_chunk_) {
            return true;
        }
        return on_chunk_(text);
    }

    const std::vector<std::string> & stops_;
    const ChunkCallback &            on_chunk_;
    std::string                      pending_;
};

// RAII wrapper so the sampler chain is freed on every path, exceptions included.
class SamplerChain {
public:
    explicit SamplerChain(const SamplingParams & params) {
        llama_sampler_chain_params cparams = llama_sampler_chain_default_params();
        cparams.no_perf                    = true;

        chain_ = llama_sampler_chain_init(cparams);
        if (chain_ == nullptr) {
            throw EngineError("failed to create the sampler chain");
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

    ~SamplerChain() {
        if (chain_ != nullptr) {
            llama_sampler_free(chain_);
        }
    }

    SamplerChain(const SamplerChain &)             = delete;
    SamplerChain & operator=(const SamplerChain &) = delete;

    llama_sampler * get() const { return chain_; }

private:
    llama_sampler * chain_ = nullptr;
};

double ms_since(const std::chrono::steady_clock::time_point & t0) {
    const auto dt = std::chrono::steady_clock::now() - t0;
    return std::chrono::duration<double, std::milli>(dt).count();
}

}  // namespace

// ---------------------------------------------------------------------------
// Engine::Impl
// ---------------------------------------------------------------------------

struct Engine::Impl {
    llama_model *       model = nullptr;
    llama_context *     ctx   = nullptr;
    const llama_vocab * vocab = nullptr;

    uint32_t n_ctx     = 0;  // context size actually in use
    uint32_t n_ctx_seq = 0;  // per-sequence capacity (what a single generation may use)
    uint32_t n_batch   = 0;  // largest batch llama_decode accepts

    std::mutex generate_mutex;

    ~Impl() {
        if (ctx != nullptr) {
            llama_free(ctx);
        }
        if (model != nullptr) {
            llama_model_free(model);
        }
    }

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

    std::string token_to_piece(llama_token token) const {
        char          buf[256];
        const int32_t n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, /*special*/ false);
        if (n >= 0) {
            return std::string(buf, static_cast<size_t>(n));
        }

        std::vector<char> big(static_cast<size_t>(-n));
        const int32_t     n2 = llama_token_to_piece(vocab, token, big.data(),
                                                    static_cast<int32_t>(big.size()), 0, /*special*/ false);
        if (n2 < 0) {
            throw EngineError("failed to convert a token to text");
        }
        return std::string(big.data(), static_cast<size_t>(n2));
    }
};

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

Engine::Engine(const EngineConfig & config) : impl_(new Impl()) {
    init_llama_once();

    if (config.model_path.empty()) {
        throw EngineError("no model path was given");
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = config.n_gpu_layers;

    impl_->model = llama_model_load_from_file(config.model_path.c_str(), mparams);
    if (impl_->model == nullptr) {
        throw EngineError("failed to load the model from '" + config.model_path +
                          "' (missing file, or not a supported GGUF)");
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

    return out;
}

std::vector<int32_t> Engine::tokenize(const std::string & text, bool add_special, bool parse_special) const {
    return impl_->tokenize(text, add_special, parse_special);
}

std::string Engine::apply_chat_template(const std::vector<ChatMessage> & messages) const {
    const char * tmpl = llama_model_chat_template(impl_->model, /*name*/ nullptr);
    if (tmpl == nullptr) {
        throw EngineError("the model has no built-in chat template");
    }

    std::vector<llama_chat_message> chat;
    chat.reserve(messages.size());
    size_t total = 0;
    for (const ChatMessage & m : messages) {
        chat.push_back(llama_chat_message{m.role.c_str(), m.content.c_str()});
        total += m.role.size() + m.content.size();
    }

    // Recommended starting size is 2x the total message length.
    std::vector<char> buf(std::max<size_t>(total * 2, 512));

    int32_t n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), /*add_ass*/ true, buf.data(),
                                          static_cast<int32_t>(buf.size()));
    if (n > static_cast<int32_t>(buf.size())) {
        buf.resize(static_cast<size_t>(n));
        n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), /*add_ass*/ true, buf.data(),
                                      static_cast<int32_t>(buf.size()));
    }
    if (n < 0) {
        throw EngineError("the model's chat template is not supported by llama_chat_apply_template");
    }

    return std::string(buf.data(), static_cast<size_t>(std::min<size_t>(static_cast<size_t>(n), buf.size())));
}

GenerateResult Engine::generate(const std::string & prompt, const SamplingParams & params,
                                const ChunkCallback & on_chunk) {
    std::lock_guard<std::mutex> lock(impl_->generate_mutex);

    llama_memory_t mem = llama_get_memory(impl_->ctx);
    llama_memory_clear(mem, /*data*/ true);

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

    // The sampler chain is built before any decoding so a bad configuration fails
    // fast, and is freed by RAII on every path.
    SamplerChain sampler(params);

    // --- prompt ---------------------------------------------------------
    const auto t_prompt = std::chrono::steady_clock::now();
    for (size_t off = 0; off < tokens.size();) {
        const size_t n_chunk = std::min<size_t>(impl_->n_batch, tokens.size() - off);

        llama_batch   batch = llama_batch_get_one(tokens.data() + off, static_cast<int32_t>(n_chunk));
        const int32_t ret   = llama_decode(impl_->ctx, batch);
        if (ret != 0) {
            throw EngineError("llama_decode failed on the prompt (code " + std::to_string(ret) + ")");
        }

        off += n_chunk;
    }
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

        const llama_token token = llama_sampler_sample(sampler.get(), impl_->ctx, -1);

        if (llama_vocab_is_eog(impl_->vocab, token)) {
            result.reason = FinishReason::Eog;
            break;
        }

        result.stats.completion_tokens++;

        const StreamFilter::Status status = filter.push(impl_->token_to_piece(token));
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
        const int32_t n_used = llama_memory_seq_pos_max(mem, 0) + 1;
        if (n_used + 1 > static_cast<int32_t>(impl_->n_ctx_seq)) {
            result.reason = FinishReason::Length;
            break;
        }

        llama_token   next  = token;
        llama_batch   batch = llama_batch_get_one(&next, 1);
        const int32_t ret   = llama_decode(impl_->ctx, batch);
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

}  // namespace llamad
