/** @file
 * @brief Model, sampling and streaming contracts over the public libllama C API.
 *
 * Thin wrapper over the libllama public C API (llama.h only).
 * No gRPC or protobuf types may appear in this header or in engine.cpp.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

/// Daemon engine, chat formatting and service contracts.
namespace llamad {

/// Model loading and context configuration, fixed for the lifetime of an Engine.
struct EngineConfig {
    std::string model_path;           ///< Path to a readable GGUF model file.
    uint32_t    n_ctx        = 4096;  ///< Requested context capacity in tokens; must be positive.
    int32_t     n_gpu_layers = 99;    ///< Number of layers to offload; zero selects CPU-only execution.
    int32_t     n_threads    = 0;     ///< Inference threads; zero chooses half the hardware threads, with a minimum of one.

    /// Names of the devices to offload to, as reported by Engine::list_devices() (e.g. "Vulkan0").
    /// Empty = llama.cpp's default: every discrete GPU, or the integrated GPU if there is none.
    /// The model's layers and KV cache are split across the chosen devices.
    std::vector<std::string> devices;

    /// Share of the model given to each offload device, in device order (e.g. {3, 1}).
    /// Empty = in proportion to each device's free memory. Must not be longer than the device list.
    std::vector<float> tensor_split;
};

/// A compute device known to the llama.cpp backends compiled into this binary.
struct DeviceInfo {
    std::string name;              ///< stable identifier, usable in EngineConfig::devices
    std::string description;       ///< Human-readable device model, such as NVIDIA GeForce GTX 1080.
    std::string type;              ///< "cpu" | "gpu" | "igpu" | "accel" | "meta"
    uint64_t    memory_free  = 0;  ///< Free device memory in bytes at query time.
    uint64_t    memory_total = 0;  ///< Total device memory in bytes.
};

/// Metadata for the loaded model and its configured context.
struct ModelInfo {
    std::string description;                ///< Human-readable model architecture and quantization description.
    uint64_t    n_params          = 0;      ///< Number of model parameters.
    uint64_t    size_bytes        = 0;      ///< Size of model tensors in bytes.
    uint32_t    n_ctx             = 0;      ///< Actual context capacity configured by the daemon, in tokens.
    uint32_t    n_ctx_train       = 0;      ///< Training context length recorded by the model, in tokens.
    bool        has_chat_template = false;  ///< Whether the model stores a template; does not guarantee it parses successfully.
    uint32_t    n_embd            = 0;      ///< Length of every vector embed() returns; zero unless serves_embeddings.
    bool        serves_embeddings = false;  ///< An embedding model: embed() works, generate() does not.
};

/// The model's built-in chat template and the token text a template may reference.
struct ChatTemplateInfo {
    std::string source;     ///< empty if the model has none
    std::string bos_token;  ///< Beginning-of-sequence token text, or empty if absent.
    std::string eos_token;  ///< End-of-sequence token text, or empty if absent.
};

/// Grammar constraint for the sampler, as plain strings; the engine resolves tokens. The chat
/// layer builds one for tool calls and response schemas; the default constrains nothing.
struct GrammarSpec {
    std::string              grammar;           ///< GBNF constraint; empty permits unconstrained generation.
    bool                     lazy = false;      ///< only active once a trigger fires
    std::vector<std::string> trigger_patterns;  ///< regexes (PATTERN as-is, PATTERN_FULL anchored ^...$)
    /// Literal words; the engine resolves preserved single tokens or escapes them as regexes.
    std::vector<std::string> trigger_words;
    /// The prompt's trailing generation prefix, which a non-lazy grammar's root expects ahead of
    /// the generated text; the engine advances the grammar past it before the first token, so the
    /// model continues rather than repeats it. Only for a grammar built from the prompt, never
    /// for one a caller wrote.
    std::string              prefill;
};

/// Defaults here are the daemon defaults documented in llamad.proto.
struct SamplingParams {
    float                    temperature = 0.8f;   ///< <= 0 means greedy
    int32_t                  top_k       = 40;     ///< <= 0 disables
    float                    top_p       = 0.95f;  ///< >= 1 disables
    float                    min_p       = 0.05f;  ///< <= 0 disables
    std::optional<uint32_t>  seed;                 ///< nullopt = random
    int32_t                  max_tokens  = -1;     ///< < 0 = until context is full
    std::vector<std::string> stop;                 ///< Literal stop strings omitted from output; empty entries are ignored.
    GrammarSpec              grammar;              ///< Grammar constraint; the default leaves generation unconstrained.
    /// Special tokens whose text must be rendered into the output stream (e.g. "<tool_call>" where it is an added token).
    std::vector<std::string> preserved_tokens;
};

/// Reason a generation terminates.
enum class FinishReason {
    Eog,  ///< The model emitted an end-of-generation token.
    Length,  ///< The token budget or context capacity was reached.
    Stop,  ///< A configured stop string matched; its bytes are withheld.
    Cancelled  ///< The callback requested cancellation.
};

/// Token counts and wall-clock milliseconds for one generation, excluding queue time.
struct GenerateStats {
    int32_t prompt_tokens        = 0;  ///< Prompt length in tokens, including special tokens and cached ones.
    int32_t completion_tokens    = 0;  ///< Generated non-EOG tokens, including any withheld stop or tool markup.
    double  prompt_ms            = 0;  ///< Time decoding the uncached prompt tokens, in ms, including backend synchronization.
    double  completion_ms        = 0;  ///< Generation time in milliseconds, including streaming callback time.
    int32_t cached_prompt_tokens = 0;  ///< Leading prompt tokens reused from the KV cache instead of decoded.
};

/// Generation outcome; streamed text is delivered separately through the callback.
struct GenerateResult {
    FinishReason  reason;  ///< Reason generation ended.
    GenerateStats stats;   ///< Counts and timings reported for the generation.
};

/// One input's embedding: ModelInfo::n_embd values, L2-normalised to unit length.
struct Embedding {
    std::vector<float> values;  ///< Unit-length vector, so a dot product of two is their cosine.
};

/// The vectors for a batch of inputs and the tokens it took.
struct EmbedResult {
    std::vector<Embedding> embeddings;        ///< One per input, in input order.
    int32_t                input_tokens = 0;  ///< Tokens decoded across every input, including special tokens.
};

/// Called with each piece of generated text, in order. Return false to cancel.
/// Text held back while it could still be the start of a stop string is delivered
/// once it is known not to be; text belonging to a matched stop string is never delivered.
/// Incomplete multi-byte sequences are held back; malformed model bytes are not repaired.
using ChunkCallback = std::function<bool(const std::string & text)>;

/// Configuration, model-loading, tokenization or generation failure reported by the engine.
struct EngineError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// How many leading tokens of `prompt` a generation keeps from a KV cache holding `cached`: the
/// prefix the two share, stopping one short of the whole prompt, because sampling needs the
/// logits of the prompt's last token and only decoding it produces them.
size_t reusable_prefix(const std::vector<int32_t> & cached, const std::vector<int32_t> & prompt);

/// Owns one model and context; generate() and embed() calls serialize, and generations share one
/// KV cache. Callbacks run synchronously while that lock is held; do not reenter the engine.
/// A model whose GGUF declares a pooling type (mean, CLS or last token) is an embedding model:
/// it serves embed() and not generate(). Any other model is the reverse.
class Engine {
public:
    /// Load the model, allocate its context and warm up the backend.
    /// @param config Model path, device selection and context limits.
    /// @throws EngineError If configuration, loading or context creation fails, or the model is a
    /// reranker, which pools into scores rather than an embedding.
    explicit Engine(const EngineConfig & config);
    /// Release the context before the model; no requests may still be using this object.
    ~Engine();

    /// Copying is disabled because this object owns its state.
    Engine(const Engine &)             = delete;
    /// Copying is disabled because this object owns its state.
    Engine & operator=(const Engine &) = delete;

    /// Every device the compiled-in backends can see. Usable before any Engine exists.
    static std::vector<DeviceInfo> list_devices();

    /// Return model metadata and the actual configured context size.
    ModelInfo info() const;

    /// Whether this is an embedding model, as ModelInfo::serves_embeddings, without building the
    /// rest of info().
    bool serves_embeddings() const;

    /// Convert text to model token IDs without modifying the generation context.
    /// @param text Input bytes to tokenize.
    /// @param add_special Add the model-specific BOS/EOS tokens.
    /// @param parse_special Recognize special-token spellings in the input.
    /// @throws EngineError If tokenization fails.
    std::vector<int32_t> tokenize(const std::string & text, bool add_special, bool parse_special) const;

    /// The model's built-in chat template, for the chat layer to render with.
    ChatTemplateInfo chat_template() const;

    /// Generates from a raw prompt. Serialized internally: concurrent callers queue.
    /// The KV cache keeps what the previous call decoded, and a call decodes only the part of its
    /// prompt after the prefix it shares with that (see reusable_prefix). The cache changes how
    /// long a prompt takes, never which prompt runs; the logits can differ from a cold run's in
    /// the last bits, because the prompt is decoded in a different batch split.
    /// @param prompt Raw model input; special-token spellings are recognized.
    /// @param params Sampling, grammar, stop strings and completion budget.
    /// @param on_chunk Synchronous receiver; empty discards text, false requests cancellation.
    /// @return Finish reason and counts including generated bytes withheld from output.
    /// @throws EngineError If the prompt does not fit, the grammar is invalid or decoding fails.
    /// Exceptions from on_chunk propagate to the caller.
    GenerateResult generate(const std::string & prompt, const SamplingParams & params, const ChunkCallback & on_chunk);

    /// Embeds each input on its own, as a single sequence decoded in one piece, so the model's
    /// pooling sees all of it and nothing else. Serialized with generate().
    /// @param inputs Texts to embed; special-token spellings are recognized, as in generate().
    /// @param keep_going Asked before each input is decoded; false cancels the rest of the batch.
    /// Empty never cancels.
    /// @return One L2-normalised vector per input, in input order; after a cancel, only those for
    /// the inputs before it.
    /// @throws EngineError If the model is not an embedding model, or an input tokenizes to nothing
    /// or to more tokens than one decode can take.
    /// @throws std::runtime_error If decoding fails or yields no pooled embedding.
    EmbedResult embed(const std::vector<std::string> & inputs, const std::function<bool()> & keep_going);

private:
    /// Dependency-specific state hidden behind the public contract.
    struct Impl;
    std::unique_ptr<Impl> impl_;  ///< Sole owner of the hidden implementation.
};

}  // namespace llamad
