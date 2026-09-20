#pragma once

// Thin wrapper over the libllama public C API (llama.h only).
// No gRPC or protobuf types may appear in this header or in engine.cpp.

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace llamad {

struct EngineConfig {
    std::string model_path;
    uint32_t    n_ctx        = 4096;
    int32_t     n_gpu_layers = 99;
    int32_t     n_threads    = 0;   // 0 = half the hardware threads (llama.cpp's own default is a fixed 4)
};

struct ModelInfo {
    std::string description;
    uint64_t    n_params          = 0;
    uint64_t    size_bytes        = 0;
    uint32_t    n_ctx             = 0;
    uint32_t    n_ctx_train       = 0;
    bool        has_chat_template = false;
};

// Defaults here are the daemon defaults documented in llamad.proto.
struct SamplingParams {
    float                    temperature = 0.8f;   // <= 0 means greedy
    int32_t                  top_k       = 40;     // <= 0 disables
    float                    top_p       = 0.95f;  // >= 1 disables
    float                    min_p       = 0.05f;  // <= 0 disables
    std::optional<uint32_t>  seed;                 // nullopt = random
    int32_t                  max_tokens  = -1;     // < 0 = until context is full
    std::vector<std::string> stop;
};

struct ChatMessage {
    std::string role;
    std::string content;
};

enum class FinishReason { Eog, Length, Stop, Cancelled };

struct GenerateStats {
    int32_t prompt_tokens     = 0;
    int32_t completion_tokens = 0;
    double  prompt_ms         = 0;
    double  completion_ms     = 0;
};

struct GenerateResult {
    FinishReason  reason;
    GenerateStats stats;
};

// Called with each piece of generated text, in order. Return false to cancel.
// Text held back while it could still be the start of a stop string is delivered
// once it is known not to be; text belonging to a matched stop string is never delivered.
// Pieces are always valid UTF-8 (incomplete multi-byte sequences are held back).
using ChunkCallback = std::function<bool(const std::string & text)>;

// Thrown for caller errors (prompt longer than the context, model has no chat template, ...).
struct EngineError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class Engine {
public:
    // Throws EngineError if the model cannot be loaded.
    explicit Engine(const EngineConfig & config);
    ~Engine();

    Engine(const Engine &)             = delete;
    Engine & operator=(const Engine &) = delete;

    ModelInfo info() const;

    std::vector<int32_t> tokenize(const std::string & text, bool add_special, bool parse_special) const;

    // Applies the model's built-in chat template with the assistant turn opened.
    std::string apply_chat_template(const std::vector<ChatMessage> & messages) const;

    // Generates from a raw prompt. Serialized internally: concurrent callers queue.
    // Each call starts from an empty KV cache.
    GenerateResult generate(const std::string & prompt, const SamplingParams & params, const ChunkCallback & on_chunk);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace llamad
