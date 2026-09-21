/** @file
 * @brief gRPC adapter for the engine and chat layer.
 *
 * gRPC service implementation: translates llamad.v1 messages to and from the
 * engine's plain-C++ types. All the gRPC/protobuf knowledge of the daemon lives
 * here and in main.cpp; the engine stays free of both.
 */

#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "chat_format.h"
#include "engine.h"
#include "llamad/v1/llamad.grpc.pb.h"

namespace llamad {

/// Stateless RPC adapter; serializes generation through Engine and owns no model or tools.
class LlamaService final : public v1::Llama::Service {
public:
    /// `chat_format` may be null: a model whose template is missing or unparsable still serves
    /// Generate, Tokenize and GetModelInfo, and Chat fails with FAILED_PRECONDITION and
    /// `chat_unavailable_reason`. Both `engine` and `chat_format` must outlive the service.
    LlamaService(Engine & engine, const ChatFormat * chat_format,
                 std::string chat_unavailable_reason = "the model has no built-in chat template")
        : engine_(engine),
          chat_format_(chat_format),
          chat_unavailable_reason_(std::move(chat_unavailable_reason)) {}

    /// Return model metadata without applying a template.
    grpc::Status GetModelInfo(grpc::ServerContext * context,
                              const v1::GetModelInfoRequest * request,
                              v1::ModelInfo * response) override;

    /// Tokenize request text with the caller's special-token flags.
    grpc::Status Tokenize(grpc::ServerContext * context,
                          const v1::TokenizeRequest * request,
                          v1::TokenizeResponse * response) override;

    /// Stream a raw completion and one final result while the client remains connected.
    grpc::Status Generate(grpc::ServerContext * context,
                          const v1::GenerateRequest * request,
                          grpc::ServerWriter<v1::GenerateChunk> * writer) override;

    /// Render history and tools, stream visible text and finalize atomic tool calls.
    /// @returns FAILED_PRECONDITION without a usable template, INVALID_ARGUMENT for caller
    /// errors, INTERNAL for other failures, or OK after generation.
    grpc::Status Chat(grpc::ServerContext * context,
                      const v1::ChatRequest * request,
                      grpc::ServerWriter<v1::GenerateChunk> * writer) override;

private:
    /// What Chat adds to the plain Generate stream. Generate passes none of it.
    ///
    /// `filter` sees every piece of generated text before it goes on the wire and returns what the
    /// client should actually see, which is "" while the text is part of a tool call. `finish` runs
    /// once generation is over (and only while the client is still there): it hands back any visible
    /// text the filter held on to and the tool calls it parsed.
    struct StreamHooks {
        /// Convert raw engine text into user-visible chat content.
        std::function<std::string(const std::string & text)>                  filter;
        /// Collect the held-back visible tail and complete tool calls after generation.
        std::function<void(std::string * tail, std::vector<ToolCall> * calls)> finish;
    };

    /// Shared body of Generate and Chat: run the engine, stream text chunks, then
    /// write the single final chunk carrying finish_reason, stats and any tool calls.
    /// `hooks` may be null (Generate).
    grpc::Status stream_generation(const char * rpc_name,
                                   grpc::ServerContext * context,
                                   const std::string & prompt,
                                   const SamplingParams & params,
                                   grpc::ServerWriter<v1::GenerateChunk> * writer,
                                   const StreamHooks * hooks);

    Engine &           engine_;                   ///< Borrowed engine; outlives the service.
    const ChatFormat * chat_format_;              ///< Borrowed immutable formatter, or null when Chat is unavailable.
    const std::string  chat_unavailable_reason_;  ///< Diagnostic returned with FAILED_PRECONDITION.
};

}  // namespace llamad
