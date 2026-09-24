/** @file
 * @brief gRPC adapter for the engine and chat layer.
 *
 * gRPC service implementation: translates llamad.v1 messages to and from the
 * engine's plain-C++ types. All the gRPC/protobuf knowledge of the daemon lives
 * here and in main.cpp; the engine stays free of both.
 */

#pragma once

#include <string>
#include <utility>

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
    /// `chat_unavailable_reason`. An embedding model serves Embed instead of Generate and Chat.
    /// Both `engine` and `chat_format` must outlive the service.
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

    /// Embed each input and return one L2-normalised vector per input, in order.
    /// @returns FAILED_PRECONDITION unless the model is an embedding model, INVALID_ARGUMENT for
    /// no inputs or an input that does not fit, CANCELLED if the client goes before the last
    /// input, INTERNAL for other failures, or OK.
    grpc::Status Embed(grpc::ServerContext * context,
                       const v1::EmbedRequest * request,
                       v1::EmbedResponse * response) override;

private:
    /// Shared body of Generate and Chat: run the engine, stream text chunks, then
    /// write the single final chunk carrying finish_reason, stats and any tool calls.
    /// `stream` is Chat's parser, which decides what text the client sees and parses the tool
    /// calls once generation is over; Generate passes null. What the engine throws propagates.
    grpc::Status stream_generation(const char * rpc_name,
                                   grpc::ServerContext * context,
                                   const std::string & prompt,
                                   const SamplingParams & params,
                                   grpc::ServerWriter<v1::GenerateChunk> * writer,
                                   ChatFormat::Stream * stream);

    Engine &           engine_;                   ///< Borrowed engine; outlives the service.
    const ChatFormat * chat_format_;              ///< Borrowed immutable formatter, or null when Chat is unavailable.
    const std::string  chat_unavailable_reason_;  ///< Diagnostic returned with FAILED_PRECONDITION.
};

}  // namespace llamad
