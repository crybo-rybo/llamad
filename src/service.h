/** @file
 * @brief gRPC adapter for the engine and chat layer.
 *
 * The llamad.v1.Llama service over the engine and the chat layer. All of the daemon's gRPC and
 * protobuf knowledge lives here, in wire.{h,cpp} and in main.cpp; the engine and the chat layer
 * stay free of both.
 */

#pragma once

#include <string>
#include <utility>

#include <grpcpp/grpcpp.h>

#include "chat_format.h"
#include "engine.h"
#include "llamad/v1/llamad.grpc.pb.h"
#include "wire.h"

namespace llamad {

/// One request's id, sent to the client as `x-request-id` metadata, and its log lines; defined in
/// service.cpp.
class RequestLog;

/// Stateless RPC adapter; serializes generation through Engine and owns no model or tools.
class LlamaService final : public v1::Llama::Service {
public:
    /// `chat_format` may be null: a model whose template is missing or unparsable still serves
    /// Generate, Tokenize and GetModelInfo, and Chat fails with FAILED_PRECONDITION and
    /// `chat_unavailable_reason`. An embedding model serves Embed instead of Generate and Chat.
    /// Both `engine` and `chat_format` must outlive the service. `daemon` is what GetModelInfo
    /// reports about the daemon beside the model.
    LlamaService(Engine & engine, wire::DaemonInfo daemon, const ChatFormat * chat_format,
                 std::string chat_unavailable_reason = "the model has no built-in chat template")
        : engine_(engine),
          daemon_(std::move(daemon)),
          chat_format_(chat_format),
          chat_unavailable_reason_(std::move(chat_unavailable_reason)) {}

    /// Return model and daemon metadata without applying a template.
    grpc::Status GetModelInfo(grpc::ServerContext * context,
                              const v1::GetModelInfoRequest * request,
                              v1::ModelInfo * response) override;

    /// Tokenize request text with the caller's special-token flags.
    grpc::Status Tokenize(grpc::ServerContext * context,
                          const v1::TokenizeRequest * request,
                          v1::TokenizeResponse * response) override;

    /// Stream a raw completion and one final result while the client remains connected.
    /// @returns FAILED_PRECONDITION on an embedding model, OUT_OF_RANGE for a prompt that does not
    /// fit the context, INVALID_ARGUMENT for other caller errors, CANCELLED if the client goes
    /// before the finish chunk, INTERNAL for other failures, or OK.
    grpc::Status Generate(grpc::ServerContext * context,
                          const v1::GenerateRequest * request,
                          grpc::ServerWriter<v1::GenerateChunk> * writer) override;

    /// Render history and tools, stream visible text and finalize atomic tool calls.
    /// @returns FAILED_PRECONDITION without a usable template, OUT_OF_RANGE for a rendered prompt
    /// that does not fit the context, INVALID_ARGUMENT for other caller errors, CANCELLED if the
    /// client goes before the finish chunk, INTERNAL for other failures, or OK after generation.
    grpc::Status Chat(grpc::ServerContext * context,
                      const v1::ChatRequest * request,
                      grpc::ServerWriter<v1::GenerateChunk> * writer) override;

    /// Embed each input and return one L2-normalised vector per input, in order.
    /// @returns FAILED_PRECONDITION unless the model is an embedding model, OUT_OF_RANGE for an
    /// input that does not fit, INVALID_ARGUMENT for no inputs or other caller errors, CANCELLED
    /// if the client goes before the last input, INTERNAL for other failures, or OK.
    grpc::Status Embed(grpc::ServerContext * context,
                       const v1::EmbedRequest * request,
                       v1::EmbedResponse * response) override;

private:
    /// Shared body of Generate and Chat: run the engine, stream text chunks and, after a second
    /// of silence, progress chunks, then write the one `finish` chunk carrying the finish reason,
    /// stats and any tool calls. A client that goes stops the engine at its next step and gets
    /// no `finish`; the status is then CANCELLED.
    /// `stream` is Chat's parser, which decides what text the client sees and parses the tool
    /// calls once generation is over; Generate passes null. `log` gets the request's one line.
    /// What the engine throws propagates.
    grpc::Status stream_generation(RequestLog & log,
                                   grpc::ServerContext * context,
                                   const std::string & prompt,
                                   const SamplingParams & params,
                                   grpc::ServerWriter<v1::GenerateChunk> * writer,
                                   ChatFormat::Stream * stream);

    Engine &               engine_;                   ///< Borrowed engine; outlives the service.
    const wire::DaemonInfo daemon_;                   ///< Version GetModelInfo reports.
    const ChatFormat *     chat_format_;              ///< Borrowed immutable formatter, or null when Chat is unavailable.
    const std::string      chat_unavailable_reason_;  ///< Diagnostic returned with FAILED_PRECONDITION.
};

}  // namespace llamad
