#pragma once

// gRPC service implementation: translates llamad.v1 messages to and from the
// engine's plain-C++ types. All the gRPC/protobuf knowledge of the daemon lives
// here and in main.cpp; the engine stays free of both.

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "chat_format.h"
#include "engine.h"
#include "llamad/v1/llamad.grpc.pb.h"

namespace llamad {

class LlamaService final : public v1::Llama::Service {
public:
    // `chat_format` may be null: a model whose template is missing or unparsable still serves
    // Generate, Tokenize and GetModelInfo, and Chat fails with FAILED_PRECONDITION and
    // `chat_unavailable_reason`. Both `engine` and `chat_format` must outlive the service.
    LlamaService(Engine & engine, const ChatFormat * chat_format,
                 std::string chat_unavailable_reason = "the model has no built-in chat template")
        : engine_(engine),
          chat_format_(chat_format),
          chat_unavailable_reason_(std::move(chat_unavailable_reason)) {}

    grpc::Status GetModelInfo(grpc::ServerContext * context,
                              const v1::GetModelInfoRequest * request,
                              v1::ModelInfo * response) override;

    grpc::Status Tokenize(grpc::ServerContext * context,
                          const v1::TokenizeRequest * request,
                          v1::TokenizeResponse * response) override;

    grpc::Status Generate(grpc::ServerContext * context,
                          const v1::GenerateRequest * request,
                          grpc::ServerWriter<v1::GenerateChunk> * writer) override;

    grpc::Status Chat(grpc::ServerContext * context,
                      const v1::ChatRequest * request,
                      grpc::ServerWriter<v1::GenerateChunk> * writer) override;

private:
    // What Chat adds to the plain Generate stream. Generate passes none of it.
    //
    // `filter` sees every piece of generated text before it goes on the wire and returns what the
    // client should actually see, which is "" while the text is part of a tool call. `finish` runs
    // once generation is over (and only while the client is still there): it hands back any visible
    // text the filter held on to and the tool calls it parsed.
    struct StreamHooks {
        std::function<std::string(const std::string & text)>                  filter;
        std::function<void(std::string * tail, std::vector<ToolCall> * calls)> finish;
    };

    // Shared body of Generate and Chat: run the engine, stream text chunks, then
    // write the single final chunk carrying finish_reason, stats and any tool calls.
    // `hooks` may be null (Generate).
    grpc::Status stream_generation(const char * rpc_name,
                                   grpc::ServerContext * context,
                                   const std::string & prompt,
                                   const SamplingParams & params,
                                   grpc::ServerWriter<v1::GenerateChunk> * writer,
                                   const StreamHooks * hooks);

    Engine &           engine_;
    const ChatFormat * chat_format_;
    const std::string  chat_unavailable_reason_;
};

}  // namespace llamad
