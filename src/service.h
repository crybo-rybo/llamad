#pragma once

// gRPC service implementation: translates llamad.v1 messages to and from the
// engine's plain-C++ types. All the gRPC/protobuf knowledge of the daemon lives
// here and in main.cpp; the engine stays free of both.

#include <string>

#include <grpcpp/grpcpp.h>

#include "engine.h"
#include "llamad/v1/llamad.grpc.pb.h"

namespace llamad {

class LlamaService final : public v1::Llama::Service {
public:
    explicit LlamaService(Engine & engine) : engine_(engine) {}

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
    // Shared body of Generate and Chat: run the engine, stream text chunks, then
    // write the single final chunk carrying finish_reason and stats.
    grpc::Status stream_generation(const char * rpc_name,
                                   grpc::ServerContext * context,
                                   const std::string & prompt,
                                   const v1::SamplingParams & sampling,
                                   grpc::ServerWriter<v1::GenerateChunk> * writer);

    Engine & engine_;
};

}  // namespace llamad
