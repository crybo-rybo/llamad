#include <grpcpp/grpcpp.h>

#include <memory>
#include <string>

#include "llamad/v1/llamad.grpc.pb.h"

int main() {
    llamad::v1::ChatRequest request;
    request.add_messages()->set_content("hello");

    llamad::v1::ChatRequest copy;
    if (!copy.ParseFromString(request.SerializeAsString()) || copy.messages(0).content() != "hello") {
        return 1;
    }

    const auto stub = llamad::v1::Llama::NewStub(
        grpc::CreateChannel("unix:/tmp/llamad-proto-consumer.sock", grpc::InsecureChannelCredentials()));
    return stub != nullptr && copy.GetDescriptor()->file()->name() == "llamad/v1/llamad.proto" ? 0 : 1;
}
