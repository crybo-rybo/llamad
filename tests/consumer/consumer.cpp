/** @file
 * @brief An application's test of its llamad integration: the client against a fake daemon.
 *
 * Built from llamad::client and llamad::proto alone, the way a project that includes llamad
 * tests itself without a model: the fake implements the generated service on a private socket.
 */

#include <llamad/client.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include <unistd.h>

#include <grpcpp/grpcpp.h>

#include "llamad/v1/llamad.grpc.pb.h"

namespace {

class FakeDaemon final : public llamad::v1::Llama::Service {
    grpc::Status GetModelInfo(grpc::ServerContext *,
                              const llamad::v1::GetModelInfoRequest *,
                              llamad::v1::ModelInfo * info) override {
        info->set_description("fake");
        return grpc::Status::OK;
    }
};

[[=llamad::client::desc{"Say hello."}]]
std::string hello() {
    return "hello";
}

}  // namespace

int main() {
    char dir[] = "/tmp/llamad-consumer-XXXXXX";
    if (mkdtemp(dir) == nullptr) {
        std::perror("mkdtemp");
        return 1;
    }
    const std::string socket = std::string(dir) + "/llamad.sock";

    FakeDaemon          daemon;
    grpc::ServerBuilder builder;
    builder.AddListeningPort("unix:" + socket, grpc::InsecureServerCredentials());
    builder.RegisterService(&daemon);
    const std::unique_ptr<grpc::Server> server = builder.BuildAndStart();

    llamad::client::Client  client(socket);
    llamad::client::ToolSet tools;
    tools.add<^^hello>();

    const bool ok = server != nullptr &&
                    client.get_model_info({.timeout = std::chrono::seconds(10)}).description == "fake" &&
                    tools.call({"call_0", "hello", "{}"}) == "hello";

    if (server != nullptr) {
        server->Shutdown();
    }
    ::unlink(socket.c_str());
    ::rmdir(dir);

    std::puts(ok ? "ok" : "failed");
    return ok ? 0 : 1;
}
