#include <cstdlib>
#include <iostream>
#include <string>

#include <unistd.h>

#include <grpcpp/grpcpp.h>

#include "flags.h"
#include "playground.h"

int main(int argc, char ** argv) {
    const char * runtime_dir = std::getenv("XDG_RUNTIME_DIR");
    std::string socket = runtime_dir != nullptr && runtime_dir[0] != '\0'
        ? std::string(runtime_dir) + "/llamad.sock"
        : "/tmp/llamad-" + std::to_string(static_cast<unsigned>(getuid())) + ".sock";

    try {
        for (llamad::cli::Args args(argc, argv); args.next();) {
            if (args.is("--socket")) {
                socket = args.value();
                if (socket.empty()) {
                    throw llamad::cli::FlagError("--socket needs a nonempty path");
                }
            } else if (args.is("--help")) {
                std::cout << "usage: " << argv[0] << " [--socket PATH]\n\n"
                          << "Connect to a running llamad daemon. No model is loaded by this client.\n"
                          << "The socket default matches llamad: $XDG_RUNTIME_DIR/llamad.sock,\n"
                          << "or /tmp/llamad-<uid>.sock.\n\n"
                          << "Type /help for commands. Ctrl+D or /quit exits.\n";
                return 0;
            } else {
                throw args.unknown();
            }
        }
    } catch (const llamad::cli::FlagError & e) {
        std::cerr << "error: " << e.what() << "\nUse --help for usage.\n";
        return 2;
    }

    auto stub = llamad::v1::Llama::NewStub(grpc::CreateChannel("unix:" + socket, grpc::InsecureChannelCredentials()));
    return run_playground(*stub, std::cin, std::cout);
}
