// llamad: hosts one llama.cpp model and serves it over gRPC on a Unix domain
// socket. There is no TCP listener, by design: access control is the socket's
// file permissions.

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include <pthread.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "engine.h"
#include "service.h"

namespace {

void print_usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s --model PATH [options]\n"
                 "\n"
                 "  --model PATH     GGUF model to load (required)\n"
                 "  --socket PATH    unix socket to listen on\n"
                 "                   (default: $XDG_RUNTIME_DIR/llamad.sock, else /tmp/llamad-<uid>.sock)\n"
                 "  --ctx N          context size (default 4096)\n"
                 "  --ngl N          layers to offload to the GPU (default 99)\n"
                 "  --threads N      threads for inference, 0 = auto, half the hardware threads (default 0)\n"
                 "  --help           show this message\n",
                 argv0);
}

std::string default_socket_path() {
    const char * runtime_dir = std::getenv("XDG_RUNTIME_DIR");
    if (runtime_dir != nullptr && runtime_dir[0] != '\0') {
        return std::string(runtime_dir) + "/llamad.sock";
    }
    return "/tmp/llamad-" + std::to_string(static_cast<unsigned>(getuid())) + ".sock";
}

// True if something is listening on `path` right now. A plain connect(2) is the
// cheapest possible liveness probe and needs no gRPC machinery.
bool socket_is_live(const std::string & path) {
    const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        return false;
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path.c_str());
    const bool live = ::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0;
    ::close(fd);
    return live;
}

// Returns false (after printing why) if the path cannot be used.
bool prepare_socket_path(const std::string & path) {
    // sockaddr_un::sun_path is 108 bytes including the NUL, so 107 usable.
    if (path.size() >= sizeof(sockaddr_un{}.sun_path)) {
        std::fprintf(stderr,
                     "error: socket path is %zu bytes, the maximum is %zu: %s\n",
                     path.size(), sizeof(sockaddr_un{}.sun_path) - 1, path.c_str());
        return false;
    }

    struct stat st {};
    if (::lstat(path.c_str(), &st) != 0) {
        if (errno == ENOENT) {
            return true;  // nothing in the way
        }
        std::fprintf(stderr, "error: cannot stat %s: %s\n", path.c_str(), std::strerror(errno));
        return false;
    }

    // Never unlink something that is not ours to remove.
    if (!S_ISSOCK(st.st_mode)) {
        std::fprintf(stderr, "error: %s exists and is not a socket; refusing to remove it\n", path.c_str());
        return false;
    }
    if (socket_is_live(path)) {
        std::fprintf(stderr, "error: another llamad is already listening on %s\n", path.c_str());
        return false;
    }
    std::fprintf(stderr, "[llamad] removing stale socket %s\n", path.c_str());
    if (::unlink(path.c_str()) != 0) {
        std::fprintf(stderr, "error: cannot remove stale socket %s: %s\n", path.c_str(), std::strerror(errno));
        return false;
    }
    return true;
}

bool parse_int(const char * text, long * out) {
    char * end = nullptr;
    errno      = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    *out = value;
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string socket_path = default_socket_path();
    long        n_ctx       = 4096;
    long        n_gpu_layers = 99;
    long        n_threads   = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--model") {
            model_path = next("--model");
        } else if (arg == "--socket") {
            socket_path = next("--socket");
        } else if (arg == "--ctx") {
            if (!parse_int(next("--ctx"), &n_ctx) || n_ctx <= 0) {
                std::fprintf(stderr, "error: --ctx needs a positive integer\n");
                return 2;
            }
        } else if (arg == "--ngl") {
            if (!parse_int(next("--ngl"), &n_gpu_layers)) {
                std::fprintf(stderr, "error: --ngl needs an integer\n");
                return 2;
            }
        } else if (arg == "--threads") {
            if (!parse_int(next("--threads"), &n_threads) || n_threads < 0) {
                std::fprintf(stderr, "error: --threads needs a non-negative integer\n");
                return 2;
            }
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    if (model_path.empty()) {
        std::fprintf(stderr, "error: --model is required\n");
        print_usage(argv[0]);
        return 2;
    }

    // Block the shutdown signals here, before any other thread exists, so every
    // thread gRPC and llama.cpp later create inherits the mask and only our
    // dedicated sigwait thread ever sees them. That thread runs ordinary code,
    // so none of the async-signal-safety restrictions of a handler apply.
    sigset_t shutdown_signals;
    sigemptyset(&shutdown_signals);
    sigaddset(&shutdown_signals, SIGINT);
    sigaddset(&shutdown_signals, SIGTERM);
    if (pthread_sigmask(SIG_BLOCK, &shutdown_signals, nullptr) != 0) {
        std::fprintf(stderr, "error: cannot block SIGINT/SIGTERM\n");
        return 1;
    }
    // A client that dies mid-stream must not take the daemon down with it.
    ::signal(SIGPIPE, SIG_IGN);

    if (!prepare_socket_path(socket_path)) {
        return 1;
    }

    // Load the model before listening: a client that can connect is a client
    // that can generate, never one waiting on a half-ready daemon.
    std::unique_ptr<llamad::Engine> engine;
    try {
        llamad::EngineConfig config;
        config.model_path   = model_path;
        config.n_ctx        = static_cast<uint32_t>(n_ctx);
        config.n_gpu_layers = static_cast<int32_t>(n_gpu_layers);
        config.n_threads    = static_cast<int32_t>(n_threads);
        std::fprintf(stderr, "[llamad] loading %s ...\n", model_path.c_str());
        engine = std::make_unique<llamad::Engine>(config);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    llamad::LlamaService service(*engine);

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    grpc::ServerBuilder builder;
    builder.AddListeningPort("unix:" + socket_path, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    // Set the mode with umask rather than a chmod after bind(): between bind()
    // and chmod() the socket would briefly be connectable by any local user, and
    // a connection accepted in that window stays open. umask closes that window
    // because the permission bits are applied as the inode is created.
    const mode_t saved_umask = ::umask(0177);  // 0666 & ~0177 == 0600
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    ::umask(saved_umask);

    if (!server) {
        std::fprintf(stderr, "error: failed to listen on unix:%s\n", socket_path.c_str());
        return 1;
    }

    std::thread signal_thread([&] {
        int signo = 0;
        while (sigwait(&shutdown_signals, &signo) != 0) {
            // interrupted; retry
        }
        std::fprintf(stderr, "[llamad] caught %s, shutting down\n", ::strsignal(signo));
        // A short deadline lets in-flight streams finish their current chunk;
        // after it gRPC cancels them, the handlers see IsCancelled() and return.
        server->Shutdown(std::chrono::system_clock::now() + std::chrono::milliseconds(500));
    });

    const llamad::ModelInfo info = engine->info();
    std::fprintf(stderr, "[llamad] model: %s (n_ctx=%u)\n", info.description.c_str(), info.n_ctx);
    std::fprintf(stderr, "[llamad] listening on unix:%s\n", socket_path.c_str());
    std::fflush(stderr);

    server->Wait();
    signal_thread.join();

    // Only remove the socket if it is still the one we created.
    struct stat st {};
    if (::lstat(socket_path.c_str(), &st) == 0 && S_ISSOCK(st.st_mode)) {
        ::unlink(socket_path.c_str());
    }
    std::fprintf(stderr, "[llamad] stopped\n");
    return 0;
}
