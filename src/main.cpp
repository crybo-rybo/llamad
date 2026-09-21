/** @file
 * @brief Daemon entry point, private Unix socket lifecycle and signal-driven shutdown.
 *
 * llamad: hosts one llama.cpp model and serves it over gRPC on a Unix domain
 * socket. There is no TCP listener, by design: access control is the socket's
 * file permissions.
 */

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "chat_format.h"
#include "engine.h"
#include "engine_flags.h"
#include "flags.h"
#include "service.h"

namespace {

using llamad::cli::help;

/// Options specific to this executable; shared flags are composed separately.
struct Options {
    [[=help{"PATH", "GGUF model to load (required)"}]]
    std::string model;  ///< GGUF model to load (required).

    [[=help{"PATH", "unix socket to listen on\n"
                    "(default: $XDG_RUNTIME_DIR/llamad.sock, else /tmp/llamad-<uid>.sock)"}]]
    std::optional<std::string> socket;  ///< Unix socket path; empty uses the per-user default.
};

/// Print usage and reflected flag descriptions to stderr.
void print_usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s --model PATH [options]\n\n", argv0);
    llamad::cli::print_flags(stderr, Options{}, llamad::EngineFlags{}, llamad::cli::HelpFlag{});
}

/// Format a byte count in binary gibibytes.
std::string format_gib(uint64_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f GiB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

/// Print backend devices and memory without loading a model.
void print_device_table() {
    const std::vector<llamad::DeviceInfo> devices = llamad::Engine::list_devices();
    if (devices.empty()) {
        std::printf("no devices (this build has no ggml backend registered)\n");
        return;
    }

    size_t w_name = std::strlen("NAME");
    size_t w_type = std::strlen("TYPE");
    for (const llamad::DeviceInfo & device : devices) {
        w_name = std::max(w_name, device.name.size());
        w_type = std::max(w_type, device.type.size());
    }

    std::printf("%-*s  %-*s  %9s  %9s  %s\n", static_cast<int>(w_name), "NAME", static_cast<int>(w_type),
                "TYPE", "FREE", "TOTAL", "DESCRIPTION");
    for (const llamad::DeviceInfo & device : devices) {
        std::printf("%-*s  %-*s  %9s  %9s  %s\n", static_cast<int>(w_name), device.name.c_str(),
                    static_cast<int>(w_type), device.type.c_str(), format_gib(device.memory_free).c_str(),
                    format_gib(device.memory_total).c_str(), device.description.c_str());
    }
}

/// Select the per-user runtime socket path with a UID-based temporary fallback.
std::string default_socket_path() {
    const char * runtime_dir = std::getenv("XDG_RUNTIME_DIR");
    if (runtime_dir != nullptr && runtime_dir[0] != '\0') {
        return std::string(runtime_dir) + "/llamad.sock";
    }
    return "/tmp/llamad-" + std::to_string(static_cast<unsigned>(getuid())) + ".sock";
}

/// True if something is listening on `path` right now. A plain connect(2) is the
/// cheapest possible liveness probe and needs no gRPC machinery.
bool socket_is_live(const std::string & path) {
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return false;
    }
    // macOS has no SOCK_CLOEXEC; this probe runs before any worker threads exist.
    if (::fcntl(fd, F_SETFD, FD_CLOEXEC) == -1) {
        ::close(fd);
        return false;
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path.c_str());
    const bool live = ::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0;
    ::close(fd);
    return live;
}

/// Returns false (after printing why) if the path cannot be used.
bool prepare_socket_path(const std::string & path) {
    // Leave room for the NUL; sun_path's capacity depends on the platform.
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

}  // namespace

/// Run the executable.
/// @return Zero on success, two for invalid command-line usage, or one for a runtime failure.
int main(int argc, char ** argv) {
    Options               options;
    llamad::EngineFlags   engine_flags;
    llamad::cli::HelpFlag help_flag;
    llamad::EngineConfig  config;

    try {
        const std::vector<std::string> positional =
            llamad::cli::parse_flags(argc, argv, options, engine_flags, help_flag);
        if (!positional.empty()) {
            throw llamad::cli::FlagError("unknown argument '" + positional.front() + "'");
        }
        if (help_flag.help) {
            print_usage(argv[0]);
            return 0;
        }
        config = llamad::to_config(engine_flags);

        // Listing devices needs no model, and is the way to find the names --devices takes.
        if (engine_flags.list_devices) {
            print_device_table();
            return 0;
        }
        if (options.model.empty()) {
            throw llamad::cli::FlagError("--model is required");
        }
        config.model_path = options.model;
    } catch (const llamad::cli::FlagError & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        print_usage(argv[0]);
        return 2;
    }

    const std::string socket_path = options.socket.value_or(default_socket_path());

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
        std::fprintf(stderr, "[llamad] loading %s ...\n", config.model_path.c_str());
        engine = std::make_unique<llamad::Engine>(config);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    // One ChatFormat for the whole daemon: it is immutable, so every request renders and parses
    // through it concurrently (the per-request parser state is a separate Stream). A model with no
    // usable template still serves Generate, Tokenize and GetModelInfo; only Chat is refused, and
    // a template that will not parse is a warning rather than a reason not to start.
    std::unique_ptr<llamad::ChatFormat> chat_format;
    std::string chat_unavailable_reason = "the model has no built-in chat template";
    {
        const llamad::ChatTemplateInfo tmpl = engine->chat_template();
        if (!tmpl.source.empty()) {
            try {
                chat_format = std::make_unique<llamad::ChatFormat>(tmpl.source, tmpl.bos_token, tmpl.eos_token);
            } catch (const std::exception & e) {
                chat_unavailable_reason =
                    std::string("the model's chat template could not be parsed: ") + e.what();
                std::fprintf(stderr, "[llamad] warning: %s\n", chat_unavailable_reason.c_str());
                std::fprintf(stderr, "[llamad] warning: Chat is disabled; Generate still works\n");
            }
        } else {
            std::fprintf(stderr, "[llamad] warning: %s; Chat is disabled\n", chat_unavailable_reason.c_str());
        }
    }

    llamad::LlamaService service(*engine, chat_format.get(), chat_unavailable_reason);

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
