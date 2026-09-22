# llamad

A small daemon that loads one llama.cpp model once and serves it to local C++
applications over gRPC on a Unix domain socket. Applications link a tiny client
library and get streaming completions, chat and tool calling without embedding
llama.cpp, and without paying the model load time in every process.

- Local only: a Unix socket created 0600, no TCP listener.
- Stateless: clients resend history; tools travel with each request and are never executed by the daemon.
- llama.cpp is an unmodified, pinned submodule. CPU, Vulkan (Linux) and Metal (macOS) backends.

## Requirements

C++26 with static reflection, so GCC 16 or later; Clang does not implement reflection.
Linux and macOS on Apple silicon are the supported platforms.

```sh
pacman -S gcc grpc protobuf cmake ninja        # Arch Linux
brew install gcc cmake ninja openssl@3         # macOS; gRPC is built from source, see below
```

## Build

```sh
git clone --recurse-submodules git@github.com:crybo-rybo/llamad.git
cd llamad
./scripts/build-deps-macos.sh               # macOS only, once: builds gRPC into build-deps/
./scripts/build.sh cpu                      # builds into build-cpu/; `gpu` for Vulkan or Metal
./scripts/test.sh cpu                       # no model needed
```

[docs/building.md](docs/building.md) covers the compiler and dependency choices, GPU
builds, multi-GPU selection and the smoke test that needs a model.

## Run

```sh
./build-cpu/llamad --model /absolute/path/to/model.gguf
# [llamad] listening on unix:/run/user/1000/llamad.sock

./build-cpu/client/llamad-chat                          # interactive REPL
./build-cpu/client/llamad-chat --once "Hello" --temp 0
./build-cpu/client/llamad-chat --demo-tools             # tool calling, end to end
./build-cpu/client/llamad-chat --demo-json              # a reply parsed into a struct
```

The project ships no model. Any GGUF works; the daemon takes `--socket`, `--ctx`, `--ngl`,
`--threads`, `--devices` and `--tensor-split`, described in [docs/daemon.md](docs/daemon.md).

## Use it from your own project

```cmake
add_subdirectory(llamad EXCLUDE_FROM_ALL)   # the repo root, not client/
target_link_libraries(myapp PRIVATE llamad_client)
```

```cpp
#include <llamad/client.h>

llamad::client::Client client;                 // default socket path
llamad::client::SamplingParams params;
params.temperature = 0.0f;

auto result = client.chat({{"user", "Name three primes."}}, params,
                          [](const std::string & text) {
                              std::fputs(text.c_str(), stdout);
                              return true;      // false cancels the request
                          });
```

`llamad/client.h` exposes no gRPC or protobuf types. Tools are plain C++ functions
registered with `ToolSet::add`; the client runs the execute-and-resend loop.
`chat<T>` returns the reply as an instance of a reflected struct, constrained by its schema.
[docs/client.md](docs/client.md) has the full walkthrough.

## Documentation

| Page | Contents |
|---|---|
| [docs/building.md](docs/building.md) | Dependencies, platform notes, GPU builds, tests, smoke test |
| [docs/daemon.md](docs/daemon.md) | Daemon flags, socket, signals, `llamad-chat` and `engine_smoke` |
| [docs/client.md](docs/client.md) | Client library, tool calling, typed replies, JSON conversions |
| [docs/design.md](docs/design.md) | Design notes: layering, stream shape, chat templates |
| `./scripts/docs.sh` | Doxygen API reference in `build-docs/html/` (needs CMake and Doxygen 1.17) |
| `AGENTS.md` | Layout and rules for changing the code |
