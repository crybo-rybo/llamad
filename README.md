# llamad

A small daemon that loads one llama.cpp model once and serves it to local C++
applications over gRPC on a Unix domain socket. Applications link a tiny client
library and get streaming completions and chat without embedding llama.cpp, and
without paying the model load time in every process.

## Dependencies

```sh
pacman -S grpc protobuf cmake ninja
```

## Build

```sh
git clone --recurse-submodules git@github.com:crybo-rybo/llamad.git
cd llamad
cmake -S . -B build -G Ninja
cmake --build build -j
```

(If you already cloned without submodules: `git submodule update --init --recursive`.)

## GPU

llama.cpp's GPU backends are enabled with their usual CMake flags. Vulkan is the
one tested here: it runs on Pascal cards (GTX 10xx), which CUDA 13 no longer targets.

```sh
pacman -S vulkan-headers spirv-headers vulkan-icd-loader shaderc
cmake -S . -B build -G Ninja -DGGML_VULKAN=ON
cmake --build build -j
```

Every discrete GPU is used by default: llama.cpp splits the model's layers
across them in proportion to each card's free memory, and ignores an integrated
GPU whenever a discrete one exists. Two 8 GB cards therefore hold a model that
fits on neither alone. The daemon prints one line per offload device at startup.

```sh
./build/llamad --list-devices                        # names, types, free/total memory
./build/llamad --model M --devices Vulkan0           # this card only
./build/llamad --model M --tensor-split 3,1          # 3:1 share, in device order
```

## Run the daemon

```sh
./build/llamad --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf
# [llamad] listening on unix:/run/user/1000/llamad.sock
```

Options: `--socket PATH` (default `$XDG_RUNTIME_DIR/llamad.sock`, else
`/tmp/llamad-<uid>.sock`), `--ctx N` (4096), `--ngl N` (99), `--threads N` (0 =
auto, half the hardware threads). The socket is created mode 0600, so only your
user can talk to it.
SIGINT/SIGTERM shut the daemon down and remove the socket.

## Chat from the terminal

```sh
./build/client/llamad-chat                       # interactive REPL
./build/client/llamad-chat --once "Hello" --temp 0
```

Also accepts `--socket`, `--system TEXT`, `--seed N`, `--max-tokens N`. Ctrl-C
cancels the reply in progress; Ctrl-C or Ctrl-D at the prompt quits.

## Use it from your own project

```cmake
add_subdirectory(llamad EXCLUDE_FROM_ALL)   # the repo root, not client/
target_link_libraries(myapp PRIVATE llamad_client)
```

`EXCLUDE_FROM_ALL` means only what `myapp` links gets built: the client and the
generated protobuf code, not llama.cpp or the daemon.

```cpp
#include <llamad/client.h>
#include <cstdio>

int main() {
    llamad::client::Client client;                 // default socket path
    llamad::client::SamplingParams params;
    params.temperature = 0.0f;
    params.max_tokens  = 128;

    auto result = client.chat({{"user", "Name three primes."}}, params,
                              [](const std::string & text) {
                                  std::fwrite(text.data(), 1, text.size(), stdout);
                                  std::fflush(stdout);
                                  return true;      // false cancels the request
                              });
    std::printf("\n%d tokens\n", result.stats.completion_tokens);
}
```

`llamad/client.h` exposes no gRPC or protobuf types, so your build needs neither
on its include path.

## Design notes

- **No TCP listener.** gRPC here is HTTP/2 over a Unix domain socket. There is
  no port to firewall; access control is the socket's file permissions (0600).
- **llama.cpp is an unmodified pinned submodule**, not a fork. The engine uses
  only the public `llama.h` C API, so updating the submodule is a submodule bump
  rather than a merge.
- **v1 serves one generation at a time.** The engine serializes `generate`, so
  concurrent clients queue rather than sharing the context. Each request starts
  from an empty KV cache.
- **Chat templating uses llama.cpp's built-in template matcher**
  (`llama_chat_apply_template`), not Jinja, so it supports the templates
  llama.cpp recognizes and nothing more.
