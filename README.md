# llamad

A small daemon that loads one llama.cpp model once and serves it to local applications over gRPC
on a Unix domain socket. The integration surface is one file,
[`proto/llamad/v1/llamad.proto`](proto/llamad/v1/llamad.proto): generate stubs for any language
and get streaming completions, chat with tool calling, schema-constrained replies, or embedding
vectors, without embedding llama.cpp and without paying the model load time in every process.

What the contract promises, in every language:

- **A strict stream shape.** Text chunks, then exactly one `finish` chunk, as a protobuf `oneof`.
  Text is user-visible output only: never tool-call markup, never part of a stop string, never
  partial UTF-8.
- **Whole tool calls.** Calls arrive complete on `finish`, with JSON arguments constrained to the
  tool's schema, and only when the finish reason says so.
- **Stateless requests.** Each request carries its whole conversation and its tools; the daemon
  never runs a tool and keeps no sessions. The KV cache keeps the prefix a resent conversation
  shares with the previous request, so a turn decodes only its new tokens.
- **Local only.** A Unix socket created 0600 and no TCP listener.

llama.cpp is an unmodified, pinned submodule, with CPU, Vulkan (Linux) and Metal (macOS) backends.

## Requirements

A C++23 compiler (GCC 13, Clang 18, Apple Clang 16, or later), CMake 3.25, Ninja, gRPC and
Protobuf. Linux and macOS on Apple silicon are the supported platforms.

```sh
pacman -S gcc grpc protobuf cmake ninja     # Arch Linux
brew install grpc protobuf cmake ninja      # macOS, with Xcode's command line tools
```

## Build

```sh
git clone --recurse-submodules git@github.com:crybo-rybo/llamad.git
cd llamad
./scripts/build.sh cpu                      # builds into build-cpu/; `gpu` for Vulkan or Metal
./scripts/test.sh cpu                       # no model needed
```

[docs/building.md](docs/building.md) covers GPU builds, multi-GPU selection and the smoke test
that needs a model.

## Run

```sh
./build-cpu/llamad --model /absolute/path/to/model.gguf
# [llamad] listening on unix:/run/user/1000/llamad.sock

grpcurl -plaintext -d '{"messages": [{"role": "user", "content": "Name three primes."}]}' \
    unix:///run/user/1000/llamad.sock llamad.v1.Llama/Chat
```

An embedding model (one whose GGUF declares a pooling type, such as bge-small-en-v1.5 or
Qwen3-Embedding) is served the same way and answers `Embed` instead of `Generate` and `Chat`.

The project ships no model. Any GGUF works; the daemon takes `--socket`, `--ctx`, `--ngl`,
`--threads`, `--devices` and `--tensor-split`, described in [docs/daemon.md](docs/daemon.md).

## Use it from your application

```python
import grpc
from llamad.v1 import llamad_pb2 as pb, llamad_pb2_grpc as rpc   # generated from llamad.proto

stub = rpc.LlamaStub(grpc.insecure_channel("unix:/run/user/1000/llamad.sock"))
for chunk in stub.Chat(pb.ChatRequest(messages=[pb.ChatMessage(role="user", content="Hi")])):
    if chunk.WhichOneof("chunk") == "text":
        print(chunk.text, end="")
```

[docs/protocol.md](docs/protocol.md) covers generating stubs, streams, the tool-calling loop,
typed replies, embeddings and errors.

## Documentation

| Page | Contents |
|---|---|
| [docs/protocol.md](docs/protocol.md) | Integrating: stubs, streams, tool calling, typed replies, embeddings, errors |
| [docs/building.md](docs/building.md) | Dependencies, platform notes, GPU builds, tests, smoke test |
| [docs/daemon.md](docs/daemon.md) | Daemon flags, socket, signals, and `engine_smoke` |
| [docs/design.md](docs/design.md) | Design notes: layering, stream shape, chat templates |
| [docs/performance.md](docs/performance.md) | Measuring prefill and decode speed, and reference numbers |
| `./scripts/docs.sh` | Doxygen reference for the daemon's sources in `build-docs/html/` |
| `AGENTS.md` | Layout and rules for changing the code |

## License

llamad is released under the MIT License ([LICENSE](LICENSE)).

It builds against:

- llama.cpp (MIT)
- gRPC (Apache-2.0)
- Protobuf (BSD-3-Clause)
- Abseil (Apache-2.0)
- nlohmann/json (MIT, vendored with llama.cpp)
