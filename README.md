# llamad

llamad is a small daemon that loads one llama.cpp model one time and serves it to local
applications. It uses gRPC on a Unix domain socket. The integration surface is one file,
[`proto/llamad/v1/llamad.proto`](proto/llamad/v1/llamad.proto). Generate stubs from this file
for your language. Your application then gets streamed completions, chat with tool calls, replies
that agree with a JSON schema, and embedding vectors. The application does not embed llama.cpp,
and each process does not load the model again.

The contract gives these guarantees in all languages:

- **A strict stream shape.** A stream has text chunks and then one `finish` chunk. A protobuf
  `oneof` holds these two types. Text chunks contain only output for the user. They never
  contain tool-call markup, part of a stop string or partial UTF-8.
- **Complete tool calls.** Tool calls are complete and are only in the `finish` chunk. The
  `finish` chunk contains tool calls only when its finish reason is `TOOL_CALLS`. A grammar
  limits the JSON arguments of each call to the schema of its tool.
- **Stateless requests.** Each request contains its full conversation and its tools. The daemon
  does not run tools and does not keep sessions. The KV cache keeps the prefix that a resent
  conversation shares with the previous request. Thus, the daemon decodes only the new tokens of
  a turn.
- **Local only.** The daemon creates a Unix socket with mode 0600. It does not open a TCP
  listener.

llama.cpp is a pinned submodule that is not modified. Its CPU, Vulkan (Linux) and Metal (macOS)
backends are available.

## Requirements

You must have a C++23 compiler (GCC 13, Clang 18, Apple Clang 16, or a later version), CMake
3.25, Ninja, gRPC and Protobuf. The supported platforms are Linux and macOS on Apple silicon.

```sh
pacman -S gcc grpc protobuf cmake ninja     # Arch Linux
brew install grpc protobuf cmake ninja      # macOS, with the Xcode command line tools
```

## Build

```sh
git clone --recurse-submodules git@github.com:crybo-rybo/llamad.git
cd llamad
./scripts/build.sh cpu                      # builds into build-cpu/ (gpu for Vulkan or Metal)
./scripts/test.sh cpu                       # a model is not necessary
```

For GPU builds, multi-GPU selection and the smoke test, refer to
[docs/building.md](docs/building.md). The smoke test must have a model.

## Run

```sh
./build-cpu/llamad --model /absolute/path/to/model.gguf
# [llamad] listening on unix:/run/user/1000/llamad.sock

grpcurl -plaintext -d '{"messages": [{"role": "user", "content": "Name three primes."}]}' \
    unix:///run/user/1000/llamad.sock llamad.v1.Llama/Chat
```

Start the daemon the same way for an embedding model. The GGUF of an embedding model declares a
pooling type, for example bge-small-en-v1.5 or Qwen3-Embedding. With such a model, the daemon
answers `Embed` requests, and not `Generate` or `Chat` requests.

The project does not include a model. You can use any GGUF file. The daemon accepts the flags
`--socket`, `--ctx`, `--ngl`, `--threads`, `--devices` and `--tensor-split`. For more
information, refer to [docs/daemon.md](docs/daemon.md).

## Use llamad from your application

```python
import grpc
from llamad.v1 import llamad_pb2 as pb, llamad_pb2_grpc as rpc   # generated from llamad.proto

stub = rpc.LlamaStub(grpc.insecure_channel("unix:/run/user/1000/llamad.sock"))
for chunk in stub.Chat(pb.ChatRequest(messages=[pb.ChatMessage(role="user", content="Hi")])):
    if chunk.WhichOneof("chunk") == "text":
        print(chunk.text, end="")
```

For information about stubs, streams, the tool-call loop, typed replies, embeddings and errors,
refer to [docs/protocol.md](docs/protocol.md).

## Python example

`examples/python/chat.py` is a runnable version of the snippet above. It is a terminal chat with
streamed replies and stats. To use it, start the daemon. Then, from the repository root, do these
steps:

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r examples/python/requirements.txt
python examples/python/chat.py
```

The client compiles `llamad.proto` when it starts, so you do not generate stubs. With
[uv](https://docs.astral.sh/uv/), `uv run examples/python/chat.py` does all of these steps.

For the commands and the stats line, refer to [docs/python-example.md](docs/python-example.md).

## Documentation

| Page | Contents |
|---|---|
| [docs/protocol.md](docs/protocol.md) | Integration: stubs, streams, tool calls, typed replies, embeddings, errors |
| [docs/python-example.md](docs/python-example.md) | Python chat client: proto loaded at start-up, streamed replies, a tool call and stats |
| [docs/building.md](docs/building.md) | Dependencies, platform notes, GPU builds, tests, smoke test |
| [docs/daemon.md](docs/daemon.md) | Daemon flags, socket, signals, and `engine_smoke` |
| [docs/design.md](docs/design.md) | Design notes: layers, stream shape, chat templates |
| [docs/performance.md](docs/performance.md) | How to measure prefill and decode speed, and reference numbers |
| `./scripts/docs.sh` | Doxygen reference for the source files of the daemon, in `build-docs/html/` |
| `AGENTS.md` | Layout of the code, and rules for changes to it |

## License

llamad has the MIT License ([LICENSE](LICENSE)).

The build uses these libraries:

- llama.cpp (MIT)
- gRPC (Apache-2.0)
- Protobuf (BSD-3-Clause)
- Abseil (Apache-2.0)
- nlohmann/json (MIT, included in llama.cpp)
