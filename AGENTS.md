# AGENTS.md

llamad is a small C++23 daemon that serves one llama.cpp model through gRPC on a Unix socket.
The proto is the product and the integration surface. There is no client library.
The project is pre-release. Breaking changes are welcome when they improve the design.
Explain their benefit and effect on callers.

## Repository map

| Path | Role |
|---|---|
| `proto/llamad/v1/llamad.proto` | Wire contract and stream guarantees |
| `src/engine.{h,cpp}` | Model, inference, KV cache and text stream |
| `src/chat_format.{h,cpp}` | Chat templates, tool grammar and tool-call parser |
| `src/wire.{h,cpp}` | Proto conversion and finish chunks |
| `src/service.{h,cpp}`, `src/main.cpp` | gRPC service and daemon lifecycle |
| `src/flags.h`, `src/engine_flags.h` | Command-line flags |
| `tests/` | Plain CTest executables, engine smoke tests and benchmarks |
| `examples/python/` | Python chat client from generated stubs, outside the CMake build |
| `third_party/llama.cpp` | Pinned submodule (do not modify) |

Read `README.md` and `docs/` for user documentation. Local GGUF files belong in `models/` and are absent from CI.

## Core principles

- Keep the project small and easy to read. Prefer simple solutions for present requirements.
- Use judgment within the requested scope. Improve or replace existing mechanisms when the benefit is clear.
- Keep the engine on the public llama.cpp C API. Only `chat_format.cpp` uses the llama.cpp `common` library.
  Keep llama.cpp types out of the chat header and gRPC/protobuf types in `service.*`, `wire.*` and `main.cpp`.
- Treat `llamad.proto` and `engine.h` as deliberate contracts. Keep conversions, tests, documentation and examples consistent.
  For proto changes, update `wire.cpp`, proto guarantees, `tests/wire_test.cpp` and `docs/protocol.md` together.
- Keep requests stateless. Clients supply history and execute tools. The daemon has no sessions or tool registries.
  The KV cache affects performance only. Every exit from generation must keep cached tokens consistent with the cache, or clear both.
- Preserve the stream contract: nonempty text chunks, then exactly one finish chunk.
  Text must contain no tool markup, partial UTF-8 or part of a matched stop string.
  Complete tool calls belong in the finish chunk. `tool_calls` is nonempty if and only if the reason is `TOOL_CALLS`.
- Keep access local: a Unix socket with mode 0600, with no TCP listener.
- Ask before you add a dependency. Measure performance changes before and after with `scripts/bench.sh`.

## Code and documentation

Follow the surrounding style: C++23, four spaces, `const T & name` and `T * name`.
Support GCC 13, Clang 18 and Apple Clang 16 with their standard libraries. Use RAII for C handles.
Keep actionable errors typed and map them to gRPC status codes in `service.cpp`.
Write daemon logs to stderr, with one `[llamad]` line per request.

Comments explain constraints and reasons. Documentation describes current behavior. History belongs in commit messages.
Update affected user documentation with the code.
Use ASD-STE100 Simplified Technical English in `README.md`, `docs/*.md` and this file.
Use active voice, simple tenses and short sentences: at most 20 words for instructions and 25 for descriptions.

## Build and verification

```sh
./scripts/build.sh cpu -DLLAMAD_WARNINGS_AS_ERRORS=ON
./scripts/test.sh cpu
./scripts/smoke-test.sh cpu models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

Use `gpu` for Vulkan on Linux or Metal on macOS. See `docs/building.md` and `docs/performance.md` for details.

- For code changes, build without warnings and run CTest.
- For chat or parser changes, extend `tests/chat_format_test.cpp`. For wire changes, extend `tests/wire_test.cpp`.
- For engine or service changes, test model inference through `engine_smoke` or the daemon, as appropriate.
  Use a private test socket and stop the daemon after the test.
- Use temperature zero and fresh processes for repeatable comparisons. A warm KV cache can change greedy output.
- CI tests CPU builds without models. Verify GPU and model inference locally when relevant.
- Report checks and any verification gaps. Match verification effort to the risk.

## Git

Work on a branch. Commit and push only when the user asks.
Keep commits coherent and independently buildable. Use `Area: what changed` subjects and explain the reason in the body.
