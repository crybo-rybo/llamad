# AGENTS.md

Guidance for AI coding agents working in this repository. `README.md` and `docs/*.md` are the user-facing
documentation; this file is about how to change the code well.

## What this project is

llamad is a small C++26 daemon that loads one llama.cpp model once and serves it to local
applications over gRPC on a Unix domain socket. Applications link a tiny client library and get
streaming completion, chat and tool calling without embedding llama.cpp.

It is deliberately small: a few thousand lines across a dozen or so files, no framework, few
dependencies. That smallness is a feature. Every change should leave it as easy to read as it was.

The project is pre-release. Nothing has compatibility obligations yet, and the source is the
single source of truth for how things work.

## Layout

| Path | Role |
|---|---|
| `proto/llamad/v1/llamad.proto` | The wire contract. Engine-agnostic: nothing in it depends on how inference runs. |
| `proto/llamad/v1/convert.h` | Name-matched conversion between those messages and the plain structs that mirror them, on both sides of the wire. Names no protobuf type: a message is reached only through the accessors protoc generates. |
| `src/engine.{h,cpp}` | Wrapper over libllama: model loading, tokenizing, sampling, the generate loop, UTF-8 and stop-string safe streaming. |
| `src/chat_format.{h,cpp}` | Chat layer: renders messages and tools through the model's Jinja template, builds the tool-call grammar, parses tool calls out of generated text. |
| `src/service.{h,cpp}`, `src/main.cpp` | gRPC service and the daemon: socket lifecycle, signals, proto ↔ engine type conversion. |
| `src/cli/flags.h` | The one command-line parser and `--help` printer, over a struct whose members are a binary's flags. Target `llamad_flags` exposes only `src/cli`, so `llamad-chat` uses it without reaching a daemon header. |
| `src/engine_flags.h` | The context and offload flags `llamad` and `engine_smoke` share, the `EngineConfig` they describe, and `print_device_table` for `--list-devices`. |
| `client/` | Client library (`include/llamad/client.h`, `src/client.cpp`) and `llamad-chat` (`examples/chat_cli.cpp`). `include/llamad/json.h` maps reflected tool arguments, results and schemas to nlohmann/json and reports what does not fit as `json::Error`; `client.h` includes it, so an application still includes one header. |
| `tests/` | Plain-executable tests registered with CTest, needing no model file and sharing the `CHECK` macros in `check.h`, and `engine_smoke.cpp`: a CLI that drives the engine and chat layer in-process, with no daemon and no gRPC. |
| `tests/consumer/` | A separate CMake project that pulls llamad in with `FetchContent` and no submodules, the way an application does, and links `llamad::client` and `llamad::proto`. Included by another project, llamad builds client-only (`LLAMAD_CLIENT_ONLY`); CI builds and runs this against each commit. |
| `third_party/llama.cpp` | Pinned, unmodified submodule. |
| `models/` | Local GGUF files. Gitignored; not available in CI. |

## Build, test, run

```sh
./scripts/build.sh cpu                   # or gpu: Vulkan on Linux, Metal on macOS
./scripts/test.sh cpu                    # ctest in build-cpu/; no model or daemon needed

./build-cpu/llamad --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --socket /tmp/llamad-dev.sock
./build-cpu/client/llamad-chat --socket /tmp/llamad-dev.sock --once "Hello" --temp 0
./build-cpu/tests/engine_smoke --help    # engine without the daemon
```

On macOS, run `./scripts/build-deps-macos.sh` once first; `build.sh` selects the macOS
compilers and the gRPC prefix. docs/building.md says why.

`build*/` directories are gitignored. When you start a daemon for testing, give it a private
`--socket` path and stop it when you are done.

## Boundaries that must hold

These are the load-bearing design decisions. A change that breaks one of them needs the user's
agreement first, not a clever workaround.

1. **Layering.** `engine.{h,cpp}` use only the public `llama.h` / `ggml-backend.h` C API and
   contain no gRPC or protobuf types. `chat_format.cpp` is the only file that touches llama.cpp's
   `common` library, which is unstable and not a public API; its header exposes no llama.cpp
   types. All gRPC and protobuf knowledge lives in `service.*` and `main.cpp`. `client.h` exposes
   no gRPC or protobuf types, and `src/cli/flags.h` includes only the standard library. The
   point: a submodule bump can break at most one file, and an application needs one header.
2. **llama.cpp is never patched.** It is a pinned submodule. If something is missing, solve it on
   this side of the boundary or raise it.
3. **The three contracts change deliberately.** `llamad.proto`, `engine.h` and `client.h` are the
   seams the rest is built against. Proto changes are additive: never renumber or repurpose a
   field. When one contract changes, update everything that mirrors it in the same change
   (proto ↔ `service.cpp` ↔ `client.h` / `client.cpp`). `tests/contract_test.cpp` fails the build
   when a mirror struct and the message it mirrors disagree, in either direction.
4. **The daemon is stateless per request.** No sessions, no registries, no state carried between
   calls that a request's meaning depends on. Clients resend history; tools travel with each
   request as opaque data and are never executed or validated by the daemon. The one thing that
   outlives a request is the engine's KV cache, and it is a performance cache only: the engine
   records the tokens the cache holds, a request decodes just the part of its prompt after the
   prefix it shares with them, and every path out of `generate` leaves the record matching the
   cache or empties both. It changes how long a prompt takes, never which prompt runs. Decoding
   the same tokens in a different batch split is not bit-for-bit identical, though, so a greedy
   reply from a warm cache can differ from a freshly started daemon's.
5. **Stream shape.** Zero or more text chunks, then exactly one final chunk carrying
   `finish_reason` and `stats`. Tool calls arrive whole on that final chunk, and `tool_calls` is
   non-empty if and only if the finish reason is `TOOL_CALLS`. Text chunks carry user-visible
   content only — never tool-call markup, never partial UTF-8, never part of a matched stop string.
6. **Local only.** A Unix socket created 0600. No TCP listener.

## How to make changes

**Solve the problem that was asked, at the size it actually is.** Before writing code, state the
change to yourself in one or two sentences. If the diff you are producing is much larger than that
sentence suggests, stop and reconsider — you have probably started solving a different problem.

**Prefer the boring solution.** A function over a class. A class over a hierarchy. A few clear
lines repeated twice over an abstraction with one and a half users. A `std::vector` and a loop
over a cache nobody has measured a need for. Reach for something heavier only when a concrete,
present requirement forces it, and say what that requirement is.

**Do not build for imagined futures.** No configuration options nobody asked for, no plugin
points, no "extensible" interfaces with a single implementation, no generic helpers for one call
site, no compatibility shims for callers that do not exist. When the future arrives, the code
will be small enough to change.

**Stay out of the weeds.** A task has a centre. Edge cases that can really happen deserve
handling; hypothetical ones deserve, at most, one line in your report. If you notice an unrelated
problem while working, mention it — do not fix it in the same change. If you have spent more
effort on a side issue than on the task itself, step back.

**Reuse before adding.** Read the surrounding code first. This codebase already has a streaming
holdback filter, RAII wrappers for llama.cpp handles, proto conversion helpers, a flag parser
driven by an options struct and error types. Extend what exists before introducing a parallel
mechanism.

**No new dependencies without asking.** The dependency list is llama.cpp, gRPC, Protobuf and
nlohmann/json (the header shipped in the pinned llama.cpp vendor tree). Tests are plain
executables with a `CHECK` macro. The client delegates JSON syntax and serialization to
nlohmann and keeps only reflection, schema generation and checked C++ conversions in `json.h`.

**Write for the next reader.** Code is read far more than it is written, mostly by someone
without your current context. Favour names that say what a thing is, straight-line control flow,
early returns over nesting, and small functions that do one thing. If a piece of code needs a
paragraph to justify its cleverness, make it less clever.

**Performance follows measurement.** The hot path is llama.cpp's, not ours. Keep per-token work
cheap and obvious; do not trade readability for speed without a number that says it matters.

## Code style

Match the file you are in; consistency beats preference.

- C++26 with static reflection (`<meta>`; GCC 16 or later, `-freflection`).
- 4-space indent. `const T & name` and `T * name` with spaces around `&` and `*`.
- Aligned declarations and assignments where the surrounding code aligns them.
- `/*name*/` comments on literal arguments whose meaning is not obvious: `tokenize(text, /*add_special*/ false, ...)`.
- Anonymous namespaces for file-local helpers. Pimpl where a header must hide a dependency.
- RAII for every llama.cpp / C handle, so every path — exceptions included — releases it.
- Errors callers can act on are typed (`EngineError`, `ChatFormatError`, `RpcError`) and map to
  gRPC status codes in `service.cpp`: caller mistakes → `INVALID_ARGUMENT`, a missing capability
  → `FAILED_PRECONDITION`, everything else → `INTERNAL`.
- The daemon logs one `[llamad] ...` line per request to stderr and writes nothing to stdout.

## Comments and documentation

- Comments explain **why**: a constraint, a non-obvious consequence, a reason the simpler-looking
  approach is wrong. Do not narrate what the next line plainly does. Match the existing density —
  terse and purposeful.
- **Describe the code as it is, in the present tense.** Do not write "used to", "no longer",
  "now", "new", "behaviour change", "previously" or migration notes in comments, the README or
  this file. When behaviour changes, rewrite the affected text as though it had always been so.
  History belongs in commit messages.
- Keep `README.md` true in the same change that makes it false: flags, defaults, behaviour and
  examples.

## Verifying a change

Match the effort to the risk, and report what you actually ran.

- Always: the project builds without new warnings, and `ctest` passes.
- Chat layer or parsing changes: add or extend a case in `tests/chat_format_test.cpp`. These tests
  run canned model output through a real template with no GGUF, so they are cheap — prefer them
  to manual checks wherever the logic can be reached that way.
- Client changes reachable without a daemon — the JSON reader, a tool's schema or dispatch —
  belong in `tests/json_test.cpp` or `tests/client_tools_test.cpp`, which need neither. The
  `Client::chat` tool loop is covered by `tests/client_chat_test.cpp`, which scripts a fake of
  the service on a private socket.
- Engine, service or client changes: exercise the real path. `engine_smoke` covers the engine
  alone (`--stop`, `--cancel-after`, `--grammar-file`, `--chat --demo-tool`); `llamad-chat --once`
  against a running daemon covers the full stack (`--demo-tools` for the tool loop, `--demo-json`
  for a typed reply). Use `--temp 0` for repeatable output; only a cold KV cache repeats exactly
  (boundary 4), so compare first requests to freshly started processes. The prompt-cache
  bookkeeping that needs no model is checked in `tests/engine_test.cpp`. The preferred local
  smoke-test model is `models/qwen2.5-0.5b-instruct-q4_k_m.gguf`, unless the behaviour needs a
  stronger one. Pass its path explicitly to `./scripts/smoke-test.sh cpu` or
  `./scripts/smoke-test.sh gpu`; the script requires a model argument, and model files are local
  and gitignored.
- CI builds Linux CPU-only with GCC and runs the tests. It does not exercise GPU execution or
  load a model, so inference paths are only verified locally. Say so when that is the case
  rather than implying coverage.
- Report results plainly, including what you did not or could not verify.

## Git

- Work on a branch; `main` stays buildable. Commit and push only when asked.
- Commits are feature-sized, each builds on its own, and history stays linear (fast-forward).
- Message style: a specific subject line (`Area: what changed`), then a body explaining what and
  why in prose, including measurements where they motivated the change.
