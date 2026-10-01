# AGENTS.md

Guidance for AI coding agents working in this repository. `README.md` and `docs/*.md` are the user-facing
documentation; this file is about how to change the code well.

## What this project is

llamad is a small C++23 daemon that loads one llama.cpp model once and serves it to local
applications over gRPC on a Unix domain socket. Its product is the wire contract,
`proto/llamad/v1/llamad.proto`: applications in any language generate stubs from it and get
streaming completion, chat, tool calling and embeddings without embedding llama.cpp. There is no
client library.

It is deliberately small: a few thousand lines across a dozen or so files, no framework, few
dependencies. That smallness is a feature. Every change should leave it as easy to read as it was.

The project is pre-release. Nothing has compatibility obligations yet, and the source is the
single source of truth for how things work. Deliberate breaking changes are welcome when they
make a meaningful improvement to the project; explain the benefit and the impact on callers.

## Layout

| Path | Role |
|---|---|
| `proto/llamad/v1/llamad.proto` | The wire contract, and the guarantees it keeps written at its top. Engine-agnostic: nothing in it depends on how inference runs. `proto/CMakeLists.txt` generates it into target `llamad_proto`. |
| `src/engine.{h,cpp}` | Wrapper over libllama: model loading, tokenizing, sampling, the generate loop, UTF-8 and stop-string safe streaming. |
| `src/chat_format.{h,cpp}` | Chat layer: renders messages and tools through the model's Jinja template, builds the tool-call grammar, parses tool calls out of generated text. |
| `src/wire.{h,cpp}` | Field-by-field conversion between wire messages and the engine and chat layer's types, and the `finish` chunk that enforces the tool-call invariant. |
| `src/service.{h,cpp}`, `src/main.cpp` | gRPC service and the daemon: socket lifecycle, signals, error-to-status mapping. |
| `src/flags.h` | `cli::Args`, the small command-line reader every binary walks its flags with, and `cli::FlagError`. Each binary has its own if/else chain over the flags it takes and its own `--help` text. |
| `src/engine_flags.h` | The context and offload flags `llamad`, `engine_smoke` and `engine_bench` share, the `EngineConfig` they describe, their `--help` lines, and `print_device_table` for `--list-devices`. |
| `tests/` | Plain-executable tests registered with CTest, needing no model file and sharing the `CHECK` macros in `check.h`. Two CLIs that need a model drive the engine and chat layer in-process, with no daemon and no gRPC: `engine_smoke.cpp` for behaviour, and `engine_bench.cpp` for prefill and decode speed (`scripts/bench.sh`, docs/performance.md). |
| `third_party/llama.cpp` | Pinned, unmodified submodule. |
| `docs/site/`, `docs/proto_reference.py` | The GitHub Pages site, built by Doxygen with `./scripts/docs.sh`: theme, layout, overview and architecture pages, and inline SVG diagrams whose links name generated pages. The wire protocol page is generated from the proto by `proto_reference.py` (Python 3, standard library), which stops the build on any construct it cannot document. |
| `models/` | Local GGUF files. Gitignored; not available in CI. |

## Build, test, run

```sh
./scripts/build.sh cpu                   # or gpu: Vulkan on Linux, Metal on macOS
./scripts/test.sh cpu                    # ctest in build-cpu/; no model or daemon needed

./build-cpu/llamad --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --socket /tmp/llamad-dev.sock
grpcurl -plaintext -d '{"messages": [{"role": "user", "content": "Hello"}], "sampling": {"temperature": 0}}' \
    unix:///tmp/llamad-dev.sock llamad.v1.Llama/Chat
./build-cpu/tests/engine_smoke --help    # engine without the daemon
```

Every platform builds with its own compiler and its package manager's gRPC; docs/building.md
has the details.

`build*/` directories are gitignored. When you start a daemon for testing, give it a private
`--socket` path and stop it when you are done.

## Design principles

These principles describe the current architecture and why it works. Use them as a starting
point, and evolve them when a better design materially improves correctness, usability, clarity
or maintainability. Explain the tradeoffs and keep the implementation, tests and documentation
consistent. Work within the user's intended scope and use approval already given in the task;
ask when a proposed change would expand that scope.

1. **Layering.** `engine.{h,cpp}` use only the public `llama.h` / `ggml-backend.h` C API and
   contain no gRPC or protobuf types. `chat_format.cpp` is the only file that touches llama.cpp's
   `common` library, which is unstable and not a public API; its header exposes no llama.cpp
   types. All gRPC and protobuf knowledge lives in `service.*`, `wire.*` and `main.cpp`.
   `flags.h` includes only the standard library. The point: a submodule bump can break at most
   one file, and the engine and chat layer can be tested without a daemon.
2. **Keep dependency changes maintainable.** llama.cpp is an unmodified, pinned submodule.
   Prefer changes in llamad or fixes upstream so dependency upgrades stay straightforward.
3. **The two contracts change deliberately.** `llamad.proto` and `engine.h` are the seams the
   rest is built against, and the proto is the one applications see. During pre-release, these
   contracts may change incompatibly when the result is meaningfully better. Weigh the benefit
   against the work callers need to update, and describe both in the PR. Keep definitions,
   conversions, tests, documentation and examples in sync. When the proto changes, update
   `wire.cpp`, the guarantees at the top of the proto and docs/protocol.md in the same change.
   `tests/wire_test.cpp` names any field of a converted message that `wire.cpp` does not carry.
4. **No client library.** The proto is the integration surface, in every language. What only an
   application can know (which tools exist and what they do, the tool-call loop, what a reply
   means as a type) stays out of this repository.
5. **The daemon is stateless per request.** No sessions, no registries, no state carried between
   calls that a request's meaning depends on. Clients resend history; tools travel with each
   request as opaque data and are never executed or validated by the daemon. The one thing that
   outlives a request is the engine's KV cache, and it is a performance cache only: the engine
   records the tokens the cache holds, a request decodes just the part of its prompt after the
   prefix it shares with them, and every path out of `generate` leaves the record matching the
   cache or empties both. It changes how long a prompt takes, never which prompt runs. Decoding
   the same tokens in a different batch split is not bit-for-bit identical, though, so a greedy
   reply from a warm cache can differ from a freshly started daemon's.
6. **Stream shape.** Zero or more `text` chunks, then exactly one `finish` chunk carrying the
   finish reason and stats (`GenerateChunk` is a `oneof` of the two). Tool calls arrive whole on
   `finish`, and `tool_calls` is non-empty if and only if the reason is `TOOL_CALLS`. Text chunks
   carry user-visible content only — never empty, never tool-call markup, never partial UTF-8,
   never part of a matched stop string.
7. **Local only.** A Unix socket created 0600. No TCP listener.

## How to make changes

**Aim for an excellent result.** Strive for correctness, a useful and coherent interface, clear
code and a design that is easy to maintain. Evaluate whether the current infrastructure serves
the problem well. Simplify, replace or remove it when doing so makes the project materially better.
Carry justified improvements through to a complete result, and support claims with appropriate
tests or measurements.

**Keep scope purposeful.** Before writing code, state the intended outcome in one or two
sentences. Choose the smallest coherent change that fully achieves it. A substantial refactor
is appropriate when it resolves a root cause or produces a clear improvement. If the diff grows,
check that each part contributes to that outcome and explain why the scope is necessary.

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

**Understand before changing.** Read the surrounding code first. This codebase already has a
streaming holdback filter, RAII wrappers for llama.cpp handles, proto conversion in `wire.cpp`,
a flag reader and error types. Reuse mechanisms that fit the task well. Replace or simplify them
when a different design makes the project meaningfully better, and remove superseded code so
the result stays coherent.

**No new dependencies without asking.** The dependency list is llama.cpp, gRPC and Protobuf
(nlohmann/json comes with llama.cpp's `common`, for the chat layer). Tests are plain executables
with a `CHECK` macro.

**Write for the next reader.** Code is read far more than it is written, mostly by someone
without your current context. Favour names that say what a thing is, straight-line control flow,
early returns over nesting, and small functions that do one thing. If a piece of code needs a
paragraph to justify its cleverness, make it less clever.

**Performance follows measurement.** The hot path is llama.cpp's, not ours. Keep per-token work
cheap and obvious; do not trade readability for speed without a number that says it matters.

## Code style

Match the file you are in; consistency beats preference.

- C++23, built by the platform's own compiler: GCC 13 or Clang 18 on Linux, Apple Clang 16 on
  macOS. Stay inside what all three standard libraries provide (libc++ has no floating-point
  `std::from_chars` before LLVM 20, for one), and build with `-DLLAMAD_WARNINGS_AS_ERRORS=ON`.
- 4-space indent. `const T & name` and `T * name` with spaces around `&` and `*`.
- Aligned declarations and assignments where the surrounding code aligns them.
- `/*name*/` comments on literal arguments whose meaning is not obvious: `tokenize(text, /*add_special*/ false, ...)`.
- Anonymous namespaces for file-local helpers. Pimpl where a header must hide a dependency.
- RAII for every llama.cpp / C handle, so every path — exceptions included — releases it.
- Errors callers can act on are typed (`EngineError`, `ChatFormatError`) and map to
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
- Wire changes: `tests/wire_test.cpp` checks the conversion in both directions and each
  converted message's field list; extend it with the field.
- Engine or service changes: exercise the real path. `engine_smoke` covers the engine alone
  (`--stop`, `--cancel-after`, `--grammar-file`, `--chat --demo-tool`); a running daemon covers
  the full stack, through grpcurl or stubs generated from the proto in any language (a tool
  round, a `response_json_schema` reply, a cancelled stream). Use `--temp 0` for repeatable output; only a cold KV cache repeats exactly
  (design principle 5), so compare first requests to freshly started processes. The prompt-cache
  bookkeeping that needs no model is checked in `tests/engine_test.cpp`. The preferred local
  smoke-test model is `models/qwen2.5-0.5b-instruct-q4_k_m.gguf`, unless the behaviour needs a
  stronger one. Pass its path explicitly to `./scripts/smoke-test.sh cpu` or
  `./scripts/smoke-test.sh gpu`; the script requires a model argument, and model files are local
  and gitignored.
- A change made for speed: measure it with `./scripts/bench.sh` against the build before it, run
  in the same sitting, and put both numbers in the commit message (docs/performance.md).
- CI builds CPU-only with GCC and Clang on Linux and Apple Clang on macOS, warnings as errors,
  and runs the tests. It does not exercise GPU execution or load a model, so inference paths
  are only verified locally. Say so when that is the case
  rather than implying coverage.
- Report results plainly, including what you did not or could not verify.

## Git

- Work on a branch; `main` stays buildable. Commit and push only when asked.
- Commits are feature-sized, each builds on its own, and history stays linear (fast-forward).
- Message style: a specific subject line (`Area: what changed`), then a body explaining what and
  why in prose, including measurements where they motivated the change.
