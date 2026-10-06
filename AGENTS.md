# AGENTS.md

This file gives guidance to AI coding agents that work in this repository. `README.md` and
`docs/*.md` are the documentation for users. This file tells how to change the code well.

## The project

llamad is a small C++23 daemon. It loads one llama.cpp model one time and serves it to local
applications through gRPC on a Unix domain socket. The product is the wire contract,
`proto/llamad/v1/llamad.proto`. Applications in all languages generate stubs from it. Then they
get streamed completion, chat, tool calls and embeddings without embedding llama.cpp. There is no
client library.

The project is small by design: a few thousand lines in approximately a dozen files, no framework
and few dependencies. This small size is a feature. After each change, the code must be as easy
to read as before the change.

The project is pre-release. No part of it has compatibility obligations, and the source is the
only reference for how the project operates. Breaking changes are welcome when they make a clear
improvement to the project. For each breaking change, explain the benefit and the effect on
callers.

## Layout

| Path | Role |
|---|---|
| `proto/llamad/v1/llamad.proto` | The wire contract, with its guarantees at the top of the file. No part of it depends on how the inference runs. `proto/CMakeLists.txt` generates it into the target `llamad_proto`. |
| `src/engine.{h,cpp}` | Wrapper over libllama: model load, tokenization, sampling, the generate loop, and streamed output that is safe for UTF-8 and stop strings. |
| `src/chat_format.{h,cpp}` | Chat layer: it renders messages and tools through the Jinja template of the model, makes the tool-call grammar and parses tool calls from generated text. |
| `src/wire.{h,cpp}` | Conversion, field by field, between wire messages and the types of the engine and the chat layer. It also makes the `finish` chunk, which enforces the tool-call invariant. |
| `src/service.{h,cpp}`, `src/main.cpp` | The gRPC service and the daemon: socket lifecycle, signals, and the map from errors to status codes. |
| `src/flags.h` | `cli::Args`, the small command-line reader that each binary uses to read its flags, and `cli::FlagError`. Each binary has its own if/else chain for its flags and its own `--help` text. |
| `src/engine_flags.h` | The context and offload flags that `llamad`, `engine_smoke` and `engine_bench` share, the `EngineConfig` that they set, their `--help` lines, and `print_device_table` for `--list-devices`. |
| `tests/` | Tests as plain executables, registered with CTest. They do not use a model file, and they share the `CHECK` macros in `check.h`. Two CLIs use a model to run the engine and the chat layer in their own process, without a daemon and without gRPC: `engine_smoke.cpp` tests behaviour, and `engine_bench.cpp` measures prefill and decode speed (`scripts/bench.sh`, docs/performance.md). |
| `third_party/llama.cpp` | Pinned submodule. It is not modified. |
| `models/` | Local GGUF files. Git ignores them, and they are not available in CI. |

## Build, test and run

```sh
./scripts/build.sh cpu                   # or gpu: Vulkan on Linux, Metal on macOS
./scripts/test.sh cpu                    # ctest in build-cpu/ (no model or daemon necessary)

./build-cpu/llamad --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --socket /tmp/llamad-dev.sock
grpcurl -plaintext -d '{"messages": [{"role": "user", "content": "Hello"}], "sampling": {"temperature": 0}}' \
    unix:///tmp/llamad-dev.sock llamad.v1.Llama/Chat
./build-cpu/tests/engine_smoke --help    # engine without the daemon
```

Each platform uses its own compiler and the gRPC from its package manager. For more information,
refer to docs/building.md.

Git ignores the `build*/` directories. If you start a daemon for a test, give it a private
`--socket` path. Stop the daemon when the test is complete.

## Design principles

These principles describe the architecture and the reasons for it. Use them as a starting point.
Change them when a better design makes a clear improvement to correctness, usability, clarity or
maintenance. Explain the tradeoffs, and keep the implementation, the tests and the documentation
consistent. Stay in the scope that the user intends, and use the approval that the task gives. If
a change makes that scope larger, ask the user first.

1. **Layers.** `engine.{h,cpp}` use only the public `llama.h` / `ggml-backend.h` C API. They
   contain no gRPC or protobuf types. `chat_format.cpp` is the only file that uses the `common`
   library of llama.cpp. This library is not stable and it is not a public API. The header of
   `chat_format` shows no llama.cpp types.

   All knowledge of gRPC and protobuf is in `service.*`, `wire.*` and `main.cpp`. `flags.h`
   includes only the standard library. The reason: a submodule update can break a maximum of one
   file, and tests can examine the engine and the chat layer without a daemon.
2. **Keep dependency changes easy to maintain.** llama.cpp is a pinned submodule that is not
   modified. Make changes in llamad, or send fixes to the upstream project. Thus, dependency
   upgrades stay simple.
3. **Change the two contracts deliberately.** `llamad.proto` and `engine.h` are the interfaces
   that the rest of the code uses. Applications see only the proto. In pre-release, these
   contracts can change incompatibly if the result is clearly better. Compare the benefit with the
   work that callers must do to update, and describe the two in the PR. Keep the definitions,
   conversions, tests, documentation and examples in agreement.

   When the proto changes, update `wire.cpp`, the guarantees at the top of the proto and
   docs/protocol.md in the same change. `tests/wire_test.cpp` names each field of a converted
   message that `wire.cpp` does not convert.
4. **No client library.** The proto is the integration surface for all languages. Some things
   only an application can know: which tools exist and what they do, the tool-call loop, and the
   type that a reply represents. Keep these things out of this repository.
5. **The daemon is stateless for each request.** It has no sessions and no registries. It keeps
   no state between calls that changes the meaning of a request. Clients send the history again.
   Tools are opaque data in each request, and the daemon never runs them or checks them.

   Only the KV cache of the engine stays after a request, and it is only a performance cache. The
   engine records the tokens that the cache holds. A request decodes only the part of its prompt
   after the prefix that it shares with these tokens. Each path out of `generate` keeps the
   record in agreement with the cache, or empties the two. The cache changes how long a prompt
   takes. It never changes which prompt runs.

   But a decode of the same tokens in a different batch split is not identical bit for bit. Thus,
   a greedy reply from a warm cache can be different from the reply of a daemon that just started.
6. **Stream shape.** A stream has zero or more `text` chunks, and then exactly one `finish` chunk
   with the finish reason and the stats. (`GenerateChunk` is a `oneof` of these two types.) Tool
   calls are complete and are in the `finish` chunk. `tool_calls` is not empty if and only if the
   reason is `TOOL_CALLS`. Text chunks contain only the content that the user sees. They are
   never empty, and they never contain tool-call markup, partial UTF-8 or part of a matched stop
   string.
7. **Local only.** The daemon creates a Unix socket with mode 0600. It has no TCP listener.

## How to make changes

**Aim for an excellent result.** Get correctness, a useful and coherent interface, clear code and
a design that is easy to maintain. Examine if the infrastructure is good for the problem. If you
can make the project clearly better, simplify, replace or remove the infrastructure. Complete each
justified improvement. Use tests or measurements to support your claims.

**Keep the scope purposeful.** Before you write code, write the result that you want in one or
two sentences. Select the smallest coherent change that gets all of that result. A large refactor
is correct when it removes a root cause or makes a clear improvement. If the diff becomes larger,
make sure that each part helps to get the result. Explain why the scope is necessary.

**Select the boring solution.** Use a function, not a class. Use a class, not a hierarchy. Repeat
a few clear lines two times, and do not make an abstraction with one and a half users. Use a
`std::vector` and a loop, not a cache that no measurement shows to be necessary. Use a heavier
solution only when a real, present requirement makes it necessary. Then tell what that
requirement is.

**Do not build for futures that you imagine.** Do not add configuration options that nobody asked
for. Do not add plugin points, or "extensible" interfaces with only one implementation. Do not
add generic helpers for one call site, or compatibility shims for callers that do not exist. When
the future comes, the code is small enough to change.

**Stay out of the weeds.** A task has a centre. Handle the edge cases that can really occur. Give
hypothetical edge cases a maximum of one line in your report. If you see an unrelated problem
during your work, tell the user about it. Do not fix it in the same change. If you put more effort
into a side issue than into the task, stop and examine your work again.

**Understand the code before you change it.** Read the code around the change first. This codebase
has a streaming holdback filter, RAII wrappers for llama.cpp handles, proto conversion in
`wire.cpp`, a flag reader and error types. Use these mechanisms again when they are good for the
task. Replace or simplify them when a different design makes the project clearly better. Remove
the code that a change replaces, so that the result stays coherent.

**Ask before you add a dependency.** The dependencies are llama.cpp, gRPC and Protobuf.
(nlohmann/json is part of the `common` library of llama.cpp, for the chat layer.) Tests are plain
executables with a `CHECK` macro.

**Write for the next reader.** People read code much more frequently than they write it. Usually,
the reader does not have your current context. Use names that tell what a thing is, straight-line
control flow, early returns instead of nesting, and small functions that do one thing. If code
must have a paragraph to justify how clever it is, make it less clever.

**Measure before you change code for speed.** The hot path is in llama.cpp, not in llamad. Keep
the work for each token cheap and clear. Do not make code less readable for speed without a
number that shows that the speed is important.

## Code style

Use the style of the file that you change. Consistency is more important than preference.

- Use C++23, built by the compiler of each platform: GCC 13 or Clang 18 on Linux, Apple Clang 16
  on macOS. Use only what all three standard libraries supply. For example, libc++ has no
  floating-point `std::from_chars` before LLVM 20. Build with `-DLLAMAD_WARNINGS_AS_ERRORS=ON`.
- Indent with 4 spaces. Write `const T & name` and `T * name`, with spaces around `&` and `*`.
- Align declarations and assignments where the code around them is aligned.
- Put `/*name*/` comments on literal arguments when their meaning is not clear: `tokenize(text, /*add_special*/ false, ...)`.
- Use anonymous namespaces for helpers in one file. Use pimpl where a header must hide a
  dependency.
- Use RAII for each llama.cpp / C handle, so that each path (exceptions also) releases it.
- Errors that callers can act on have types (`EngineError`, `ChatFormatError`). `service.cpp` maps
  them to gRPC status codes: caller mistakes → `INVALID_ARGUMENT`, a missing capability →
  `FAILED_PRECONDITION`, all other errors → `INTERNAL`.
- The daemon writes one `[llamad] ...` log line for each request to stderr, and writes nothing to
  stdout.

## Comments and documentation

- Comments explain **why**: a constraint, a consequence that is not clear, or a reason that the
  simpler approach is wrong. Do not describe what the next line clearly does. Use the same comment
  density as the code around it: short, with a purpose.
- **Describe the code as it is, in the present tense.** Do not write "used to", "no longer",
  "now", "new", "behaviour change", "previously" or migration notes in comments, the README or
  this file. When behaviour changes, write the affected text again as if it was always true.
  History goes in commit messages.
- Keep `README.md` correct. Update it in the same change that makes it incorrect: flags, defaults,
  behaviour and examples.
- Write `README.md`, `docs/*.md` and this file in ASD-STE100 Simplified Technical English. Use
  approved words, and the technical names and technical verbs of this domain. Use one word for one
  meaning. Write a maximum of 20 words in a procedural sentence, 25 words in a descriptive
  sentence, and 6 sentences in a paragraph. Use the active voice and simple tenses. Do not use the
  -ing form of a verb, phrasal verbs or semicolons. Write instructions in the imperative, with the
  condition first.

## Verify a change

Make the effort agree with the risk. Report the checks that you ran.

- For all changes: make sure that the project builds without new warnings, and that `ctest`
  passes.
- For changes to the chat layer or to parsers: add or extend a case in
  `tests/chat_format_test.cpp`. These tests send canned model output through a real template
  without a GGUF, so they are cheap. Where these tests can get to the logic, use them and not
  manual checks.
- For wire changes: `tests/wire_test.cpp` checks the conversion in the two directions and the
  field list of each converted message. Extend it with the field.
- For engine or service changes: test the real path. `engine_smoke` tests only the engine
  (`--stop`, `--cancel-after`, `--grammar-file`, `--chat --demo-tool`). A daemon tests the full
  stack. Send requests to it through grpcurl, or through stubs generated from the proto in any
  language (a tool round, a `response_json_schema` reply, a cancelled stream).

  Use `--temp 0` for repeatable output. Only a cold KV cache gives exactly the same output again
  (design principle 5). Thus, compare first requests to processes that just started.
  `tests/engine_test.cpp` checks the prompt-cache bookkeeping that does not use a model.

  The preferred local smoke-test model is `models/qwen2.5-0.5b-instruct-q4_k_m.gguf`. Use a
  stronger model only if the behaviour must have one. Give the path of the model to
  `./scripts/smoke-test.sh cpu` or `./scripts/smoke-test.sh gpu`. The script must have a model
  argument. Model files are local, and git ignores them.
- For a change for speed: measure it with `./scripts/bench.sh`, and compare it with the build
  before the change in the same session. Put the two numbers in the commit message
  (docs/performance.md).
- CI builds CPU-only with GCC and Clang on Linux and with Apple Clang on macOS, with warnings as
  errors. Then it runs the tests. It does not test GPU execution or load a model. Thus, only local
  runs verify the inference paths. If this applies to your change, say so. Do not imply coverage
  that CI does not give.
- Report the results clearly. Also report what you did not verify or could not verify.

## Git

- Work on a branch. `main` must always build. Commit and push only when the user asks.
- Make commits feature-sized. Each commit builds alone, and the history stays linear
  (fast-forward).
- Message style: a specific subject line (`Area: what changed`), and then a body. The body
  explains in prose what changed and why. Include measurements if they were the reason for the
  change.
