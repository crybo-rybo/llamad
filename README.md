# llamad

A small daemon that loads one llama.cpp model once and serves it to local C++
applications over gRPC on a Unix domain socket. Applications link a tiny client
library and get streaming completions and chat without embedding llama.cpp, and
without paying the model load time in every process.

## Dependencies

llamad is written in C++26 and uses static reflection, so it needs GCC 16 or later. Clang and
Apple Clang do not implement reflection, which makes Linux with GCC the supported platform.

The client uses the header-only nlohmann/json library shipped in the pinned llama.cpp
submodule. It needs no separate package or runtime library.

Arch Linux:

```sh
pacman -S gcc grpc protobuf cmake ninja
```

## Build

```sh
git clone --recurse-submodules git@github.com:crybo-rybo/llamad.git
cd llamad
./scripts/build.sh cpu                      # builds into build-cpu/
./scripts/test.sh cpu                       # chat template, flags, wire contract, JSON and tool set tests
```

(If you already cloned without submodules: `git submodule update --init --recursive`.)

The build includes llama.cpp's `common` library, which the chat layer needs for Jinja
templates and tool-call parsing; it is the bulk of a first build. The tests need no
model file and no daemon: the chat-template ones render and parse against a template
checked into the submodule, the rest need nothing but the build.

GitHub Actions builds on Arch Linux (CPU-only) on every push and pull request, and can
also be started manually. The job builds the daemon, client, and engine smoke executable, and
runs the tests without a GPU or model download. GPU execution and inference with a real model
are not covered by CI.

## GPU

llama.cpp's GPU backends are enabled with their usual CMake flags. Vulkan is the
one tested here: it runs on Pascal cards (GTX 10xx), which CUDA 13 no longer targets.

```sh
pacman -S vulkan-headers spirv-headers vulkan-icd-loader shaderc
./scripts/build.sh gpu                  # cmake -DGGML_VULKAN=ON in build-gpu/
./scripts/test.sh gpu
./scripts/smoke-test.sh gpu /absolute/path/to/model.gguf
```

`test.sh` needs no model. `smoke-test.sh` accepts an absolute or working-directory-relative
model path; omitting it uses `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` under the repository root.

A working Vulkan driver for the GPU is also required. The smoke test fails if
inference falls back to the CPU; listing devices alone does not test model loading
or token generation.

Every discrete GPU is used by default: llama.cpp splits the model's layers
across them in proportion to each card's free memory, and ignores an integrated
GPU whenever a discrete one exists. Two 8 GB cards therefore hold a model that
fits on neither alone. The daemon prints one line per offload device at startup.

```sh
./build-gpu/llamad --list-devices                    # names, types, free/total memory
./build-gpu/llamad --model M --devices Vulkan0       # this card only
./build-gpu/llamad --model M --tensor-split 3,1      # 3:1 share, in device order
```

## Run the daemon

```sh
./build-cpu/llamad --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf
# [llamad] listening on unix:/run/user/1000/llamad.sock
```

Options: `--socket PATH` (default `$XDG_RUNTIME_DIR/llamad.sock`, else
`/tmp/llamad-<uid>.sock`), `--ctx N` (4096), `--ngl N` (99), `--threads N` (0 =
auto, half the hardware threads). The socket is created mode 0600, so only your
user can talk to it.
SIGINT/SIGTERM shut the daemon down and remove the socket.

## Chat from the terminal

```sh
./build-cpu/client/llamad-chat                       # interactive REPL
./build-cpu/client/llamad-chat --once "Hello" --temp 0
./build-cpu/client/llamad-chat --demo-tools          # with one built-in tool, see below
```

Also accepts `--socket`, `--system TEXT`, `--seed N`, `--max-tokens N`. Ctrl-C
cancels the reply in progress; Ctrl-C or Ctrl-D at the prompt quits.

`./build-cpu/tests/engine_smoke` drives the engine in-process, with no daemon and no gRPC:
`--chat` renders the prompt through the model's chat template, `--demo-tool`
adds the same `get_current_time` tool to that rendering, and `--grammar-file
PATH` constrains generation with a GBNF file of your own.

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
on its include path. It does reflect over your own tool functions, so linking
`llamad_client` puts C++26, `-freflection` and nlohmann's include directory on
whatever includes it.

## Tool calling

The daemon is a formatter and a parser, not a tool registry. Tools are opaque
per-request data — a name, a description and a JSON Schema — carried alongside the
messages, exactly like the history. The daemon never executes a tool, never checks
that one exists, and remembers nothing between requests. **The client owns the
execute-and-resend loop.**

What the daemon does do: render the tools into the prompt the way the model was
trained to see them, constrain the arguments to the tool's JSON Schema while they
are being generated, and parse the model's output back into structured calls.

- Tool calls arrive **whole, on the final chunk**, together with
  `FINISH_REASON_TOOL_CALLS`. There are no argument deltas.
- Streamed `text` chunks only ever carry user-visible content. Tool-call markup
  (`<tool_call>` and friends) never reaches the client.
- `tool_calls` is non-empty **iff** the finish reason is `TOOL_CALLS`. A reply cut
  short by `max_tokens` or by a cancel reports `LENGTH`/`CANCELLED` and no calls.
- Arguments are always a complete, valid JSON object, because a grammar built from
  the schema is what the sampler was allowed to produce.

The message sequence for one tool round is:

```
user                                    "what time is it in Tokyo?"
assistant { tool_calls: [...] }         finish_reason = TOOL_CALLS
tool      { tool_call_id, content }     one per call, the result, as a string
assistant "It is 06:28 in Tokyo."       finish_reason = EOG
```

In an application a tool is a C++ function. `ToolSet::add` reads its name off the
identifier, its descriptions off `desc` annotations and the argument schema off the
parameter list; the `chat` overload that takes a `ToolSet` runs the loop above, parses
each call's arguments, invokes the function and sends the result back.

```cpp
using llamad::client::desc;

[[=desc{"Get the current date and time in a given IANA timezone."}]]
std::string get_current_time([[=desc{"IANA timezone, e.g. Europe/Paris"}]] std::string timezone);

llamad::client::ToolSet tools;
tools.add<^^get_current_time>();

std::vector<llamad::client::ChatMessage> history = {{"user", "What time is it in Tokyo?"}};

auto result = client.chat(history, tools, params, [](const std::string & text) {
    std::fputs(text.c_str(), stdout);       // user-visible content only
    return true;
});                                         // history holds every turn the answer took
```

A tool returning `std::string` is handed to the model as it is; any other return type
is written as JSON, as are the arguments read out of a call. A tool that does not
exist, arguments that do not parse and an exception thrown by the tool all become an
`{"error":"..."}` result the model can recover from. The loop stops after eight rounds
of tool calls, which the caller sees as a `ToolCalls` result; that limit is `chat`'s last
argument and must be positive.

Tool arguments use checked C++ conversions: integer arguments must be integers in range,
floating-point arguments must fit their type, and enums use their enumerator names.
Absent or null optional arguments are unset; unknown object keys are ignored. JSON parsing
and serialization use nlohmann/json. Non-finite numbers in tool results are errors, and bytes
that are not UTF-8 are written as U+FFFD. `json::read` and `json::write` report all of this as
`json::Error`, whose message is what the model reads in the `{"error":"..."}` result.

`llamad-chat --demo-tools` is that worked through end to end
(`client/examples/chat_cli.cpp`): it offers one `get_current_time` tool and answers
with it.

```sh
./build-cpu/client/llamad-chat --demo-tools --once "What time is it in Tokyo right now?" --temp 0
# [tool] get_current_time(Asia/Tokyo) -> 2026-09-21 08:44:44 JST
# The current time in Tokyo is 2026-09-21 08:44:44 JST.
# [stats] finish=eog prompt_tokens=461 completion_tokens=52 ...
```

The `chat` overload taking a `std::vector<Tool>` is the same thing with the loop left
to the caller: it takes the name, description and JSON Schema as strings, and returns
each round of `tool_calls` for the caller to answer.

Not supported: `tool_choice` (the model always decides), streamed argument deltas
(calls are atomic), and reasoning separation (a model's `<think>` block, if any, is
left in the content).

## Design notes

- **No TCP listener.** gRPC here is HTTP/2 over a Unix domain socket. There is
  no port to firewall; access control is the socket's file permissions (0600).
- **llama.cpp is an unmodified pinned submodule**, not a fork. `src/engine.cpp`
  uses only the public `llama.h` C API. `src/chat_format.cpp` is the one file that
  touches llama.cpp's `common` library — its Jinja chat templates, its per-model
  tool-call parsers and its JSON-Schema-to-grammar converter — which is unstable
  and not a public API, so a submodule bump can break at most that single file.
- **v1 serves one generation at a time.** The engine serializes `generate`, so
  concurrent clients queue rather than sharing the context. Each request starts
  from an empty KV cache.
- **Chat renders the model's own Jinja template**, the one stored in the GGUF,
  through llama.cpp's `common` library — the same path `llama-server` takes. So
  whatever scaffolding the model was trained on (tool blocks, role markers, its
  default system prompt) is what it actually sees. A model with no template, or
  with one that will not parse, still serves `Generate`, `Tokenize` and
  `GetModelInfo`; only `Chat` is refused, with `FAILED_PRECONDITION`.

  The template is in charge of the prompt, defaults included: Qwen2.5, for
  instance, injects its own default system prompt when the client sends no
  `system` message.
