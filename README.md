# llamad

A small daemon that loads one llama.cpp model once and serves it to local C++
applications over gRPC on a Unix domain socket. Applications link a tiny client
library and get streaming completions and chat without embedding llama.cpp, and
without paying the model load time in every process.

## Dependencies

llamad is written in C++26 and uses static reflection, so it needs GCC 16 or later. Clang and
Apple Clang do not implement reflection, which makes Linux with GCC the supported platform.

Arch Linux:

```sh
pacman -S gcc grpc protobuf cmake ninja
```

## Build

```sh
git clone --recurse-submodules git@github.com:crybo-rybo/llamad.git
cd llamad
cmake -S . -B build -G Ninja
cmake --build build -j
ctest --test-dir build --output-on-failure    # chat template / tool-call parser tests
```

(If you already cloned without submodules: `git submodule update --init --recursive`.)

The build includes llama.cpp's `common` library, which the chat layer needs for Jinja
templates and tool-call parsing; it is the bulk of a first build. The tests need no
model file: they render and parse against a template checked into the submodule.

GitHub Actions builds on Arch Linux (CPU-only) on every push and pull request, and can
also be started manually. The job builds the daemon, client, and engine smoke executable, and
runs the tests without a GPU or model download. GPU execution and inference with a real model
are not covered by CI.

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
./build/client/llamad-chat --demo-tools          # with one built-in tool, see below
```

Also accepts `--socket`, `--system TEXT`, `--seed N`, `--max-tokens N`. Ctrl-C
cancels the reply in progress; Ctrl-C or Ctrl-D at the prompt quits.

`engine_smoke` drives the engine in-process, with no daemon and no gRPC:
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
on its include path.

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

`llamad-chat --demo-tools` is that loop, worked through end to end
(`client/examples/chat_cli.cpp`): it offers one `get_current_time` tool, runs it
whenever the model asks, and feeds the result back.

```sh
./build/client/llamad-chat --demo-tools --once "What time is it in Tokyo right now?" --temp 0
# [stats] finish=tool_calls prompt_tokens=194 completion_tokens=23 ...
# [tool] get_current_time({"timezone": "Asia/Tokyo"}) -> {"timezone":"Asia/Tokyo","time":"..."}
# The current time in Tokyo (Japan Standard Time) is ...
```

From your own code it is the same loop:

```cpp
std::vector<llamad::client::Tool> tools = {
    {"get_current_time", "Get the current time in a given IANA timezone.",
     R"({"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]})"}};

std::vector<llamad::client::ChatMessage> history = {{"user", "What time is it in Tokyo?"}};

for (;;) {
    std::string reply;
    auto result = client.chat(history, tools, params, [&](const std::string & text) {
        reply += text;                      // user-visible content only
        return true;
    });
    if (result.reason != llamad::client::FinishReason::ToolCalls) {
        break;                              // ordinary answer; reply holds it
    }
    history.push_back({"assistant", reply, result.tool_calls, ""});
    for (const llamad::client::ToolCall & call : result.tool_calls) {
        history.push_back({"tool", run_my_tool(call.name, call.arguments_json), {}, call.id});
    }                                       // call.id goes in tool_call_id
}
```

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
