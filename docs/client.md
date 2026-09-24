# The client library

## Linking

```cmake
include(FetchContent)
FetchContent_Declare(llamad
    GIT_REPOSITORY https://github.com/crybo-rybo/llamad.git
    GIT_TAG        main                    # better, a commit
    GIT_SUBMODULES "")                     # the client needs nothing from llama.cpp
FetchContent_MakeAvailable(llamad)

target_link_libraries(myapp PRIVATE llamad::client)
```

Included by another project, with `FetchContent` or `add_subdirectory`, llamad builds in
client-only mode (`LLAMAD_CLIENT_ONLY`, on unless llamad is the top-level project): the
client, `llamad::client`, and the generated protocol code, `llamad::proto`, and nothing else.
No llama.cpp, so no submodules to clone; no daemon, no C compiler, and no tests registered with
your CTest. Your build type and compile-commands settings are left alone.

It still needs, found through their CMake configs:

- gRPC and Protobuf. On macOS, the ones `scripts/build-deps-macos.sh` builds, with its
  `build-deps/prefix` on your `CMAKE_PREFIX_PATH` (see [building.md](building.md)).
- nlohmann/json: your project's `nlohmann_json::nlohmann_json` target if it defines one before
  including llamad, or else an installed package (`pacman -S nlohmann-json`,
  `brew install nlohmann-json`). There is only ever one copy of the header in your build.

`llamad/client.h` exposes no gRPC or protobuf types, so your build needs neither on its
include path. It does reflect over your own tool functions, so linking `llamad::client` puts
C++26, `-freflection` and nlohmann's include directory on whatever includes it, and your code
is compiled with GCC 16 or later too.

`llamad::proto` is the service itself, generated from `llamad.proto`, for a test that needs a
daemon but no model: implement `llamad::v1::Llama::Service` with scripted replies, serve it on
a private socket and point a `Client` at it. `tests/consumer/` is such a project, built by CI
against every commit.

## Streaming chat

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

The callback receives user-visible text only: never tool-call markup, never partial UTF-8,
never part of a matched stop string. The result carries the finish reason and the stats from
the stream's final chunk.

## Stopping a call

A call blocks its thread until the stream ends, and returning `false` from the callback can
cancel it only when text arrives. To call it off from somewhere else, such as a cancel button or
a shutdown signal, or to bound it in time, pass `CallOptions` as the last argument:

```cpp
std::stop_source stop;                      // stop.request_stop() from any thread

auto result = client.chat(history, params, on_chunk,
                          {.stop = stop.get_token(), .timeout = std::chrono::seconds(30)});
```

- A stop cancels the call even when no text is arriving: while the daemon reads a long prompt,
  while it serves another client first, during a tool-call round. A streaming call returns
  `FinishReason::Cancelled`, unless its final chunk had already arrived, whose reason stands.
  `get_model_info`, `tokenize` and `embed` throw `RpcError` with code `CANCELLED` (1).
- The timeout applies to each RPC, from its start. When it runs out the call throws `RpcError`
  with code `DEADLINE_EXCEEDED` (4), so a `get_model_info` with a short timeout is a health
  check that cannot hang.
- The tool loop takes `CallOptions` after `max_rounds`. The stop covers the whole loop, and no
  tool runs once it has been requested; the timeout applies to each round.

`std::jthread` passes its function a `std::stop_token`, so a call made on one with that token
stops when the thread is asked to. `llamad-chat` stops a reply on Ctrl-C this way.

## Tool calling

The daemon is a formatter and a parser, not a tool registry. Tools are opaque
per-request data, a name, a description and a JSON Schema, carried alongside the
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

### Tools as C++ functions

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

A tool that needs state, such as an open document or a game world, is a member function,
registered with the object it is called on. The tool's name is the member's name:

```cpp
class Inventory {
public:
    [[=desc{"Take items out of stock and say how many are left."}]]
    int take([[=desc{"Item name"}]] std::string item, int count);
};

Inventory inventory;
tools.add<^^Inventory::take>(inventory);
```

The tool set holds a reference to `inventory`, so the object must outlive the tool set and
every copy of it.

A tool returning `std::string` is handed to the model as it is; any other return type
is written as JSON, as are the arguments read out of a call. A function with no parameters
is a tool that takes no arguments: its schema is an object with no properties, and the
model calls it with `{}`. A tool that does not exist, arguments that do not parse and an
exception thrown by the tool all become an `{"error":"..."}` result the model can recover
from. The loop stops after eight rounds of tool calls, which the caller sees as a
`ToolCalls` result; that limit is `chat`'s last argument and must be positive.

`llamad-chat --demo-tools` is that worked through end to end
(`client/examples/chat_cli.cpp`): it offers one `get_current_time` tool and answers
with it.

### JSON conversions

Tool arguments use checked C++ conversions: integer arguments must be integers in range,
floating-point arguments must fit their type, and enums use their enumerator names.
Absent or null optional arguments are unset; unknown object keys are ignored. JSON parsing
and serialization use nlohmann/json. Non-finite numbers in tool results are errors, and bytes
that are not UTF-8 are written as U+FFFD. `json::read` and `json::write` report all of this as
`json::Error`, whose message is what the model reads in the `{"error":"..."}` result.

### Running the loop yourself

The `chat` overload taking a `std::vector<Tool>` is the same thing with the loop left
to the caller: it takes the name, description and JSON Schema as strings, and returns
each round of `tool_calls` for the caller to answer.

## Typed replies

`chat<T>` asks for a struct instead of prose:

```cpp
enum class Confidence { low, medium, high };

struct Verdict {
    [[=desc{"whether the message is spam"}]]         bool                     spam;
    [[=desc{"short reasons, most important first"}]] std::vector<std::string> reasons;
    Confidence                                       confidence;
    std::optional<std::string>                       note;
};

auto reply = client.chat<Verdict>(history, params, on_chunk);
if (reply.value) { use(*reply.value); }
```

`json::schema<T>()` travels with the request, the daemon builds a grammar from it, and the
sampler can only produce a JSON object of that shape; the finished reply is read back with
`json::read`. `T` is an aggregate whose members `json.h` supports. The schema also goes into the
prompt, as a system instruction to reply with one matching JSON object, which is how the `desc`
annotations on `T` reach the model as property descriptions and can steer the answer.

- `value` is set when the reply is a complete JSON document: the model finished it, or a stop
  string matched after it. A reply cut short by `max_tokens`, by a cancel, or by a stop string
  matched inside the JSON is not whole, so `value` is empty and `result.reason` says which.
- The JSON still streams through the callback as it is generated, chunk by chunk, exactly as an
  ordinary reply does.
- Tools are not offered on a typed turn: the daemon refuses a schema alongside tools.
- Thinking is off on a typed turn. On a template that opens a `<think>` block the daemon asks
  for it closed, so the reply is the JSON object and nothing else.
- A complete reply that does not fit `T` throws `json::Error`. The grammar makes that a
  disagreement between the schema and the reader rather than a model mistake.

`llamad-chat --demo-json` is that worked through end to end
(`client/examples/chat_cli.cpp`): it asks for a verdict like this one on a single message,
streams the JSON and prints the fields.

## Embeddings

A daemon serving an embedding model answers `embed`, which takes a batch of texts and returns
one vector per text, in order:

```cpp
llamad::client::Client client("/tmp/embed.sock");

auto result = client.embed({"How do I bake sourdough?", "Sourdough needs a starter.", "Tax law"});
const std::vector<float> & query = result.embeddings[0].values;   // ModelInfo::n_embd values
```

Every vector is L2-normalised, so the dot product of two is their cosine similarity; there is no
option to get them unnormalised. `result.input_tokens` counts the tokens across every input.
`get_model_info()` says whether the daemon serves embeddings and how long the vectors are.

- A daemon serving a generative model refuses `embed`, and an embedding daemon refuses
  `generate` and `chat`, both with `RpcError` code `FAILED_PRECONDITION` (9).
- An empty batch, or an input longer than the daemon's per-input limit (at most 512 tokens),
  is `INVALID_ARGUMENT` (3), and no vectors come back.
- An instruction prefix a model expects on queries is part of the text you send.

`llamad-chat --embed` is that worked through (`client/examples/chat_cli.cpp`).

### Not supported

`tool_choice` (the model always decides), streamed argument deltas (calls are atomic),
reasoning separation (on an ordinary turn a model's `<think>` block, if any, is left in the
content; a typed turn turns thinking off), a typed reply whose root is not an object, and
partial structs during streaming (`value` arrives whole, at the end).
