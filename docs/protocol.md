# Integration with llamad

An application uses one file to communicate with llamad: `proto/llamad/v1/llamad.proto`. There is
no client library to link. The top of that file gives the guarantees that each response keeps.
This page shows how to use them.

To use the daemon, do these steps:

1. Generate stubs for your language with `protoc`.
2. Open a gRPC channel to the Unix socket of the daemon.
3. Call the `llamad.v1.Llama` service.

## Connect to the daemon

Use the `llamad.proto` of the llamad version that you run. Each GitHub release attaches this
file. Copy it into your project. As an alternative, add this repository as a pinned submodule,
and give its `proto/` directory to `protoc`. Then generate the code as for all other gRPC
services:

```sh
# C++
protoc -I proto --cpp_out=gen --grpc_out=gen \
    --plugin=protoc-gen-grpc="$(command -v grpc_cpp_plugin)" llamad/v1/llamad.proto

# Python
python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. llamad/v1/llamad.proto
```

The gRPC target is `unix:` and then the socket path. The default socket path is
`$XDG_RUNTIME_DIR/llamad.sock`. If `XDG_RUNTIME_DIR` is not set, the default is
`/tmp/llamad-<uid>.sock`. The `--socket` flag sets a different path. The socket has mode 0600.
Thus, only the user that started the daemon can connect. The channel is plaintext. There is no
TLS, and the only authentication is the permissions of the socket.

```python
import grpc
from llamad.v1 import llamad_pb2 as pb, llamad_pb2_grpc as rpc

stub = rpc.LlamaStub(grpc.insecure_channel("unix:/run/user/1000/llamad.sock"))

request = pb.ChatRequest(messages=[pb.ChatMessage(role="user", content="Name three primes.")],
                         sampling=pb.SamplingParams(temperature=0))
for chunk in stub.Chat(request):
    kind = chunk.WhichOneof("chunk")
    if kind == "text":
        print(chunk.text, end="", flush=True)
    elif kind == "finish":
        print("\n", pb.FinishReason.Name(chunk.finish.reason), chunk.finish.stats.completion_tokens)
    # Skip all other chunks, for example progress chunks.
```

The daemon also serves gRPC server reflection and the standard health service. Thus, a tool can
call the daemon without a copy of the proto. An example of such a tool is
[grpcurl](https://github.com/fullstorydev/grpcurl):

```sh
grpcurl -plaintext unix:///run/user/1000/llamad.sock list
grpcurl -plaintext unix:///run/user/1000/llamad.sock llamad.v1.Llama/GetModelInfo
grpcurl -plaintext -d '{"messages": [{"role": "user", "content": "Name three primes."}]}' \
    unix:///run/user/1000/llamad.sock llamad.v1.Llama/Chat
```

`GetModelInfo` identifies the daemon and the model that answer. `llamad_version` is the version
of the daemon. `model_file` is the file name of the GGUF, without its directory. `model_name` is
the `general.name` metadata of the GGUF. It is empty if the GGUF does not have this metadata.
Record these values with your logs or benchmark results.

## Streams

`Generate` and `Chat` send a stream of `GenerateChunk` messages. Each chunk is one of these three
types:

- `text`: The next part of the output that the user sees. It is never empty. It never contains a
  partial UTF-8 sequence, part of a matched stop string or tool-call markup.
- `progress`: A sign that the request is alive. It contains no output. It gives the stage of the
  request (`stage`) and the token counts at that time.
- `finish`: The reason that the generation stopped (`reason`), the token counts and times
  (`stats`), and the tool calls, if there are tool calls. It is always the last chunk of a
  successful stream, and there is exactly one.

A successful stream has zero or more `text` and `progress` chunks, in any sequence. Then it has
one `finish` chunk. Skip each chunk that is not `text` or `finish`, if you do not use it. A later
version can add other types of chunks.

If a stream ends without a `finish` chunk, the call was cancelled or an error occurred. The gRPC
status tells which. Some errors occur before the generation starts, for example an empty
conversation or a bad schema. Such an error ends the stream with its status and with no chunks.

To stop a generation, cancel the call with your gRPC library, or set a deadline. The daemon stops
the request at the next step of the engine. A step is one token or one batch of the prompt. A
request in the queue stops in 250 ms or less. The stream then ends with the status `CANCELLED` or
`DEADLINE_EXCEEDED`, and with no `finish` chunk. The prompt tokens that the engine decoded stay in
the KV cache. Thus, if you send the same prompt again, the daemon does not decode them again.

### Progress and idle timeouts

The daemon sends a `progress` chunk only after approximately one second without a chunk. Thus, a
fast request gets no `progress` chunks. A slow request gets approximately one chunk each second.
These are the stages:

| Stage | Meaning |
|---|---|
| `STAGE_QUEUED` | The request waits for another request in the queue. |
| `STAGE_PROMPT` | The engine decodes the prompt. `ready_prompt_tokens` is the number of prompt tokens in the KV cache. |
| `STAGE_GENERATING` | The engine generates tokens, but the daemon holds back their text. An example is a tool call. |

The daemon can send a `progress` chunk only between two steps of the engine. The engine cannot
stop in a step. The longest step is one batch of the prompt, which has a maximum of 2048 tokens.
On a CPU, one batch can take 10 seconds or more. Thus, the longest time without a chunk is
approximately one second plus the time of one batch. If you use an idle timeout, make it longer
than this time. To find the time of one batch, look at `prompt_ms` for a cold prompt of 2048
tokens. A batch late in a long prompt takes more time than the first batch.

`GenerateStats.cached_prompt_tokens` is the number of prompt tokens that came from the KV cache.
These tokens are at the start of the prompt, and the previous request decoded them. If you send a
conversation again with a new turn at the end, the cache usually holds all the tokens before that
turn.

## Tool calls

The daemon formats and parses tool calls. It is not a tool registry. Each request contains its
tools as data. Each tool has a name, a description and a JSON Schema for its arguments. The
schema must be a JSON object. If the schema has a `type`, the type must be `"object"`. An empty
schema means that the tool has no arguments. The daemon rejects other schemas with
`INVALID_ARGUMENT`. The daemon does these steps:

- It renders the tools into the prompt, in the format that the model was trained on.
- During sampling, it limits the arguments to the schema.
- It parses the output of the model into tool calls.

The daemon never runs a tool, and never makes sure that a tool exists. It keeps no data between
requests. **Your application must run the tools and send the results in a new request.**

- Tool calls are complete and are in the `finish` chunk, with `reason` =
  `FINISH_REASON_TOOL_CALLS`. There are no partial arguments. The markup of the tool calls is
  never in a `text` chunk.
- `finish.tool_calls` is not empty if and only if the reason is `TOOL_CALLS`. If `max_tokens`
  stops a reply, the reason is `LENGTH`, and there are no tool calls. A cancelled reply has no
  `finish` chunk.
- `arguments_json` is always a complete JSON object. For a tool without parameters, it is `{}`.
- If one call is not complete, or its arguments are not a JSON object, the daemon sends no calls
  for the reply. The text before the calls still comes. Thus, the reply can have no text and no
  calls, with the reason `EOG` or `STOP`.
- Each `id` is unique in the conversation. It is different from each id in the history of the
  request, and from each other call in the reply. The daemon makes an id from 32 random
  characters. Thus, ids from different requests and daemon runs are also different in practice.

This example shows one round:

```
request:  messages = [user "What time is it in Tokyo?"], tools = [get_current_time]
reply:    finish { reason: TOOL_CALLS, tool_calls: [{id: "abc", name: "get_current_time",
                                                     arguments_json: "{\"timezone\":\"Asia/Tokyo\"}"}] }

request:  messages = [user "What time is it in Tokyo?",
                      assistant { content: <the text of that reply>, tool_calls: [<the call, as received>] },
                      tool { tool_call_id: "abc", content: "2026-10-01 09:00 JST" }],
          tools = [get_current_time]
reply:    text "The current time in Tokyo is 09:00." ... finish { reason: EOG }
```

Send one `tool` message for each call. Put the `id` of the call in that message. To the daemon,
the result is an opaque string. JSON is a good format for the result. An `{"error": ...}` object
is also good, because the model can read the error and continue. Set a limit on the number of
rounds in your application. The daemon does not stop a model that calls tools again and again.

## Typed replies

Set `ChatRequest.response_json_schema` to get a JSON object as the reply, and not prose. The
daemon makes a grammar from the schema. Thus, the sampler can only make an object of that shape.
The daemon also puts the schema into the prompt as a system instruction. Thus, the model gets the
property descriptions.

- The JSON comes in usual `text` chunks. Concatenate the chunks, then parse the result.
- The object is complete when `finish.reason` is `EOG`. It is also complete when the reason is
  `STOP` and the stop string matched after the object. `LENGTH` tells that the object is not
  complete. A cancelled reply has no `finish` chunk.
- You cannot use a schema and tools in the same request. The daemon rejects the schema `{}`,
  because it puts no limit on the reply. The two errors are `INVALID_ARGUMENT`.
- A schema turn always has reasoning off, as with `REASONING_OFF`. Thus, the text is only the
  JSON object. Refer to [Reasoning](#reasoning).

## Reasoning

Some models write reasoning before the reply, for example in a `<think>` block. The chat template
of the model opens this block. Set `ChatRequest.reasoning` to control it:

- `REASONING_DEFAULT`, or no value: The daemon does not change the template. If the model writes
  reasoning, the reasoning comes in `text` chunks, as part of the reply.
- `REASONING_OFF`: The daemon tells the template to turn off reasoning. The result depends on the
  template.

| Template | Result of `REASONING_OFF` |
|---|---|
| It obeys the switch, for example Qwen3. | The template closes the block in the prompt. Thus, the model replies without reasoning and uses no tokens for it. |
| It has no reasoning, for example Qwen2.5. | The template ignores the field. The prompt and the reply do not change. |
| It always opens the block, for example DeepSeek-R1-Distill-Llama. | The model still writes reasoning. The daemon does not send it in `text`, but `completion_tokens` counts it. If the model does not close the block, the reply has no text. |

`REASONING_OFF` does not filter the reply. If the model opens a new block after the closed
block, that block comes in `text`.

You can use `REASONING_OFF` with tools. A request with `response_json_schema` always has reasoning
off. Thus, `REASONING_OFF` with a schema changes nothing.

## Embeddings

The GGUF of an embedding model declares a pooling type. With an embedding model, the daemon
answers `Embed` and rejects `Generate` and `Chat`. With all other models, the daemon does the
opposite. `GetModelInfo` tells which type of model the daemon serves (`serves_embeddings`) and
the length of the vectors (`n_embd`).

`Embed` returns one vector for each input, in the same sequence as the inputs. Each vector is
L2-normalised. Thus, the dot product of two vectors is their cosine similarity. The daemon embeds
each input separately, in one decode. An input can have a maximum of 512 tokens. A short `--ctx`
or a short training context makes this limit lower. If an input is longer, the daemon embeds no
input and returns `OUT_OF_RANGE`. Some models expect an instruction prefix on
queries, for example `Instruct: ...\nQuery:` for Qwen3-Embedding. Put this prefix in the text
that you send.

## Errors

| Status | Meaning |
|---|---|
| `OUT_OF_RANGE` | An input has more tokens than the model can take. Examples: a `Generate` or `Chat` prompt that does not fit the context, an `Embed` input that is longer than the limit. A shorter input can succeed. Refer to [Inputs that are too long](#inputs-that-are-too-long). |
| `INVALID_ARGUMENT` | The request is not correct, and the application can correct it. Examples: no messages, schema JSON that does not parse, a schema with tools. |
| `RESOURCE_EXHAUSTED` | The request message is larger than `ModelInfo.max_request_bytes`. gRPC refuses it before the daemon reads it. Send a smaller request, for example a smaller `Embed` batch. As an alternative, start the daemon with a larger `--max-request-bytes`. Your gRPC library also refuses a response that is larger than its receive limit, often 4 MiB, with this status. A large `Embed` batch can cause this. Increase that limit in the channel options. |
| `FAILED_PRECONDITION` | The loaded model cannot do this request. Examples: `Chat` without a usable chat template, `Embed` on a generative model, `Generate` or `Chat` on an embedding model. |
| `CANCELLED`, `DEADLINE_EXCEEDED` | The application cancelled the call, or its deadline passed. A `Generate` or `Chat` stream then has no `finish` chunk. |
| `INTERNAL` | All other errors, for example a decode error. The message tells what occurred. |

The daemon serves one generation at a time. Thus, concurrent requests wait in a queue. A request
in the queue gets `progress` chunks with the stage `STAGE_QUEUED`. A deadline includes the time in
the queue and the time of the generation. If you cancel a request in the queue, it leaves the
queue in 250 ms or less, and the daemon decodes nothing for it.

### Inputs that are too long

A conversation becomes longer with each turn. Thus, its prompt can become too long for the
context. The daemon then returns `OUT_OF_RANGE`, and not `INVALID_ARGUMENT`. Use the status code
to find this error. Do not parse the message. The message gives the token count and the limit
for a person to read.

To recover, remove the oldest turns and send the request again. Keep the system message. Remove
an assistant tool call together with its tool results. The daemon tokenizes a prompt and checks
its length before the request goes into the queue. Thus, a prompt that is too long fails
immediately, also while another request uses the model. A loop that removes a turn and tries
again is fast.

A `Generate` prompt must have fewer tokens than `ModelInfo.n_ctx`. To count the tokens of a
prompt, call `Tokenize` with `add_special` and `parse_special` set to true. A `Chat` prompt also
contains the text of the chat template, and `Tokenize` cannot count it. For `Chat`, remove turns
until the request succeeds.

An `Embed` input that is too long also fails with `OUT_OF_RANGE`, and the daemon embeds no input
of that request. Make the input shorter, or split it into parts.

## Request ids

The daemon gives each call an id. It sends the id in the initial metadata of the call, with the
key `x-request-id`. The daemon writes the same id in its log line for the call
([daemon.md](daemon.md)). Thus, if a call fails, record the id with the error. The id then
identifies the log line of that call.

- The id is in the metadata of a successful call and of a call that fails with any status.
- `Generate` and `Chat` send the id before the first chunk. Thus, a request in the queue already
  has its id.
- The id is an opaque string. Do not parse it. The daemon makes it from 64 random bits, and not
  from a counter. Thus, ids do not repeat after a restart of the daemon.
- The daemon does not read an `x-request-id` that the client sends.

```python
call = stub.Chat(request)
request_id = dict(call.initial_metadata()).get("x-request-id")  # waits for the headers only

try:
    info, call = stub.GetModelInfo.with_call(pb.GetModelInfoRequest())
    request_id = dict(call.initial_metadata()).get("x-request-id")
except grpc.RpcError as error:
    request_id = dict(error.initial_metadata() or ()).get("x-request-id")
```

Some calls have no id and no log line. The gRPC transport refuses some calls before the service
gets them, for example a message that is too large or an unknown method. The health service and
the reflection service also give no id.
