# Integration with llamad

An application uses one file to communicate with llamad: `proto/llamad/v1/llamad.proto`. There is
no client library to link. The top of that file gives the guarantees that each response keeps.
This page shows how to use them.

To use the daemon, do these steps:

1. Generate stubs for your language with `protoc`.
2. Open a gRPC channel to the Unix socket of the daemon.
3. Call the `llamad.v1.Llama` service.

## Connect to the daemon

Copy `llamad.proto` into your project. As an alternative, add this repository as a pinned
submodule, and give its `proto/` directory to `protoc`. Then generate the code as for all other
gRPC services:

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
    if chunk.WhichOneof("chunk") == "text":
        print(chunk.text, end="", flush=True)
    else:
        print("\n", pb.FinishReason.Name(chunk.finish.reason), chunk.finish.stats.completion_tokens)
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

## Streams

`Generate` and `Chat` send a stream of `GenerateChunk` messages. Each chunk is one of these two
types:

- `text`: The next part of the output that the user sees. It is never empty. It never contains a
  partial UTF-8 sequence, part of a matched stop string or tool-call markup.
- `finish`: The reason that the generation stopped (`reason`), the token counts and times
  (`stats`), and the tool calls, if there are tool calls. It is always the last chunk of a
  successful stream, and there is exactly one.

If a stream ends without a `finish` chunk, the call was cancelled or an error occurred. The gRPC
status tells which. Some errors occur before the generation starts, for example an empty
conversation or a bad schema. Such an error ends the stream with its status and with no chunks.

To stop a generation, cancel the call with your gRPC library, or set a deadline. The daemon stops
the generation at the next token.

`GenerateStats.cached_prompt_tokens` is the number of prompt tokens that came from the KV cache.
These tokens are at the start of the prompt, and the previous request decoded them. If you send a
conversation again with a new turn at the end, the cache usually holds all the tokens before that
turn.

## Tool calls

The daemon formats and parses tool calls. It is not a tool registry. Each request contains its
tools as data. Each tool has a name, a description and a JSON Schema for its arguments. The
daemon does these steps:

- It renders the tools into the prompt, in the format that the model was trained on.
- During sampling, it limits the arguments to the schema.
- It parses the output of the model into tool calls.

The daemon never runs a tool, and never makes sure that a tool exists. It keeps no data between
requests. **Your application must run the tools and send the results in a new request.**

- Tool calls are complete and are in the `finish` chunk, with `reason` =
  `FINISH_REASON_TOOL_CALLS`. There are no partial arguments. The markup of the tool calls is
  never in a `text` chunk.
- `finish.tool_calls` is not empty if and only if the reason is `TOOL_CALLS`. If `max_tokens` or
  a cancel stops a reply, the reason is `LENGTH` or `CANCELLED`, and there are no tool calls.
- `arguments_json` is always a complete JSON object. For a tool without parameters, it is `{}`.

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
  `STOP` and the stop string matched after the object. `LENGTH` and `CANCELLED` tell that the
  object is not complete.
- You cannot use a schema and tools in the same request. The daemon rejects the schema `{}`,
  because it puts no limit on the reply. The two errors are `INVALID_ARGUMENT`.
- A schema turn has no reasoning. If the template opens a `<think>` block, the daemon tells the
  template to close it. Thus, the reply is only the JSON object.

## Embeddings

The GGUF of an embedding model declares a pooling type. With an embedding model, the daemon
answers `Embed` and rejects `Generate` and `Chat`. With all other models, the daemon does the
opposite. `GetModelInfo` tells which type of model the daemon serves (`serves_embeddings`) and
the length of the vectors (`n_embd`).

`Embed` returns one vector for each input, in the same sequence as the inputs. Each vector is
L2-normalised. Thus, the dot product of two vectors is their cosine similarity. The daemon embeds
each input separately, in one decode. An input can have a maximum of 512 tokens. A short `--ctx`
or a short training context makes this limit lower. Some models expect an instruction prefix on
queries, for example `Instruct: ...\nQuery:` for Qwen3-Embedding. Put this prefix in the text
that you send.

## Errors

| Status | Meaning |
|---|---|
| `INVALID_ARGUMENT` | The request is not correct, and the application can correct it. Examples: no messages, a prompt that is longer than the context, schema JSON that does not parse, a schema with tools, an embedding input that is too long. |
| `FAILED_PRECONDITION` | The loaded model cannot do this request. Examples: `Chat` without a usable chat template, `Embed` on a generative model, `Generate` or `Chat` on an embedding model. |
| `CANCELLED`, `DEADLINE_EXCEEDED` | The application cancelled the call, or its deadline passed. |
| `INTERNAL` | All other errors, for example a decode error. The message tells what occurred. |

The daemon serves one generation at a time. Thus, concurrent requests wait in a queue. A deadline
includes the time in the queue and the time of the generation.
