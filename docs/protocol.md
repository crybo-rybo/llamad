# Integrating with llamad

An application talks to llamad through one file, `proto/llamad/v1/llamad.proto`. There is no
client library to link: generate stubs for your language with `protoc`, point a gRPC channel at
the daemon's Unix socket, and call the `llamad.v1.Llama` service. The guarantees every response
keeps are written at the top of that file; this page walks through using them.

## Connecting

Copy `llamad.proto` into your project (or pin this repository as a submodule and point `protoc`
at its `proto/` directory) and generate code the way you would for any gRPC service:

```sh
# C++
protoc -I proto --cpp_out=gen --grpc_out=gen \
    --plugin=protoc-gen-grpc="$(command -v grpc_cpp_plugin)" llamad/v1/llamad.proto

# Python
python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. llamad/v1/llamad.proto
```

The gRPC target is `unix:` followed by the socket path. The daemon listens on
`$XDG_RUNTIME_DIR/llamad.sock` unless started with `--socket`, falling back to
`/tmp/llamad-<uid>.sock` where `XDG_RUNTIME_DIR` is unset. The socket is mode 0600, so only the
user that started the daemon can connect, and the channel is plaintext: there is no TLS and no
authentication beyond the socket's permissions.

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

The daemon also serves gRPC server reflection and the standard health service, so a tool such as
[grpcurl](https://github.com/fullstorydev/grpcurl) needs no copy of the proto:

```sh
grpcurl -plaintext unix:///run/user/1000/llamad.sock list
grpcurl -plaintext unix:///run/user/1000/llamad.sock llamad.v1.Llama/GetModelInfo
grpcurl -plaintext -d '{"messages": [{"role": "user", "content": "Name three primes."}]}' \
    unix:///run/user/1000/llamad.sock llamad.v1.Llama/Chat
```

## Streams

`Generate` and `Chat` stream `GenerateChunk`s. Each chunk is one of two things:

- `text`: the next piece of user-visible output. Never empty, never a partial UTF-8 sequence,
  never part of a matched stop string, and never tool-call markup.
- `finish`: how the generation ended (`reason`), its token counts and timings (`stats`), and any
  tool calls. It is always the last chunk of a successful stream, and there is exactly one.

A stream that ends without a `finish` was cancelled or failed; the gRPC status says which. An
error that is found before generation starts (an empty conversation, a bad schema) ends the
stream with that status and no chunks at all.

To cancel, cancel the call the way your gRPC library does it, or set a deadline. The daemon stops
generating at the next token.

`GenerateStats.cached_prompt_tokens` says how many leading prompt tokens the daemon reused from
the previous request's KV cache. Resending a conversation with a new turn on the end typically
reuses everything before that turn.

## Tool calling

The daemon is a formatter and a parser, not a tool registry. Tools are data that travels with
each request: a name, a description and a JSON Schema for the arguments. The daemon renders them
into the prompt the way the model was trained to see them, constrains the arguments to the schema
while sampling, and parses the model's output back into calls. It never runs a tool, never checks
that one exists, and remembers nothing between requests. **Your application owns the
execute-and-resend loop.**

- Calls arrive whole, on `finish`, with `reason` = `FINISH_REASON_TOOL_CALLS`. There are no
  argument deltas, and the markup the model wrote them in never reaches `text`.
- `finish.tool_calls` is non-empty if and only if the reason is `TOOL_CALLS`. A reply cut short by
  `max_tokens` or a cancel reports `LENGTH` or `CANCELLED` and carries no calls.
- `arguments_json` is always a complete JSON object, `{}` for a tool without parameters.

One round looks like this:

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

Send one `tool` message per call, each with the `id` it answers. The result is an opaque string to
the daemon; JSON is a good choice, and so is an `{"error": ...}` object the model can recover from.
Bound the number of rounds yourself: nothing in the daemon stops a model calling tools forever.

## Typed replies

`ChatRequest.response_json_schema` asks for a JSON object instead of prose. The daemon builds a
grammar from the schema, so the sampler can only produce an object of that shape, and puts the
schema into the prompt as a system instruction, so property descriptions reach the model.

- The JSON streams as ordinary `text`. Concatenate the chunks and parse the result.
- It is a complete object when `finish.reason` is `EOG`, or `STOP` with the stop string matched
  after the object. `LENGTH` and `CANCELLED` mean it was cut off.
- A schema cannot be combined with tools, and `{}` is refused because it constrains nothing; both
  are `INVALID_ARGUMENT`.
- Thinking is off on a schema turn: on a template that opens a `<think>` block, the daemon asks
  for it closed, so the reply is the JSON object and nothing else.

## Embeddings

A daemon serving an embedding model, one whose GGUF declares a pooling type, answers `Embed`,
and refuses `Generate` and `Chat`; any other model is the reverse. `GetModelInfo` says which
(`serves_embeddings`) and how long the vectors are (`n_embd`).

`Embed` returns one vector per input, in order, each L2-normalised, so the dot product of two is
their cosine similarity. Each input is embedded on its own, in one decode, and may have at most
512 tokens (fewer with a short `--ctx` or a short training context). An instruction prefix a model
expects on queries, such as Qwen3-Embedding's `Instruct: ...\nQuery:`, is part of the text you
send.

## Errors

| Status | Meaning |
|---|---|
| `INVALID_ARGUMENT` | The request is wrong and the caller can fix it: no messages, a prompt longer than the context, schema JSON that does not parse, a schema with tools, an embedding input that is too long. |
| `FAILED_PRECONDITION` | The loaded model cannot do this: `Chat` without a usable chat template, `Embed` on a generative model, `Generate` or `Chat` on an embedding model. |
| `CANCELLED`, `DEADLINE_EXCEEDED` | The caller cancelled, or its deadline passed. |
| `INTERNAL` | Anything else, such as a failed decode. The message says what. |

The daemon serves one generation at a time, so concurrent requests queue. A deadline covers the
time spent queued as well as the time generating.
