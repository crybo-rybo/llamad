# Terminal playground

`llamad_playground` is a small C++ example client. It connects to the daemon through generated gRPC stubs.
It uses the project's gRPC and Protobuf dependencies. It does not load a model or use the engine library.
The source is in `examples/playground*.{h,cpp}`.

## Start the playground

Build the project:

```sh
./scripts/build.sh cpu
```

In one terminal, start the daemon with a chat model:

```sh
./build-cpu/llamad --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --ctx 4096
```

In another terminal, start the playground:

```sh
./build-cpu/llamad_playground
```

For a different socket, give both programs the same `--socket PATH`.
The default socket matches the daemon. Use `--help` for details.
For a GPU build, use `gpu` and `build-gpu` instead of `cpu` and `build-cpu`.

## Read a reply

Type a message to chat with the model. Text appears as the daemon sends it.
At the end of each stream, the client shows the finish reason and these stats:

- Prompt tokens.
- Cached prompt tokens. The client sends the full history again with each message. Thus, from
  the second message, this count shows how many prompt tokens the daemon reuses from its KV
  cache. Refer to [design.md](design.md) for the stateless design and the cache.
- Output tokens.
- Prefill time.
- Decode speed. This value includes stream time. It is not a benchmark.

## Commands

| Command | Action |
|---|---|
| `/help` | Show the commands |
| `/clear` | Reset the client's chat history |
| `/quit` or Ctrl+D | Exit |

The client uses temperature zero and a limit of 512 output tokens for each reply.
Each chat RPC has a two-minute deadline. Ctrl+C exits the client and disconnects its stream.
For a conversation that exceeds the model's context, use `/clear`.
If a request fails, the client keeps the history from before that message.
The client returns a nonzero exit code if an RPC fails.

The playground shows only chat. For tool calls, typed replies (`response_json_schema`) and embeddings, refer to [protocol.md](protocol.md).
