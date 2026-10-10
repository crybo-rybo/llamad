# Python example

`examples/python/chat.py` is a small terminal chat client. It shares only `llamad.proto` with
the daemon. It uses stubs that `grpc_tools.protoc` generates from the proto, and no client library.
The script shows how an application uses the wire contract:

- The client keeps the chat history and sends all of it with each message.
- Replies stream as text chunks. The `finish` chunk gives the finish reason and the stats.
- The client loads no model and does not use the libraries of the project.

## Start the example

1. Build the project, and start the daemon with a chat model:

   ```sh
   ./scripts/build.sh cpu
   ./build-cpu/llamad --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --ctx 4096
   ```

2. In another terminal, create a Python virtual environment and install the requirements:

   ```sh
   python3 -m venv .venv
   . .venv/bin/activate
   pip install -r examples/python/requirements.txt
   ```

3. Generate the stubs from the proto:

   ```sh
   python -m grpc_tools.protoc -I proto --python_out=examples/python \
       --grpc_python_out=examples/python llamad/v1/llamad.proto
   ```

4. Start the client:

   ```sh
   python examples/python/chat.py
   ```

The stubs go into `examples/python/llamad/`, and git ignores them. Thus, the repository has no
generated code that can differ from the proto. If the proto changes, do step 3 again.

For a different socket, give the daemon and the client the same `--socket PATH`. The default
socket of the client is the same as the default socket of the daemon. Use `--help` for details.

## Sample session

```text
llamad | qwen2 1B Q4_K - Medium | context 4096 tokens

you> Hi, my name is Sam. I like sailing.
assistant> Hello Sam! I'm glad to meet you. It's nice to meet someone like you. What kind of sailing do you like to do?
[finish EOG | prompt 32 (cached 0) | output 29 | prefill 53.6 ms | decode 123.5 tok/s]
you> What is my name and what do I like?
assistant> Your name is Sam, and you like sailing.
[finish EOG | prompt 81 (cached 61) | output 10 | prefill 54.5 ms | decode 127.8 tok/s]
```

## Read a reply

Text appears as the daemon sends it. At the end of each stream, the client shows the finish
reason and these stats:

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
Each chat RPC has a two-minute deadline. Ctrl+C during a reply cancels the RPC and keeps the
history from before that message. Ctrl+C at the prompt exits the client.
For a conversation that exceeds the model's context, use `/clear`.
If a request fails, the client keeps the history from before that message.
The client returns a nonzero exit code if an RPC fails.

The example shows only chat. For tool calls, typed replies (`response_json_schema`) and
embeddings, refer to [protocol.md](protocol.md).
