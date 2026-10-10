# Python example

`examples/python/chat.py` is a small terminal chat client. It shares only `llamad.proto` with
the daemon, and it uses stubs that `grpc_tools.protoc` generates. There is no client library.

## Start the example

1. Build the project, and start the daemon with a chat model:

   ```sh
   ./scripts/build.sh cpu
   ./build-cpu/llamad --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --ctx 4096
   ```

2. In another terminal, install the requirements and generate the stubs:

   ```sh
   python3 -m venv .venv && . .venv/bin/activate
   pip install -r examples/python/requirements.txt
   python -m grpc_tools.protoc -I proto --python_out=examples/python \
       --grpc_python_out=examples/python llamad/v1/llamad.proto
   ```

3. Start the client:

   ```sh
   python examples/python/chat.py
   ```

Git ignores the generated stubs in `examples/python/llamad/`. If the proto changes, generate them
again. For a different socket, give the daemon and the client the same `--socket PATH`.

## Sample session

```text
you> Hi, my name is Sam. I like sailing.
assistant> Hello Sam! I'm glad to meet you. What kind of sailing do you like to do?
[finish EOG | prompt 32 (cached 0) | output 29 | prefill 53.6 ms | decode 123.5 tok/s]
you> What is my name and what do I like?
assistant> Your name is Sam, and you like sailing.
[finish EOG | prompt 81 (cached 61) | output 10 | prefill 54.5 ms | decode 127.8 tok/s]
```

The daemon keeps no conversation, so the client sends the full history with each message.
The `cached` count shows the part of that prompt that the daemon reuses from its KV cache
([design.md](design.md)).

Use `/clear` to reset the history, `/help` for the commands, and `/quit` or Ctrl+D to exit.
Ctrl+C cancels a reply.

The example shows only chat. For tool calls, typed replies and embeddings, refer to
[protocol.md](protocol.md).
