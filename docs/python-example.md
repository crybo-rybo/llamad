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

## Use the client

Each reply ends with a stats line. The client sends the full history with each message, because
the daemon keeps no conversation. The `cached` count is the part of that prompt that the daemon
reuses from its KV cache ([design.md](design.md)).

Use `/clear` to reset the history, `/help` for the commands, and `/quit` or Ctrl+D to exit.
Ctrl+C cancels a reply.

The client offers one tool, `get_current_time`. When the model calls it, the client runs it and
sends the result back. A dim line shows each call and its result. A turn can have at most four
replies. Small models, such as the 0.5B model, often do not call the tool. For typed replies,
embeddings and tool details, refer to [protocol.md](protocol.md).
