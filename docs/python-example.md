# Python example

`examples/python/chat.py` is a small terminal chat client. It shares only `llamad.proto` with
the daemon. There is no client library and there are no generated stubs. When the client starts,
`grpcio-tools` compiles `proto/llamad/v1/llamad.proto` from the checkout. Thus the client always
agrees with the current proto.

## Start the example

1. Build the project, and start the daemon with a chat model:

   ```sh
   ./scripts/build.sh cpu
   ./build-cpu/llamad --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --ctx 4096
   ```

2. In another terminal, install the requirements with pip:

   ```sh
   python3 -m venv .venv && . .venv/bin/activate
   pip install -r examples/python/requirements.txt
   ```

3. Start the client:

   ```sh
   python examples/python/chat.py
   ```

If you use [uv](https://docs.astral.sh/uv/), you can skip step 2. Start the client with
`uv run examples/python/chat.py`. uv reads the dependencies from the top of `chat.py`.

Run the client from a llamad checkout, because it reads the proto from the `proto/` directory.
For a different socket, give the daemon and the client the same `--socket PATH`.

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
