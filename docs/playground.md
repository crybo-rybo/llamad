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

## Try the tools

Enter these prompts:

```text
What time is it?
Roll two six-sided dice for my goblin's attack.
Flip a coin to decide whether we enter the haunted cave.
Get the time and roll one twenty-sided die.
```

The client supplies three tools with each chat request:

| Tool | Arguments | Result |
|---|---|---|
| `get_time` | None | Local time with a UTC offset, and UTC time |
| `roll_dice` | `count`: 1–10, `sides`: 2–100 | Each random roll and their total |
| `flip_coin` | None | Random `heads` or `tails` |

The daemon constrains tool arguments with JSON schemas. Tool choice depends on the model.
A larger tool-capable model can give better results than the small smoke-test model.

Text appears as the daemon sends it. At the end of each stream, the client shows the finish reason and stats.
Stats include prompt tokens, cached prompt tokens, output tokens, prefill time and decode speed.
Decode speed includes stream time. It is not a benchmark.

For each complete tool call, the client prints the name, arguments and result.
The client executes the tool, then sends its result with the call ID and the full history.
The daemon never executes tools. Tool errors return as JSON so the model can respond to them.
The client permits at most four tool rounds per user prompt.

## Commands

| Command | Action |
|---|---|
| `/help` | Show commands and example prompts |
| `/clear` | Reset the client's chat history |
| `/json PROMPT` | Generate a quest object with `title`, `objective` and `reward` |
| `/quit` or Ctrl+D | Exit |

JSON requests use `response_json_schema` without tools. Each JSON request is separate from the chat history.
For example:

```text
/json Invent a tiny quest about a dragon who lost its glasses.
```

The client uses temperature zero and a limit of 512 output tokens per stream.
Each chat RPC has a two-minute deadline. Ctrl+C exits the client and disconnects its stream.
For a conversation that exceeds the model's context, use `/clear`.
If a request fails, the client preserves the history from before that user prompt.
The client returns a nonzero exit code if an RPC fails or a reply is incomplete in JSON mode.

The playground demonstrates chat, tool calls, structured replies and cache stats.
An embedding model cannot serve chat. See [protocol.md](protocol.md) for the `Embed` RPC and other capabilities.
