# Running the daemon

```sh
./build-cpu/llamad --model /absolute/path/to/model.gguf
# [llamad] listening on unix:/run/user/1000/llamad.sock
```

| Flag | Default | Meaning |
|---|---|---|
| `--model PATH` | required | GGUF model to load |
| `--socket PATH` | `$XDG_RUNTIME_DIR/llamad.sock`, else `/tmp/llamad-<uid>.sock` | Unix socket to listen on |
| `--ctx N` | 4096 | context size in tokens |
| `--ngl N` | 99 | layers to offload to the GPU; 0 disables offload |
| `--threads N` | 0 | inference threads; 0 = auto, half the hardware threads |
| `--devices NAMES` | every discrete GPU | comma-separated offload devices, e.g. `Vulkan0` or `MTL0` |
| `--tensor-split S` | by free memory | comma-separated share per device, e.g. `3,1` |
| `--list-devices` | | print the devices this build can offload to, and exit |

The socket is created mode 0600, so only your user can talk to it. The daemon logs one
`[llamad] ...` line per request to stderr and writes nothing to stdout.

SIGINT/SIGTERM shut the daemon down and remove the socket. A daemon put in the background by a
script inherits SIGINT ignored, and macOS discards a signal that is both ignored and blocked, so
stop one started that way with SIGTERM.

The engine serves one generation at a time: concurrent clients queue rather than sharing the
context, and each request starts from an empty KV cache.

## Chat from the terminal

```sh
./build-cpu/client/llamad-chat                       # interactive REPL
./build-cpu/client/llamad-chat --once "Hello" --temp 0
./build-cpu/client/llamad-chat --demo-tools          # with one built-in tool
```

Also accepts `--socket`, `--system TEXT`, `--seed N`, `--max-tokens N`. Ctrl-C
cancels the reply in progress; Ctrl-C or Ctrl-D at the prompt quits.

`--demo-tools` offers a `get_current_time` tool and runs the execute-and-resend loop for any
call the model makes; [client.md](client.md) explains what that involves.

```sh
./build-cpu/client/llamad-chat --demo-tools --once "What time is it in Tokyo right now?" --temp 0
# [tool] get_current_time(Asia/Tokyo) -> 2026-09-21 08:44:44 JST
# The current time in Tokyo is 2026-09-21 08:44:44 JST.
# [stats] finish=eog prompt_tokens=461 completion_tokens=52 ...
```

## The engine without the daemon

`./build-cpu/tests/engine_smoke` drives the engine in-process, with no daemon and no gRPC.
It takes the same context and offload flags as the daemon, plus `--chat` to render the prompt
through the model's chat template, `--demo-tool` to add the same `get_current_time` tool to
that rendering, `--grammar-file PATH` to constrain generation with a GBNF file of your own,
`--stop` and `--cancel-after`. `--help` lists them all.
