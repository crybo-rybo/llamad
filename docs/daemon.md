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
| `--threads N` | 0 | inference threads; 0 = auto: half the hardware threads to generate, all of them for prompts |
| `--devices NAMES` | every discrete GPU | comma-separated offload devices, e.g. `Vulkan0` or `MTL0` |
| `--tensor-split S` | by free memory | comma-separated share per device, e.g. `3,1` |
| `--list-devices` | | print the devices this build can offload to, and exit |

The socket is created mode 0600, so only your user can talk to it. The daemon logs one
`[llamad] ...` line per request to stderr and writes nothing to stdout.

SIGINT/SIGTERM shut the daemon down and remove the socket. A daemon put in the background by a
script inherits SIGINT ignored, and macOS discards a signal that is both ignored and blocked, so
stop one started that way with SIGTERM.

The engine serves one generation at a time: concurrent clients queue rather than sharing the
context. The KV cache keeps what the previous request decoded, and a request decodes only the
part of its prompt after the prefix the two share, so a chat turn or a tool round pays for its
new messages rather than the whole history. `cached_prompt_tokens`, in the stats and in the
daemon's log line, is the number of prompt tokens reused. A reused prefix was decoded in a
different batch split than a cold run would use, and the logits are not bit-for-bit identical
across batch splits, so a `--temp 0` reply from a warm cache can differ from a freshly started
daemon's.

## Embedding models

What the daemon serves follows from the model. A GGUF that declares a pooling type (mean, CLS or
last token), as embedding models such as bge-small-en-v1.5 and Qwen3-Embedding do, makes it an
embedding daemon: it answers `Embed`, and `Generate` and `Chat` fail with `FAILED_PRECONDITION`.
Any other model is the reverse. `GetModelInfo` reports which, as `serves_embeddings`, and the
vector length as `n_embd`. Reranking models, which pool into scores rather than vectors, are
refused at startup.

`Embed` takes a batch of inputs and returns one L2-normalised vector per input, so the dot product
of two vectors is their cosine similarity. Each input is embedded on its own, in a single decode,
so an input may have at most 512 tokens (llama.cpp's physical batch), fewer if `--ctx` or the
model's training context is shorter; a longer one fails the request with `INVALID_ARGUMENT`.
Inputs are tokenized as `Generate` prompts are, with the model's special tokens added, and any
instruction prefix a model expects (Qwen3-Embedding's `Instruct: ...\nQuery:` for queries, say)
is the caller's to write. A client that cancels, or whose deadline passes, stops the batch before
its next input.

## Chat from the terminal

```sh
./build-cpu/client/llamad-chat                       # interactive REPL
./build-cpu/client/llamad-chat --once "Hello" --temp 0
./build-cpu/client/llamad-chat --demo-tools          # with one built-in tool
./build-cpu/client/llamad-chat --demo-json           # one reply parsed into a struct
```

Also accepts `--socket`, `--system TEXT`, `--seed N`, `--max-tokens N`. Ctrl-C
cancels the reply in progress; Ctrl-C or Ctrl-D at the prompt quits.

`--demo-tools` offers a `get_current_time` tool and runs the execute-and-resend loop for any
call the model makes; [client.md](client.md) explains what that involves.

```sh
./build-cpu/client/llamad-chat --demo-tools --once "What time is it in Tokyo right now?" --temp 0
# [tool] get_current_time(Asia/Tokyo) -> 2026-09-21 08:44:44 JST
# The current time in Tokyo is 2026-09-21 08:44:44 JST.
# [stats] finish=eog prompt_tokens=465 cached_prompt_tokens=219 completion_tokens=52 ...
```

`--demo-json` runs one turn whose reply is constrained to a struct's JSON Schema, streams the
JSON and prints the parsed fields. It cannot be combined with `--once` or `--demo-tools`.

`--embed TEXT`, repeatable, asks a daemon serving an embedding model for the vectors of every
TEXT in one request, prints the first few values of each and its cosine with the first, and exits.

```sh
./build-cpu/client/llamad-chat --embed "A cat sits on the mat." --embed "A kitten is resting on a rug." \
    --embed "The central bank raised interest rates."
# embedding 0: cosine with 0: 1.0000  values: 0.035994 -0.024416 0.018401 0.058833 ...
# embedding 1: cosine with 0: 0.7900  values: -0.010850 0.003091 0.075260 0.107859 ...
# embedding 2: cosine with 0: 0.3779  values: -0.031368 0.013564 -0.028393 0.028225 ...
# [stats] inputs=3 n_embd=384 input_tokens=28
```

## The engine without the daemon

`./build-cpu/tests/engine_smoke` drives the engine in-process, with no daemon and no gRPC.
It takes the same context and offload flags as the daemon, plus `--chat` to render the prompt
through the model's chat template, `--demo-tool` to add the same `get_current_time` tool to
that rendering, `--grammar-file PATH` to constrain generation with a GBNF file of your own,
`--stop` and `--cancel-after`. With an embedding model, `--embed TEXT` (repeatable, in place of
the prompt) embeds instead and prints what `llamad-chat --embed` does. `--help` lists them all.
