# Run the daemon

```sh
./build-cpu/llamad --model /absolute/path/to/model.gguf
# [llamad] listening on unix:/run/user/1000/llamad.sock
```

| Flag | Default | Meaning |
|---|---|---|
| `--model PATH` | necessary | The GGUF model to load |
| `--socket PATH` | `$XDG_RUNTIME_DIR/llamad.sock`, else `/tmp/llamad-<uid>.sock` | The Unix socket to listen on |
| `--ctx N` | 4096 | The context size in tokens |
| `--ngl N` | 99 | The number of layers to offload to the GPU. 0 disables offload. |
| `--threads N` | 0 | The number of inference threads. 0 is automatic: half of the hardware threads for generation, and all of them for prompts. |
| `--devices NAMES` | all discrete GPUs | A comma-separated list of offload devices, for example `Vulkan0` or `MTL0` |
| `--tensor-split S` | by free memory | A comma-separated list with the share of each device, for example `3,1` |
| `--list-devices` | | Show the devices that this build can offload to, then exit |

The daemon creates the socket with mode 0600. Thus, only your user can connect to it. The daemon
writes one `[llamad] ...` log line for each request to stderr. It writes nothing to stdout.

SIGINT or SIGTERM stops the daemon and removes the socket. If a script starts the daemon in the
background, the daemon inherits SIGINT as ignored. macOS discards a signal that is both ignored
and blocked. Thus, use SIGTERM to stop a daemon that a script started in the background.

The engine serves one generation at a time. Concurrent requests wait in a queue and do not share
the context.

The KV cache keeps the tokens that the previous request decoded. A request decodes only the part
of its prompt after the prefix that it shares with these tokens. Thus, a chat turn or a tool
round uses time only for its new messages, and not for the full history. `cached_prompt_tokens`
is the number of prompt tokens that came from the cache. It is in the stats and in the log line
of the daemon.

The engine decoded a cached prefix in a batch split that is different from the split of a cold
run. The logits are not identical bit for bit across batch splits. Thus, a reply at temperature 0
from a warm cache can be different from the reply of a daemon that just started.

## Embedding models

The model sets what the daemon serves. Embedding models (for example bge-small-en-v1.5 and
Qwen3-Embedding) have a GGUF that declares a pooling type: mean, CLS or last token. With such a
model, the daemon is an embedding daemon. It answers `Embed`, and it returns
`FAILED_PRECONDITION` for `Generate` and `Chat`. With all other models, the daemon does the
opposite. `GetModelInfo` tells which type of model the daemon serves, in `serves_embeddings`, and
the vector length, in `n_embd`.

Reranking models pool into scores, and not into vectors. If you give the daemon a reranking model,
the daemon does not start.

`Embed` accepts a batch of inputs. It returns one L2-normalised vector for each input. Thus, the
dot product of two vectors is their cosine similarity. The daemon embeds each input separately,
in one decode. Thus, an input can have a maximum of 512 tokens, which is the physical batch of
llama.cpp. The limit is lower if `--ctx` or the training context of the model is shorter. If an
input is longer, the daemon returns `INVALID_ARGUMENT` for the request.

The daemon tokenizes inputs the same way as `Generate` prompts, and adds the special tokens of the
model. Some models expect an instruction prefix, for example `Instruct: ...\nQuery:` for
Qwen3-Embedding queries. The application must write this prefix. If the application cancels the
call, or if its deadline passes, the daemon stops the batch before the next input.

## Send requests from a terminal

The daemon serves gRPC server reflection. Thus,
[grpcurl](https://github.com/fullstorydev/grpcurl) can call the daemon without a copy of the
proto. grpcurl shows each chunk of a stream as one JSON object.

```sh
sock=unix:///run/user/1000/llamad.sock
grpcurl -plaintext $sock llamad.v1.Llama/GetModelInfo
grpcurl -plaintext -d '{"prompt": "The capital of France is", "sampling": {"temperature": 0, "max_tokens": 16}}' \
    $sock llamad.v1.Llama/Generate
grpcurl -plaintext -d '{"inputs": ["a cat", "a kitten"]}' unix:///tmp/embed.sock llamad.v1.Llama/Embed
```

For information about how an application uses these calls, refer to [protocol.md](protocol.md).
It tells about streams, the tool-call loop, typed replies and embeddings.

## The engine without the daemon

`./build-cpu/tests/engine_smoke` runs the engine in its own process, without the daemon and
without gRPC. It accepts the same context and offload flags as the daemon. It also accepts these
flags:

- `--chat`: Render the prompt through the chat template of the model.
- `--demo-tool`: Add a `get_current_time` tool to the rendered chat prompt.
- `--grammar-file PATH`: Limit the generation with a GBNF file that you supply.
- `--stop TEXT`: Stop the generation at this stop string.
- `--cancel-after N`: Cancel the generation after N chunks.
- `--embed TEXT`: With an embedding model, embed this text and not a prompt. You can use this
  flag more than one time. The output shows the first values of each vector, and the cosine of
  each vector with the first vector.

`--help` shows all the flags.
