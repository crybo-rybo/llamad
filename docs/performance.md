# Performance

Prefill and decode speed are measured in-process by `engine_bench`, through
`Engine::generate`: the same tokenizing, prompt-cache, sampling and streaming path every
`Generate` and `Chat` request takes, without the socket. `scripts/bench.sh` runs it in an
existing build and appends each scenario's median to `build-<mode>/bench.tsv`, tagged with the
commit it measured:

```sh
./scripts/bench.sh gpu models/qwen2.5-0.5b-instruct-q4_k_m.gguf
./scripts/bench.sh cpu models/qwen2.5-0.5b-instruct-q4_k_m.gguf --reps 3 --only tg
```

| Scenario | Measures |
|---|---|
| `pp128`, `pp512`, `pp2048` | Prefill: a cold prompt of about N tokens, nothing generated |
| `tg128-greedy` | Decode: up to 128 tokens at temperature 0 after a short prompt |
| `tg128-sampled` | Decode with the default sampler chain (temperature 0.8, top-k 40, top-p 0.95, min-p 0.05) |
| `tg128-greedy@2048` | Decode after a 2048-token prompt, where attention over the cache costs more |
| `tg128-json` | Decode under the chat layer's grammar for a response schema, as a typed `Chat` reply gets |

Each scenario runs once to warm up and then `--reps` times (default 5); the table shows the
median, and the TSV adds the slowest and fastest run. Every run's prompt opens with its own run
number, so the prompt cache reuses at most a token or two and prefill is always cold. Prefill
tokens per second counts the tokens decoded; decode counts the tokens generated over the time
from the first sample to the last. The bench streams to no callback, so the socket's per-chunk
cost is not in these numbers.

Numbers move a few percent between runs of the same binary, more on a machine that is busy or
warm. Compare a change against a baseline measured in the same sitting, not against the table
below.

## Reference numbers

Apple M3 Pro (6 performance and 6 efficiency cores, 18 GB), macOS, Metal build (`gpu`) and
CPU-only build (`cpu`). Tokens per second, median of 5 runs (3 for `cpu`).

| Build | Model | pp128 | pp512 | pp2048 | tg greedy | tg sampled | tg after 2048 | tg json |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gpu | Qwen2.5 0.5B Instruct Q4_K_M | 4583 | 5509 | 5224 | 204.9 | 204.0 | 196.0 | 201.4 |
| gpu | Qwen3 0.6B Q8_0 | 4046 | 4519 | 3791 | 153.5 | 153.1 | 121.6 | 151.5 |
| cpu | Qwen2.5 0.5B Instruct Q4_K_M | 860 | 1072 | 933 | 216.5 | 216.2 | 159.4 | 210.2 |
| cpu | Qwen3 0.6B Q8_0 | 1158 | 978 | 596 | 158.7 | 158.5 | 71.4 | 149.7 |

Short prompts on the CPU vary by up to 20% between runs, because prompt decoding uses the
efficiency cores as well as the performance cores.

For scale, llama.cpp's own `llama-bench` at the pinned commit measures the Qwen2.5 model on the
Metal build at 5562 (pp512) and 208.7 (tg128). It decodes without reading logits, so it can
queue the next token's work before the last one finishes; `generate` cannot, because each token
depends on the one sampled before it.
