# Performance

`engine_bench` measures prefill and decode speed in its own process, through `Engine::generate`.
This is the same path that all `Generate` and `Chat` requests use: tokenization, the prompt
cache, sampling and the stream of output. Only the socket is not included. `scripts/bench.sh`
runs `engine_bench` from a build. It adds the median of each scenario to
`build-<mode>/bench.tsv`, with the commit that it measured:

```sh
./scripts/bench.sh gpu models/qwen2.5-0.5b-instruct-q4_k_m.gguf
./scripts/bench.sh cpu models/qwen2.5-0.5b-instruct-q4_k_m.gguf --reps 3 --only tg
```

| Scenario | Measures |
|---|---|
| `pp128`, `pp512`, `pp2048` | Prefill: a cold prompt of approximately N tokens, with no generation |
| `tg128-greedy` | Decode: a maximum of 128 tokens at temperature 0, after a short prompt |
| `tg128-sampled` | Decode with the default sampler chain (temperature 0.8, top-k 40, top-p 0.95, min-p 0.05) |
| `tg128-greedy@2048` | Decode after a 2048-token prompt. Attention over the cache costs more here. |
| `tg128-json` | Decode with the grammar that the chat layer makes for a response schema, as for a typed `Chat` reply |

Each scenario runs one time to warm up, and then `--reps` times (the default is 5). The table
shows the median. The TSV also gives the slowest run and the fastest run. The prompt of each run
starts with the number of that run. Thus, the prompt cache holds a maximum of one or two tokens of
the prompt, and the prefill is always cold.

For prefill, tokens per second counts the decoded tokens. For decode, it counts the generated
tokens in the time from the first sample to the last sample. The bench does not send the stream
to a callback. Thus, these numbers do not include the cost of the socket for each chunk.

The numbers change by a few percent between runs of the same binary. They change more on a
machine that is busy or warm. Compare a change with a baseline that you measure in the same
session. Do not compare it with the table below.

## Reference numbers

The machine is an Apple M3 Pro (6 performance cores, 6 efficiency cores, 18 GB) with macOS. The
table shows a Metal build (`gpu`) and a CPU-only build (`cpu`). The values are tokens per second,
as the median of 5 runs (3 runs for `cpu`).

| Build | Model | pp128 | pp512 | pp2048 | tg greedy | tg sampled | tg after 2048 | tg json |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gpu | Qwen2.5 0.5B Instruct Q4_K_M | 4583 | 5509 | 5224 | 204.9 | 204.0 | 196.0 | 201.4 |
| gpu | Qwen3 0.6B Q8_0 | 4046 | 4519 | 3791 | 153.5 | 153.1 | 121.6 | 151.5 |
| cpu | Qwen2.5 0.5B Instruct Q4_K_M | 860 | 1072 | 933 | 216.5 | 216.2 | 159.4 | 210.2 |
| cpu | Qwen3 0.6B Q8_0 | 1158 | 978 | 596 | 158.7 | 158.5 | 71.4 | 149.7 |

On the CPU, the results for short prompts can change by a maximum of 20% between runs. This
occurs because the prompt decode uses the efficiency cores and also the performance cores.

For comparison, use `llama-bench` from llama.cpp at the pinned commit. For the Qwen2.5 model on
the Metal build, it gives 5562 (pp512) and 208.7 (tg128). `llama-bench` does not read the logits
when it decodes. Thus, it can put the work for the next token in the queue before the work for
the last token is complete. `generate` cannot do this, because each decode must have the token
that the sampler selected before it. Also, the tg128 of `llama-bench` is 128 decodes. But
`generate` decodes 127 of the 128 tokens that it samples, because the token that ends a reply
never goes into the cache.
