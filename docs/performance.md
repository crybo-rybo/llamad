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
depends on the one sampled before it. Its tg128 is also 128 decodes, where `generate` decodes 127
of the 128 tokens it samples: the one that ends a reply never enters the cache.

## Compiler comparison on Linux x86_64

The September 29, 2026 CPU compiler comparison uses the pinned llama.cpp commit `a894dae939d426954ce54bb604824f1ae918a0c5`
on an Intel Core Ultra 5 250K Plus, with GCC 16.2.1 and Clang 22.1.8. The machine reports
18 physical cores and 18 logical threads, so the physical/logical thread-count cases coincide.
Vulkan offload is available locally on a GTX 1080 and GTX 1070 Ti, but is unused here.
Both builds are Release, use native architecture settings, and disable Vulkan, CUDA, BLAS and
OpenMP. OpenMP is disabled in both because this host has no Clang OpenMP runtime. No source in
llama.cpp is modified.

The models are the local Qwen2.5 0.5B Instruct Q4_K_M file (its GGUF metadata reports 630M
parameters) and Qwen2.5 7B Instruct Q4_K_M. Qwen3 0.6B Q8_0 is unavailable. These results do
not establish Q8_0, Linux aarch64, macOS or GPU performance.

Each cell has six measured repetitions: two runs with three repetitions each, alternating
GCC/Clang and Clang/GCC order. Both binaries use the same 18 threads, 2048-token batch and
micro-batch, F16 KV cache and CPU-only execution. No builds or other model runs compete with
these measurements. The table reports median tokens per second and the full sample range.

The configuration and command matrix are:

```sh
cmake -S third_party/llama.cpp -B build-bench-gcc -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DGGML_NATIVE=ON -DGGML_OPENMP=OFF -DGGML_BLAS=OFF -DGGML_VULKAN=OFF -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF
cmake -S third_party/llama.cpp -B build-bench-clang -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DGGML_NATIVE=ON -DGGML_OPENMP=OFF -DGGML_BLAS=OFF -DGGML_VULKAN=OFF -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF
cmake --build build-bench-gcc --target llama-bench --parallel 3
cmake --build build-bench-clang --target llama-bench --parallel 3

# Run each binary/model with these options, then repeat in the opposite compiler order.
build-bench-gcc/bin/llama-bench -m MODEL -ngl 0 -dev none -t 18 -ub 2048 \
  -p 128,512,2048 -n 0 -r 3 -o json
build-bench-gcc/bin/llama-bench -m MODEL -ngl 0 -dev none -t 18 -ub 2048 \
  -p 0 -n 64 -d 0,2048 -r 3 -o json
```

| Model | Scenario | GCC median (range) | Clang median (range) | Clang change |
| --- | --- | ---: | ---: | ---: |
| 0.5B | pp128 | 997.8 (866.8–1090.0) | 826.2 (544.0–1035.7) | -17.2% |
| 0.5B | pp512 | 1041.6 (896.7–1074.3) | 953.5 (361.4–1062.9) | -8.5% |
| 0.5B | pp2048 | 946.8 (907.4–985.8) | 965.1 (841.5–996.4) | +1.9% |
| 7B | pp128 | 117.8 (93.2–120.5) | 149.4 (143.9–155.7) | +26.9% |
| 7B | pp512 | 95.5 (67.8–119.5) | 137.9 (106.4–159.3) | +44.4% |
| 7B | pp2048 | 97.9 (76.9–113.0) | 123.1 (93.6–149.6) | +25.7% |
| 0.5B | tg64@0 | 158.4 (141.7–187.2) | 136.4 (1.8–192.6) | -13.9% |
| 0.5B | tg64@2048 | 146.3 (134.7–152.1) | 62.4 (3.7–158.6) | -57.3% |
| 7B | tg64@0 | 17.8 (14.6–18.4) | 16.7 (0.9–18.5) | -6.5% |
| 7B | tg64@2048 | 15.7 (7.9–16.7) | 16.7 (4.9–17.7) | +6.2% |

The larger model's prefill advantage appears in both compiler orders, but its magnitude varies
between runs. Several decode samples have long stalls, especially in the first Clang run;
the full ranges retain those samples. These observations do not identify the cause of the
stalls or establish a general Clang advantage. The tiny model's long prefill gap is small,
while its short prompts and decode do not support replacing GCC based on this matrix alone.

Raw per-repetition JSON, stderr, exact commands and machine information are saved under the
ignored `build-issue-records/` directory. The comparison uses a live Linux desktop with its
ordinary scheduler; unrelated desktop activity and hybrid-core scheduling remain possible
sources of variation.

### Engine comparison

The engine comparison holds `engine_bench`, libllama and llama-common at GCC and changes only
`libggml`, `libggml-base` and `libggml-cpu` to the Clang build. The shared libraries use the same
pinned source and configuration; loader inspection confirms that libllama/common remain in
the GCC build directory and all three ggml libraries switch together. Both compilers use
libstdc++ on this Linux host.

`scripts/bench.sh` runs all seven engine scenarios, once to warm up and then three measured
repetitions per scenario. The default engine configuration uses a 4096-token context, 18
prefill threads, 9 decode threads, and 2048-token batches and micro-batches. GCC runs first for
the small model and Clang first for the larger model. No builds or other model checks run
during the comparison. The runs exercise engine tokenization, cache handling, sampling and
grammar. Throughput uses the engine's prompt/completion timers: tokenization and prompt
construction precede those timers, and RPC and socket overhead is absent.

The following commands run the isolated library substitution experiment:

```sh
CMAKE_BUILD_PARALLEL_LEVEL=3 ./scripts/build-test.sh cpu \
  -DGGML_OPENMP=OFF -DGGML_NATIVE=ON -DGGML_BLAS=OFF
mkdir -p build-issue-records/clang-ggml
cp -P build-bench-clang/bin/libggml*.so* build-issue-records/clang-ggml/

# Run Clang first and GCC second for the 7B model.
ldd build-cpu/tests/engine_bench
./scripts/bench.sh cpu MODEL --reps 3
LD_LIBRARY_PATH="$PWD/build-issue-records/clang-ggml" ldd build-cpu/tests/engine_bench
LD_LIBRARY_PATH="$PWD/build-issue-records/clang-ggml" ./scripts/bench.sh cpu MODEL --reps 3
```

Tokens per second, median and full range of three measured runs:

| Model | Scenario | GCC ggml median (range) | Clang ggml median (range) | Clang change |
| --- | --- | ---: | ---: | ---: |
| 0.5B | pp128 | 949.5 (881.3–991.1) | 1040.3 (1036.9–1075.8) | +9.6% |
| 0.5B | pp512 | 1050.3 (1047.5–1065.7) | 1075.0 (1067.0–1082.8) | +2.4% |
| 0.5B | pp2048 | 945.0 (909.8–954.7) | 968.8 (949.6–993.9) | +2.5% |
| 0.5B | tg128-greedy | 162.8 (162.6–163.3) | 163.8 (162.7–164.0) | +0.6% |
| 0.5B | tg128-sampled | 165.4 (158.8–166.7) | 165.1 (165.1–165.5) | -0.2% |
| 0.5B | tg128-greedy@2048 | 126.9 (126.7–127.0) | 127.2 (126.9–127.8) | +0.2% |
| 0.5B | tg128-json | 159.2 (159.0–159.5) | 158.9 (158.2–159.0) | -0.2% |
| 7B | pp128 | 110.7 (107.4–114.2) | 149.1 (148.3–152.7) | +34.7% |
| 7B | pp512 | 117.6 (117.2–117.9) | 154.0 (153.3–154.2) | +31.0% |
| 7B | pp2048 | 111.1 (109.8–112.0) | 142.0 (138.3–142.4) | +27.8% |
| 7B | tg128-greedy | 16.7 (16.6–16.8) | 16.4 (16.3–16.5) | -1.8% |
| 7B | tg128-sampled | 15.6 (15.4–16.0) | 16.5 (16.5–16.5) | +5.8% |
| 7B | tg128-greedy@2048 | 14.8 (14.2–14.9) | 14.8 (14.7–14.8) | +0.0% |
| 7B | tg128-json | 16.6 (15.8–16.7) | 16.3 (16.1–16.4) | -1.8% |

The larger model's prefill gains persist through the actual engine, at 28–35% in this pair of
runs. Small-model prefill ranges from +2% to +10%, while decode changes stay within about 6%.
The engine runs have tighter ranges than the earlier kernel matrix, but one compiler pair per
model does not establish results for other models, machines or desktop loads. Each engine
scenario's median/min/max, commands and loader output are saved in
`build-issue-records/engine-compiler-results/` and `build-cpu/bench.tsv`.

### Build decision

The default C++ build uses GCC. The 7B prefill benefit supports a focused follow-up
experiment with ggml compiled by Clang; the small-model and decode results do not establish a
broad compiler advantage. Q8_0, Linux aarch64 and macOS remain unmeasured here. The product
build retains its existing compiler requirements and single configuration step.

The candidate build has two stages: build and install only the pinned `third_party/llama.cpp/ggml`
with Clang into a private prefix, then configure llama.cpp inside the GCC llamad build with
`LLAMA_USE_SYSTEM_GGML=ON` and an explicit `ggml_DIR` pointing at that prefix. libllama,
llama-common and llamad stay on GCC. The existing ggml C API supplies the boundary; llama.cpp
needs no patch.

A usable implementation must reject stale installs by checking the pinned commit, compiler
version, architecture flags and backend/OpenMP/BLAS settings. CPU and GPU builds need distinct
prefixes, and discovery and runtime library paths must select that same private prefix. CI needs Clang plus a separate build/test of this configuration;
the ordinary GCC and consumer builds remain independent.

The Linux library experiment establishes CPU model loading, generation and shutdown with one
libstdc++ runtime. It does not validate the installed CMake package or dynamic GPU backend
loading. Before adoption, repeat representative workloads and test Vulkan/Metal loading,
cancellation and shutdown. macOS additionally needs its libc++/libstdc++ boundary verified.
The follow-up implementation starts with these focused checks.
