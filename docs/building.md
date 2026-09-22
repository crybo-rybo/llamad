# Building

## Compiler and platforms

llamad is written in C++26 and uses static reflection, so it needs GCC 16 or later. Clang and
Apple Clang do not implement reflection. Linux and macOS on Apple silicon are the supported
platforms; on both, GCC compiles every C++ source, including your own if you link the client.

## Linux

```sh
pacman -S gcc grpc protobuf cmake ninja     # Arch Linux
git clone --recurse-submodules git@github.com:crybo-rybo/llamad.git
cd llamad
./scripts/build.sh cpu                      # builds into build-cpu/
./scripts/test.sh cpu                       # chat template, flags, wire contract, JSON and tool set tests
```

If you already cloned without submodules: `git submodule update --init --recursive`.

## macOS

```sh
brew install gcc cmake ninja openssl@3
git clone --recurse-submodules git@github.com:crybo-rybo/llamad.git
cd llamad
./scripts/build-deps-macos.sh               # once: builds gRPC into build-deps/
./scripts/build.sh cpu
./scripts/test.sh cpu
```

Homebrew's gRPC, Protobuf and Abseil are built against libc++, and `std::string` and its
relatives cross their APIs, so GCC's libstdc++ cannot link against them.
`./scripts/build-deps-macos.sh` builds gRPC from source instead, with the same compilers the
project uses, and installs it under `build-deps/`. It takes about fifteen minutes and a gigabyte.

`./scripts/build.sh` then compiles llama.cpp's C and Objective-C Metal sources with Apple clang,
which GCC cannot parse, and every C++ source with `g++-16`. Accelerate and BLAS are off, because
their headers do not compile with GCC; that costs only prompt-processing speed on the CPU path.

## What the build contains

The build includes llama.cpp's `common` library, which the chat layer needs for Jinja
templates and tool-call parsing; it is the bulk of a first build.

The tests need no model file and no daemon: the chat-template ones render and parse against a
template checked into the submodule, the rest need nothing but the build.
`./scripts/build-test.sh [cpu|gpu]` builds and then runs them in one step. Extra arguments to
`build.sh` are passed to CMake configure, and `CMAKE_BUILD_PARALLEL_LEVEL` sets the job count
(default 4).

## GPU

llama.cpp's GPU backends are enabled with their usual CMake flags, and `build.sh gpu` passes the
one for the platform: Vulkan on Linux, Metal on macOS. `build.sh cpu` turns that backend off.
Vulkan is the Linux backend tested here: it runs on Pascal cards (GTX 10xx), which CUDA 13 no
longer targets.

```sh
pacman -S vulkan-headers spirv-headers vulkan-icd-loader shaderc   # Linux; Metal needs nothing
./scripts/build.sh gpu                  # -DGGML_VULKAN=ON or -DGGML_METAL=ON, in build-gpu/
./scripts/test.sh gpu
./scripts/smoke-test.sh gpu /absolute/path/to/model.gguf
```

The project does not include a model. `test.sh` needs none; `smoke-test.sh` requires
an explicit model path, absolute or relative to your working directory. It runs one chat
turn through `engine_smoke` at temperature 0.

On Linux a working Vulkan driver for the GPU is also required. The smoke test fails if
inference falls back to the CPU; listing devices alone does not test model loading
or token generation.

### Device selection

Under Vulkan every discrete GPU is used by default: llama.cpp splits the model's
layers across them in proportion to each card's free memory, and ignores an
integrated GPU whenever a discrete one exists. Two 8 GB cards therefore hold a
model that fits on neither alone. Apple silicon offers one Metal device, `MTL0`,
over unified memory. The daemon prints one line per offload device at startup.

```sh
./build-gpu/llamad --list-devices                    # names, types, free/total memory
./build-gpu/llamad --model M --devices Vulkan0       # this card only; MTL0 on macOS
./build-gpu/llamad --model M --tensor-split 3,1      # 3:1 share, in device order
```

## API reference

Only CMake and Doxygen (1.17 or later) are needed to generate HTML; no compiler, submodule
checkout, Graphviz, model or gRPC installation is required:

```sh
pacman -S cmake doxygen
./scripts/docs.sh                       # open build-docs/html/index.html
```
