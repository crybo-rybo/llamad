# Build llamad

## Compiler and platforms

llamad is C++23. Use the compiler of your platform: GCC 13 or Clang 18 on Linux, or Apple Clang
16 (Xcode 16) on macOS. Later versions are also satisfactory. The supported platforms are Linux
and macOS on Apple silicon. Install gRPC and Protobuf with the package manager of the system. If
the Protobuf package includes a CMake config, CMake uses it to find Protobuf. If not (Debian and
Ubuntu), CMake uses its own module.

## Linux

```sh
pacman -S gcc grpc protobuf cmake ninja     # Arch Linux
apt install g++ cmake ninja-build libgrpc++-dev libprotobuf-dev \
    protobuf-compiler protobuf-compiler-grpc  # Debian and Ubuntu
git clone --recurse-submodules https://github.com/crybo-rybo/llamad.git
cd llamad
./scripts/build.sh cpu                      # builds into build-cpu/
./scripts/test.sh cpu                       # tests without a model
```

If you cloned the repository without submodules, use this command:
`git submodule update --init --recursive`.

## macOS

```sh
xcode-select --install                      # Apple Clang, if Xcode is not installed
brew install grpc protobuf cmake ninja
git clone --recurse-submodules https://github.com/crybo-rybo/llamad.git
cd llamad
./scripts/build.sh cpu
./scripts/test.sh cpu
```

The llama.cpp build uses its usual Apple settings: Accelerate for BLAS, the native features of
the CPU, and Metal in a `gpu` build.

## Release archive

Each GitHub release attaches `llamad-<version>.tar.gz`. This archive includes the source of the
pinned llama.cpp commit. Thus, it needs no submodule step. Install the dependencies for your
platform, as shown above. Then do these steps:

```sh
tar -xzf llamad-<version>.tar.gz
cd llamad-<version>
./scripts/build.sh cpu
./scripts/test.sh cpu
```

The archive has no Git repository. Thus, the llama.cpp configure step shows a warning about its
build info. You can ignore this warning. The "Source code" archives of GitHub do not include
llama.cpp, so they do not build.

## Contents of the build

The build includes the `common` library of llama.cpp. The chat layer uses this library for Jinja
templates and to parse tool calls. Most of the time of a first build is for this library.

The tests do not use a model file or a daemon. The chat template tests use a template that is in
the submodule. The other tests use only the build. `./scripts/build-test.sh [cpu|gpu]` builds the
project and then runs the tests, in one step.

`build.sh` gives all additional arguments to the CMake configure step. `CMAKE_BUILD_PARALLEL_LEVEL`
sets the number of jobs (the default is 4). With `-DLLAMAD_WARNINGS_AS_ERRORS=ON`, a warning in
the llamad source files stops the build. CI uses this option.

## GPU

The usual CMake flags of llama.cpp enable its GPU backends. `build.sh gpu` sets the flag for the
platform: Vulkan on Linux, Metal on macOS. `build.sh cpu` disables that backend. Vulkan is the
Linux backend that this project tests. Vulkan operates on Pascal cards (GTX 10xx), and CUDA 13
does not support these cards.

```sh
pacman -S vulkan-headers spirv-headers vulkan-icd-loader shaderc   # Linux only (Metal: no packages)
./scripts/build.sh gpu                  # -DGGML_VULKAN=ON or -DGGML_METAL=ON, in build-gpu/
./scripts/test.sh gpu
./scripts/smoke-test.sh gpu /absolute/path/to/model.gguf
```

The project does not include a model. `test.sh` does not use a model. `smoke-test.sh` must have a
model path as an argument. The path can be absolute or relative to your working directory. The
script runs one chat turn through `engine_smoke` at temperature 0.

On Linux, you must also have a Vulkan driver for the GPU that operates correctly. If the
inference runs on the CPU, the smoke test stops with an error. A device list does not test the
model load or the token generation.

### Device selection

With Vulkan, llama.cpp uses all discrete GPUs by default. It divides the layers of the model
between them, in proportion to the free memory of each card. If there is a discrete GPU,
llama.cpp does not use an integrated GPU. Thus, two 8 GB cards can hold a model that is too large
for one card. Apple silicon has one Metal device, `MTL0`, with unified memory. When the daemon
starts, it writes one line for each offload device.

```sh
./build-gpu/llamad --list-devices                    # names, types, free/total memory
./build-gpu/llamad --model M --devices Vulkan0       # this card only (MTL0 on macOS)
./build-gpu/llamad --model M --tensor-split 3,1      # 3:1 share, in device order
```
