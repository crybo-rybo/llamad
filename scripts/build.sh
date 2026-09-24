#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help ]]; then
    echo "Usage: $0 [cpu|gpu] [CMake configure arguments...]"
    echo "Builds build-cpu or build-gpu, offloading with Vulkan on Linux and Metal on macOS."
    echo "On macOS, run build-deps-macos.sh once first. CMAKE_BUILD_PARALLEL_LEVEL defaults to 4."
    exit 0
fi

mode=${1:-cpu}
if (( $# )); then shift; fi
case "$mode" in
    cpu) offload=OFF ;;
    gpu) offload=ON ;;
    *) echo "Expected cpu or gpu; see $0 --help" >&2; exit 2 ;;
esac

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build="$root/build-$mode"

if [[ $(uname -s) == Darwin ]]; then
    # Without the prefix CMake would pick up a Homebrew gRPC, which is libc++ and fails at link.
    if [[ ! -d "$root/build-deps/prefix" ]]; then
        echo "build-deps/prefix not found; run $root/scripts/build-deps-macos.sh first" >&2
        exit 1
    fi
    # Apple clang compiles llama.cpp's C and Objective-C Metal sources, which GCC's C compiler
    # rejects; g++-16 compiles every C++ source, because reflection needs it. Accelerate's
    # headers do not compile with GCC either, and it only accelerates prompt processing.
    platform=(-DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=g++-16
        -DGGML_METAL="$offload" -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF
        -DCMAKE_PREFIX_PATH="$root/build-deps/prefix;$(brew --prefix openssl@3)")
    # GCC's -mcpu=native on Apple silicon leaves out FP16 vector arithmetic, which ggml's F16
    # kernels need (the KV cache is F16, so attention runs through them), and ggml's feature
    # probes hand GCC a -mcpu=native+feature it rejects. Name the features the CPU reports.
    arch=armv8.2-a
    for feature in fp16:FP16 dotprod:DotProd i8mm:I8MM bf16:BF16; do
        if [[ $(sysctl -n "hw.optional.arm.FEAT_${feature#*:}" 2>/dev/null) == 1 ]]; then
            arch+="+${feature%%:*}"
        fi
    done
    platform+=(-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="$arch")
else
    platform=(-DGGML_VULKAN="$offload")
fi

cmake -S "$root" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    "${platform[@]}" -DGGML_CUDA=OFF "$@"
# Build every test, and fail if missing gRPC/Protobuf leaves only the engine targets.
cmake --build "$build" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}" \
    --target all llamad llamad-chat
