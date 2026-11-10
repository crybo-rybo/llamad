#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help ]]; then
    echo "Usage: $0 [cpu|gpu] [CMake configure arguments...]"
    echo "Builds build-cpu or build-gpu. The gpu mode offloads with Metal on macOS."
    echo "On Linux, it uses CUDA if it finds nvcc, else Vulkan."
    echo "It looks for nvcc on PATH, in \$CUDA_PATH/bin, /usr/local/cuda/bin and /opt/cuda/bin."
    echo "Your CMake arguments come last, so -DGGML_CUDA=OFF -DGGML_VULKAN=ON forces Vulkan."
    echo "CMAKE_BUILD_PARALLEL_LEVEL defaults to 4."
    exit 0
fi

mode=${1:-cpu}
if (( $# )); then shift; fi
case "$mode" in
    cpu|gpu) ;;
    *) echo "Expected cpu or gpu; see $0 --help" >&2; exit 2 ;;
esac

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# llama.cpp enables no GPU backend by default except Metal, so the script chooses one.
# Compile only one: with both, a single NVIDIA card shows as CUDA0 and also as Vulkan0.
if [[ $mode == cpu ]]; then
    backend=(-DGGML_METAL=OFF -DGGML_VULKAN=OFF -DGGML_CUDA=OFF)
elif [[ $(uname -s) == Darwin ]]; then
    backend=(-DGGML_METAL=ON)
    echo "build.sh: GPU backend Metal" >&2
else
    nvcc=$(command -v nvcc || true)
    for candidate in ${CUDA_PATH:+"$CUDA_PATH/bin/nvcc"} /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc; do
        if [[ -z $nvcc && -x $candidate ]]; then nvcc=$candidate; fi
    done
    if [[ -n $nvcc ]]; then
        backend=(-DGGML_CUDA=ON -DGGML_VULKAN=OFF -DCMAKE_CUDA_COMPILER="$nvcc")
        echo "build.sh: GPU backend CUDA ($nvcc)" >&2
    else
        backend=(-DGGML_VULKAN=ON -DGGML_CUDA=OFF)
        echo "build.sh: GPU backend Vulkan (nvcc not found)" >&2
    fi
fi

cmake -S "$root" -B "$root/build-$mode" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    "${backend[@]}" "$@"
cmake --build "$root/build-$mode" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
