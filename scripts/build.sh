#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help ]]; then
    echo "Usage: $0 [cpu|gpu] [CMake configure arguments...]"
    echo "Builds build-cpu or build-gpu, offloading with Vulkan on Linux and Metal on macOS."
    echo "CMAKE_BUILD_PARALLEL_LEVEL defaults to 4."
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

if [[ $(uname -s) == Darwin ]]; then
    platform=(-DGGML_METAL="$offload")
else
    platform=(-DGGML_VULKAN="$offload")
fi

cmake -S "$root" -B "$root/build-$mode" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    "${platform[@]}" -DGGML_CUDA=OFF "$@"
cmake --build "$root/build-$mode" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
