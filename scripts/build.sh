#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help ]]; then
    echo "Usage: $0 [cpu|gpu] [CMake configure arguments...]"
    echo "Builds build-cpu or build-gpu (Vulkan). CMAKE_BUILD_PARALLEL_LEVEL defaults to 4."
    exit 0
fi

mode=${1:-cpu}
if (( $# )); then shift; fi
case "$mode" in
    cpu) vulkan=OFF ;;
    gpu) vulkan=ON ;;
    *) echo "Expected cpu or gpu; see $0 --help" >&2; exit 2 ;;
esac

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build="$root/build-$mode"

cmake -S "$root" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN="$vulkan" -DGGML_CUDA=OFF "$@"
# Build every test, and fail if missing gRPC/Protobuf leaves only the engine targets.
cmake --build "$build" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}" \
    --target all llamad llamad-chat
