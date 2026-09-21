#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help ]]; then
    echo "Usage: $0 [CMake configure arguments...]"
    echo "Builds Doxygen HTML in build-docs/html; needs CMake and Doxygen only."
    exit 0
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build="$root/build-docs"

cmake -S "$root/docs" -B "$build" "$@"
cmake --build "$build" --target llamad_docs
echo "Documentation: $build/html/index.html"
