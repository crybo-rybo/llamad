#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help ]]; then
    echo "Usage: $0 [cpu|gpu] [CMake configure arguments...]"
    echo "Builds, then runs CTest. For inference with a model, use smoke-test.sh."
    exit 0
fi

scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
"$scripts/build.sh" "$@"
"$scripts/test.sh" "${1:-cpu}"
