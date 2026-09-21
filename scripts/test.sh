#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help ]]; then
    echo "Usage: $0 [cpu|gpu] [CTest arguments...]"
    echo "Runs tests in an existing build; no model or GPU is needed."
    echo "For model inference, use smoke-test.sh <cpu|gpu> <model.gguf>."
    exit 0
fi

mode=${1:-cpu}
if (( $# )); then shift; fi
case "$mode" in
    cpu|gpu) ;;
    *) echo "Expected cpu or gpu; see $0 --help" >&2; exit 2 ;;
esac

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ctest --test-dir "$root/build-$mode" --output-on-failure --no-tests=error "$@"
