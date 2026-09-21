#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help ]]; then
    echo "Usage: $0 [cpu|gpu] [model.gguf]"
    echo "Runs chat inference in an existing build, using the local Qwen2.5 0.5B model by default."
    echo "GPU mode requires Vulkan offload; CPU fallback fails the test."
    exit 0
fi

mode=${1:-cpu}
case "$mode" in
    cpu) ngl=0 ;;
    gpu) ngl=99 ;;
    *) echo "Expected cpu or gpu; see $0 --help" >&2; exit 2 ;;
esac
if (( $# > 2 )); then
    echo "Usage: $0 [cpu|gpu] [model.gguf]" >&2
    exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
model=${2:-"$root/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"}
if [[ ! -f "$model" ]]; then
    echo "Model not found: $model (pass a local GGUF as the second argument)" >&2
    exit 1
fi

log=$(mktemp)
trap 'rm -f -- "$log"' EXIT
"$root/build-$mode/tests/engine_smoke" "$model" --ngl "$ngl" \
    --chat --temp 0 --max-tokens 32 "Count from one to three." 2>&1 | tee "$log"

if [[ "$mode" == gpu ]] && ! grep -Eq '^\[llamad\] device Vulkan[0-9]+:' "$log"; then
    echo "FAIL: inference did not report Vulkan GPU offload." >&2
    exit 1
fi
if ! grep -Eq 'finish: (Eog|Length) .*completion_tokens: [1-9][0-9]* ' "$log"; then
    echo "FAIL: inference did not generate tokens successfully." >&2
    exit 1
fi
echo "PASS: $mode model load and chat inference."
