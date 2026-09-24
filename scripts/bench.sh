#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || $# -lt 2 ]]; then
    echo "Usage: $0 <cpu|gpu> <model.gguf> [engine_bench options...]"
    echo "Measures prefill and decode tokens per second in an existing build, and appends each"
    echo "scenario's median to build-<mode>/bench.tsv with the commit it was measured at."
    echo "CPU mode offloads nothing; see engine_bench --help for the scenarios and options."
    [[ ${1:-} == --help ]] && exit 0 || exit 2
fi

mode=$1
model=$2
shift 2
case "$mode" in
    cpu) ngl=0 ;;
    gpu) ngl=99 ;;
    *) echo "Expected cpu or gpu; see $0 --help" >&2; exit 2 ;;
esac

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ! -f "$model" ]]; then
    echo "Model not found: $model" >&2
    exit 1
fi

commit=$(git -C "$root" describe --always --dirty)
log="$root/build-$mode/bench.tsv"
[[ -f "$log" ]] || printf 'date\tcommit\tmode\tmodel\tscenario\ttokens\ttok_per_s\tmin\tmax\n' > "$log"

"$root/build-$mode/tests/engine_bench" "$model" --ngl "$ngl" "$@" | tee /dev/stderr | tail -n +2 |
    while IFS= read -r row; do
        printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%F\ %T)" "$commit" "$mode" "$(basename "$model")" "$row"
    done >> "$log"
