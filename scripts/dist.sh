#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help ]]; then
    echo "Usage: $0 [OUTPUT_DIR]"
    echo "Writes OUTPUT_DIR/llamad-<version>.tar.gz and prints its path. OUTPUT_DIR defaults to dist."
    echo "The archive holds the commit at HEAD and the llama.cpp commit that HEAD pins."
    echo "GitHub's source archives leave out the submodule, so they cannot build."
    exit 0
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=${1:-$root/dist}

# The version of the commit that the archive holds, so that the name always matches the contents.
version=$(git -C "$root" show HEAD:CMakeLists.txt | sed -n 's/^project(llamad VERSION \([0-9][0-9.]*\).*/\1/p')
if [[ -z $version ]]; then
    echo "error: no 'project(llamad VERSION x.y.z' line in CMakeLists.txt at HEAD" >&2
    exit 1
fi
name=llamad-$version

# git archive reads the commit, so edits and new files in the working tree stay out.
if [[ -n $(git -C "$root" status --porcelain) ]]; then
    echo "warning: the working tree has uncommitted changes; the archive holds HEAD only" >&2
fi

submodule=$root/third_party/llama.cpp
pin=$(git -C "$root" rev-parse HEAD:third_party/llama.cpp)
if [[ ! -e $submodule/.git ]] || ! git -C "$submodule" cat-file -e "$pin^{commit}" 2>/dev/null; then
    echo "error: llama.cpp commit $pin is not in $submodule" >&2
    echo "Run: git -C '$root' submodule update --init --recursive" >&2
    exit 1
fi

staging=$(mktemp -d)
trap 'rm -rf "$staging"' EXIT

# Flags that both GNU tar and the bsdtar of macOS accept.
git -C "$root" archive --prefix="$name/" HEAD | tar -x -C "$staging"
git -C "$submodule" archive --prefix="$name/third_party/llama.cpp/" "$pin" | tar -x -C "$staging"

mkdir -p "$out"
out=$(cd -- "$out" && pwd)
tar -c -z -f "$out/$name.tar.gz" -C "$staging" "$name"
echo "$out/$name.tar.gz"
