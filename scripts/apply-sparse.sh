#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/apply-sparse.sh /path/to/llama.cpp
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
llama_dir=${1:?usage: apply-sparse.sh /path/to/llama.cpp}
base=9d57ce456c94d241dde672b2db9cf18879766568

if [[ $(git -C "$llama_dir" rev-parse HEAD) != "$base" ]]; then
    echo "Expected llama.cpp $base. Use a separate checkout at that commit." >&2
    exit 1
fi

combined="$repo_dir/patches/saker-combined.patch"
sparse="$repo_dir/patches/saker-sparse-attention.patch"
if git -C "$llama_dir" apply --reverse --check "$sparse" 2>/dev/null; then
    echo "Sparse patch is already applied."
    exit 0
fi
if git -C "$llama_dir" apply --reverse --check "$combined" 2>/dev/null; then
    git -C "$llama_dir" apply --check "$sparse"
    git -C "$llama_dir" apply "$sparse"
elif git -C "$llama_dir" apply --check "$combined" "$sparse"; then
    git -C "$llama_dir" apply "$combined" "$sparse"
else
    echo "The checkout has conflicting changes. Nothing was applied." >&2
    exit 1
fi
echo "Applied experimental sparse attention. It is disabled unless SAKER_SPARSE_TOKENS is set."
