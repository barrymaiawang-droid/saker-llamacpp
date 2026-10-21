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
selectors="$repo_dir/patches/saker-quest-sparq.patch"
if git -C "$llama_dir" apply --reverse --check "$selectors" 2>/dev/null; then
    echo "Sparse attention with Quest and SparQ is already applied."
    exit 0
fi
if git -C "$llama_dir" apply --reverse --check "$sparse" 2>/dev/null; then
    pending=("$selectors")
elif git -C "$llama_dir" apply --reverse --check "$combined" 2>/dev/null; then
    pending=("$sparse" "$selectors")
else
    pending=("$combined" "$sparse" "$selectors")
fi
# Check dependent patches against a private index, then apply one aggregate diff.
# Snapshot only affected paths; the user's real index and unrelated files stay intact.
tmp_dir=$(mktemp -d)
trap 'rm -rf -- "$tmp_dir"' EXIT
index="$tmp_dir/index"
GIT_INDEX_FILE="$index" git -C "$llama_dir" read-tree HEAD
mapfile -t paths < <(git -C "$llama_dir" apply --numstat "${pending[@]}" | cut -f3 | sort -u)
for path in "${paths[@]}"; do
    if [[ -e "$llama_dir/$path" ]] || git -C "$llama_dir" ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then
        GIT_INDEX_FILE="$index" git -C "$llama_dir" add -A -- "$path"
    fi
done
before=$(GIT_INDEX_FILE="$index" git -C "$llama_dir" write-tree)
for patch in "${pending[@]}"; do
    if ! GIT_INDEX_FILE="$index" git -C "$llama_dir" apply --cached "$patch"; then
        echo "The checkout has conflicting changes. Nothing was applied." >&2
        exit 1
    fi
done
GIT_INDEX_FILE="$index" git -C "$llama_dir" diff --cached --binary "$before" > "$tmp_dir/combined.patch"
git -C "$llama_dir" apply --check "$tmp_dir/combined.patch"
git -C "$llama_dir" apply "$tmp_dir/combined.patch"
echo "Applied experimental sparse attention (representative, quest, sparq). Disabled unless SAKER_SPARSE_TOKENS is set."
