# Path B: two-branch workflow and integration plan

Set up 2026-10-19 after the user pointed out that two isolated directory copies have no merge path
(they isolate the writers, but the two fixes are complementary and must be recombined to reach 40).

## Branches

| workstream | tree | branch | agent |
|---|---|---|---|
| (a) shortlist head for the MTP draft lane | /home/hycris/llama.cpp-qwen38-a | `saker-shortlist` | bc2c67d3 |
| (b) multi-position verify compute | /home/hycris/llama.cpp-qwen38-b | `saker-verify` | 9e72da9f |

Both trees were created by `cp -a` from /home/hycris/llama.cpp-qwen38 and were on **detached HEAD** at
`9d57ce456`. They carry the same pre-existing uncommitted instrumentation
(`common/speculative.cpp` SAKER_SPEC_TRACE, `ggml/src/ggml-vulkan/ggml-vulkan.cpp` SAKER_VK_TRACE and
SAKER_FORCE_QY, plus the `*.orig-saker` backups). Branches were created with `git checkout -b`, which
does not touch the working tree, so it was safe to do while both agents were already editing.

## Why the merge is expected to be clean

Both branches share commit `9d57ce456` as their merge base, and both carry the *identical*
instrumentation delta. A 3-way merge therefore sees the same addition on both sides and resolves it
without conflict; only each agent's unique changes need to merge. The conflict risk is
`ggml/src/ggml-vulkan/ggml-vulkan.cpp`, which **both** agents are editing - they have been told to
keep edits localized and to avoid that file if the fix can be done elsewhere.

## Integration procedure (parent owns this, one writer)

    # 1. clean merge tree (clone uses HEAD, so the parent's dirty working tree does not leak in)
    git clone /home/hycris/llama.cpp-qwen38 /home/hycris/llama.cpp-qwen38-merge
    cd /home/hycris/llama.cpp-qwen38-merge

    # 2. pull both branches in by path
    git fetch /home/hycris/llama.cpp-qwen38-a saker-shortlist:saker-shortlist
    git fetch /home/hycris/llama.cpp-qwen38-b saker-verify:saker-verify

    # 3. combine
    git checkout -b saker-combined 9d57ce456
    git merge --no-edit saker-shortlist
    git merge --no-edit saker-verify

    # 4. build and measure the COMBINED result
    cmake --build build --target llama-server -j 20

## Expected outcomes

| outcome | round | tok/s |
|---|---|---|
| baseline | 72.0 ms | 34.4 (measured, delivered) |
| (a) alone | 64.9 ms | 37.9 |
| (b) alone | 67.8 ms | 36.3 |
| **(a) + (b)** | **60.7 ms** | **40.5** |

Each branch must be measured on its own before the merge (that is what "then test then merge" means),
so a regression can be attributed to a specific branch rather than to the combination.

## ⚠️ BUILD LANDMINE — read before building any copied tree

`cp -a` of a configured llama.cpp tree inherits a `build/` directory whose CMake cache and binary
RUNPATH point back at the ORIGINAL tree. Symptom: `cmake --build build` silently compiles the
*parent's* sources and the binary loads the *parent's* `.so` files, so you measure parent code while
believing you are measuring your change. This wasted two full experiment rounds in tree b and
produced wrong near-zero deltas that looked like a real (null) result.

Check both, every time, before trusting a measurement:

    grep -m1 CMAKE_HOME_DIRECTORY build/CMakeCache.txt        # must name YOUR tree
    readelf -d build/bin/llama-server | grep -i runpath       # must point at YOUR build/bin

Observed 2026-10-19: tree b's `build/` was wired to /home/hycris/llama.cpp-qwen38 (WRONG, worked
around by building in a separate `build-b/`); tree a's `build/` was correctly self-wired.

**Consequence for the merge:** the merge tree must NOT be a `cp -a` copy. Use `git clone` (which
carries no build dir, since it is gitignored) and configure a fresh build, then re-run both checks
above before measuring.
