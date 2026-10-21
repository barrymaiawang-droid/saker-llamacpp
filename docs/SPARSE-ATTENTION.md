# Experimental sparse attention for Swift-27B

This is an opt-in approximation for the `qwen35` model path, supplied as an additional patch to llama.cpp `9d57ce456c94d241dde672b2db9cf18879766568` after Saker's existing combined patch. It is disabled by default.

This page describes the original `representative` selector. The apply script also installs `saker-quest-sparq.patch`; see [Quest and SparQ](QUEST-SPARQ.md) for the new modes, updated validation and quality results. Leaving `SAKER_SPARSE_MODE` unset retains the representative selector.

**This is not native QSA, is not lossless, and has no measured Strix Halo speed or model-quality result yet.** Swift-27B has no trained Flash-Next QSA indexer to transplant. The original Saker quality and throughput measurements do not apply to this mode.

## Mechanism

For each full-attention layer and each query token:

1. Partition the existing KV cache into blocks of 16 physical slots.
2. Use the latest causally visible key in each block as a representative. Score it against the current query heads and take the maximum head score. This is a training-free heuristic; it can miss a relevant token elsewhere in the block.
3. Protect blocks containing the earliest and most recent visible positions, then choose the remaining blocks by score.
4. Gather the selected K/V blocks and their original causal mask into compact tensors. Flash attention runs over these tensors, not over the entire history with most entries masked.

With `SAKER_SPARSE_TOKENS=2048`, the expensive attention step handles at most 128 blocks / 2,048 cache slots per query, including the protected blocks. The full KV cache is retained. GDN layers and the existing draft vocabulary shortlist remain in place.

Representatives come directly from the cache. There is no separate pooled-key state to rebuild or invalidate. Block metadata is refreshed from current positions and masks on each call, including cache reuse and speculative rollback. Representatives are chosen separately for every query, so a later token in an MTP verification batch cannot influence an earlier query's selection. The gathered token mask also excludes future, empty and other-sequence slots.

**Total decode cost is still not constant.** The selector reads one key per block, and host metadata/mask processing grows with context. Selection, gathering, extra launches and per-query MTP attention can cost more than dense attention at some lengths. Performance must be measured.

## Scope and fallback

The sparse path requires all of:

- `qwen35` architecture, including its MTP draft graph;
- Flash attention enabled and F16 K/V caches;
- one unique sequence in the microbatch;
- at most eight query tokens in the microbatch;
- a padded KV view longer than the token budget, divisible by 16.

Other cases use existing dense attention. Bulk prefill stays dense. A short prefill tail can use the sparse path if it meets the conditions above. Quantized KV caches and concurrent multi-sequence batches are not accelerated by this version. The threshold uses the padded KV length; at short actual histories all visible blocks can still fit in the budget.

MTP verification uses the approximate target model. It does not restore the original dense model's output distribution. Outputs can also differ around the dense/sparse batch-size boundary. Keep the same batch settings when comparing runs.

## Apply and build

Use a separate llama.cpp checkout at the pinned base:

```bash
git clone https://github.com/ggml-org/llama.cpp.git llama-saker
cd llama-saker
git checkout 9d57ce456c94d241dde672b2db9cf18879766568
bash /path/to/saker-llamacpp/scripts/apply-sparse.sh "$PWD"

cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON -DGGML_HIP=OFF -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build --target llama-server test-saker-sparse -j 8

ctest --test-dir build -R '^test-saker-sparse$' --output-on-failure
build/bin/test-saker-sparse Vulkan0
```

The apply script accepts a clean pinned checkout or one already carrying `saker-combined.patch`. It checks before writing and recognizes an already applied sparse patch. It does not reset local changes or change branches.

The Vulkan test deliberately fails if an operation cannot execute on the requested backend. It does not silently fall back to CPU. Use the Vulkan device name reported by your build if it differs from `Vulkan0`.

## Run

```bash
SAKER_SHORTLIST=3584 \
SAKER_SPARSE_TOKENS=2048 \
SAKER_SPARSE_TRACE=1 \
build/bin/llama-server \
  -m /path/to/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf \
  -ngl 99 -fa on -ctk f16 -ctv f16 -c 32768 -np 1 \
  --host 127.0.0.1 --port 8080 --temp 0 --no-webui \
  --spec-type draft-mtp --spec-draft-n-max 2
```

Supply a prompt longer than the budget. Trace lines such as `saker-sparse: layer=3 queries=3 cache=... selected=2048` show graph construction for a sparse verification batch. These can also appear during startup reservation. An additional `saker-sparse: active queries=3 cache=... selected=2048` line confirms that an actual batch supplied sparse inputs. A reused graph reports activation once. Omit `SAKER_SPARSE_TRACE` after checking activation.

| Variable | Default | Meaning |
|---|---:|---|
| `SAKER_SPARSE_TOKENS` | `0` | Disabled. Positive values must be multiples of 16, at most 8192. |
| `SAKER_SPARSE_RECENT_BLOCKS` | `16` | Protect the blocks with the most recent visible positions. |
| `SAKER_SPARSE_SINK_BLOCKS` | `1` | Protect the blocks with the earliest visible positions. |
| `SAKER_SPARSE_TRACE` | unset | Any set value enables graph trace output. |

The budget must have room for the configured protected block counts plus at least one retrieved block. With default protection, the minimum budget is 288. Protection is measured in **blocks**, not tokens; fragmented physical storage can change how many recent tokens those blocks contain. These options are read once per process. Restart the server to change them. Unset `SAKER_SPARSE_TOKENS` or set it to `0` to restore the existing Saker attention path.

## Compare on Strix Halo

```bash
python3 /path/to/saker-llamacpp/scripts/benchmark-sparse.py \
  --server build/bin/llama-server \
  --model /path/to/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf \
  --prompt-file /path/to/real-long-code-or-document.txt \
  --depths 4096 16384 32768 --predict 128 --repetitions 3 \
  --budget 2048 --output results/sparse-2048.jsonl
```

This starts and stops its own localhost server for dense, representative, Quest and SparQ in AR and MTP modes. To limit it to the original experiment, pass `--selectors representative --dense-layers 0`. It uses identical token IDs, a warm-up at each depth, a fixed generation length, and fresh prompt evaluation for each measured request. It refuses an undersized corpus rather than repeating it, verifies actual evaluated context length, and records timings, generated text, hashes, settings and commands. Trace logging is enabled to confirm actual sparse input activation before recording results. Keep the same model, driver, clocks and power settings; repeat with `--reverse-arms` if thermal drift is material.

The input is a raw text-completion corpus, not an automatically formatted chat. Use representative workloads; repetitive text can greatly inflate MTP acceptance. JSONL outputs contain the supplied model path and generated text, so review before sharing.

Before using this mode for real work, evaluate long-document retrieval, multi-document reasoning and task accuracy against dense attention. For accounting, include checks that require combining distant ledger entries. Measure any quality loss at each proposed budget. Standard bulk-prefill perplexity alone can miss the change because that path remains dense; a teacher-forced decode evaluation must actually use small query batches. Output equality on a few prompts is not a quality guarantee.

## Validation supplied with the patch

- Full CPU `llama-server` build on the pinned base plus both Saker patches.
- Twelve GGML graph cases, each with three cache updates: one, three and eight queries; full-budget and pruned attention; contiguous and fragmented storage. Checks include independent scalar selection/attention references, protected blocks, unique block selection, full-budget equivalence, changing positions, and poisoning future slots without changing earlier output.
- Configuration checks for default/off, valid budget, malformed numbers, negatives, overflow, nonmultiples and insufficient budget.
- A small random-weight Qwen35 fixture and server smoke runner to exercise the real hybrid/MTP graph integration. This fixture tests execution, not language quality or performance.

The CPU build and all twelve reference cases passed. Both fixture variants passed all four server modes (dense AR, dense MTP, sparse AR and sparse MTP): 32 completion requests covering fresh prompts, cache resets, prefix reuse, short-context fallback, and accepted/rejected drafts. Actual sparse activation was confirmed for single-query decode and three-query MTP verification. The benchmark runner also completed eight measured fixture requests across two context lengths and all four modes; those timings are only a harness check, not a Swift-27B speed result. The packaged patch applied cleanly to the pinned base, and a second application was correctly recognized as already applied.

```bash
# numpy is needed; gguf is imported from this llama.cpp checkout.
python3 /path/to/saker-llamacpp/tests/make-tiny-qwen35.py \
  --llama-dir . --output /tmp/saker-tiny.gguf
python3 /path/to/saker-llamacpp/tests/smoke-sparse.py \
  --server build/bin/llama-server --model /tmp/saker-tiny.gguf \
  --log-dir /tmp/saker-smoke --expect-mtp-acceptance zero

# Exercise accepted drafts as well as rejection/rollback.
python3 /path/to/saker-llamacpp/tests/make-tiny-qwen35.py \
  --llama-dir . --output /tmp/saker-predictable.gguf --predictable
python3 /path/to/saker-llamacpp/tests/smoke-sparse.py \
  --server build/bin/llama-server --model /tmp/saker-predictable.gguf \
  --log-dir /tmp/saker-smoke-accepted --expect-mtp-acceptance some
```

The existing draft-head shortlist emits a nonfatal `get_logits_ith: invalid logits id` diagnostic during some fixture prefill transitions, including the sparse-disabled MTP control. This pre-existing shortlist issue is not changed by the sparse patch; do not interpret these smoke checks as a clean audit of the original Saker patch.

No Strix Halo / Vulkan execution or real Swift-27B quality test was possible in the implementation environment. This is an experimental implementation for those measurements, not a claim of Strixite-equivalent speed or accuracy. The pre-existing Saker Vulkan retune defect documented in the main README is unchanged.

Implementation assistance: OpenAI Codex. No upstream llama.cpp submission is included.
