# Quest and SparQ for Saker

`saker-quest-sparq.patch` adds two experimental selectors on top of the original sparse patch at llama.cpp `9d57ce456c94d241dde672b2db9cf18879766568`. Both operate on Qwen35 full-attention layers, including the MTP draft graph and small verification batches. GDN layers remain unchanged. All modes retain the complete KV cache.

These modes change the model's attention. They are disabled by default. CPU correctness and small-model checks passed; Swift-27B quality and Strix Halo/Vulkan runtime performance have not been measured.

## Algorithms and adaptations

| Mode | Ranking | Selection unit | Additional persistent state |
|---|---|---|---|
| `representative` | Latest visible key in each page, maximum score across all query heads | Shared 16-slot pages | None |
| `quest` | Sum of signed per-channel min/max bounds, maximum across the query heads sharing each KV head | Separate 16-slot pages per KV group | Two F16 bounds per channel per page |
| `sparq` | Top query channels, approximate softmax, then aggregate within each KV group | Separate tokens per KV group | None |

### Quest

For each page, cache the minimum and maximum of every key channel. Given a query, score a page with:

`sum_d max(q[d] * page_min[d], q[d] * page_max[d])`

This covers all keys in the page and handles negative query components. Each KV group selects pages using the maximum bound across its query heads. This GQA grouping is a Saker adaptation; it does not select a separate page set for every query head.

Only pages touched by a cache write are refreshed during ordinary decode, by reducing the 16 stored F16 keys again. Recomputing the written page allows bounds to shrink after overwrites and speculative rollback. Prefill also updates metadata. Cache clears, state loads and RoPE shifts invalidate metadata; the next graph rebuilds it. Metadata is marked valid only after successful graph submission, and graph reuse checks the update-list size.

The fast path requires each query's visible slots to form a physical prefix. Its last partial page is always selected, so future tokens in an MTP batch cannot affect page ranking through that page's bounds. Every gathered key also retains its original causal mask. Fragmented visibility falls back to dense attention. This restriction avoids unbounded work to repair partially visible pages.

The first two **full-attention** layers stay dense by default; GDN layers do not count toward this number. The paper's early-layer protection motivates this policy, but its exact effect on Swift-27B still needs measurement.

### SparQ

Within each KV group:

1. Sum absolute query components across the group's query heads and choose the top `r` channels.
2. Gather those channels from all cached keys. For each query head, use `tau = sqrt(D * selected_L1 / full_L1)` and compute a causally masked approximate softmax. Zero norms are guarded.
3. Aggregate the approximate probabilities across the group and select the top token positions, including protected recent and initial tokens.
4. Gather full K/V vectors for those positions and run ordinary Flash attention over the compact cache.

The probability aggregation uses a mean, which has the same ranking as a sum for a fixed group size. Following the paper's GQA variant, there is no mean-value reallocation step. This is specifically a GQA implementation for Qwen35, not the paper's separate MHA interpolation variant.

CPU and Vulkan F16/F32 `GET_ROWS` now support strided source elements, so selection reads chosen channels directly from the existing key layout. Other backends use a contiguous copy before channel gathering. There is **no duplicate transposed key cache**: GPU memory transactions can fetch more than the selected scalars, so the paper's ideal bandwidth reduction is not a throughput prediction for this implementation.

Budgets of 1024 or more selected items use argsort plus a slice, because the pinned Vulkan top-k implementation cannot handle those output counts. Large SparQ budgets therefore carry a full-sort cost.

## Build or upgrade

Use a separate llama.cpp checkout at the pinned base. The script handles a clean checkout, Saker's combined patch, or the previous representative-only sparse patch. It preflights the dependent patches in a temporary index, then applies one aggregate diff. It does not change the user's index or reset files.

```bash
git checkout 9d57ce456c94d241dde672b2db9cf18879766568
bash /path/to/saker-llamacpp/scripts/apply-sparse.sh "$PWD"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
  -DGGML_VULKAN=ON -DGGML_HIP=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build --target llama-server test-saker-sparse test-saker-selectors \
  test-backend-ops saker-quality -j 8
ctest --test-dir build -R '^test-saker-(sparse|selectors)$' --output-on-failure
python3 /path/to/saker-llamacpp/tests/check-config.py build/bin/test-saker-sparse

# Require every selector operation to execute on this backend; no CPU fallback in these tests.
build/bin/test-saker-selectors Vulkan0
build/bin/test-backend-ops -b Vulkan0 -o GET_ROWS
```

The Vulkan shader change compiled for F16, F32, BF16 and I32 inputs locally. No Vulkan device was available for execution. The backend tests above are required before treating the Vulkan path as verified on Strix Halo. Other GPU backends are untested.

## Run

```bash
SAKER_SHORTLIST=3584 \
SAKER_SPARSE_MODE=quest \
SAKER_SPARSE_TOKENS=2048 \
SAKER_SPARSE_TRACE=1 \
build/bin/llama-server \
  -m /path/to/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf \
  -ngl 99 -fa on -ctk f16 -ctv f16 -c 32768 -np 1 \
  --host 127.0.0.1 --port 8080 --temp 0 --no-webui \
  --spec-type draft-mtp --spec-draft-n-max 2
```

For SparQ, set `SAKER_SPARSE_MODE=sparq`, `SAKER_SPARQ_CHANNELS=32` and, for the same early-layer protection, `SAKER_SPARSE_DENSE_LAYERS=2`.

Look for both a layer trace ending in `mode=quest` or `mode=sparq` and `saker-sparse: active queries=...`. The latter confirms real input execution rather than startup graph reservation. Disable tracing after checking activation.

| Environment variable | Default | Meaning |
|---|---:|---|
| `SAKER_SPARSE_TOKENS` | `0` | Off. Positive budgets must be multiples of 16, at most 8192, with room for protection and retrieval. |
| `SAKER_SPARSE_MODE` | `representative` | `representative`, `quest`, or `sparq`. |
| `SAKER_SPARSE_DENSE_LAYERS` | Quest: `2`; others: `0` | Number of initial full-attention layers to leave dense. |
| `SAKER_SPARSE_RECENT_BLOCKS` | `16` | Recent protected pages for representative/Quest. |
| `SAKER_SPARSE_SINK_BLOCKS` | `1` | Initial protected pages for representative/Quest. |
| `SAKER_SPARQ_CHANNELS` | `32` | Channels selected per KV group; clamped to the head dimension. |
| `SAKER_SPARQ_RECENT_TOKENS` | `256` | Recent protected tokens for SparQ. |
| `SAKER_SPARQ_SINK_TOKENS` | `16` | Initial protected tokens for SparQ. |
| `SAKER_SPARSE_TRACE` | unset | Any set value enables tracing. |

Options are read once per process. `SAKER_SPARSE_TOKENS=0` restores dense attention. With default protection, the minimum enabled budget is 288. Protection is included in the budget. Quest reserves space for its partial page even when recent-page protection is zero.

Common eligibility limits: `qwen35`, causal Flash attention, F16 K/V, one unique sequence in the microbatch, at most eight query tokens, and padded KV length above the budget. Bulk prefill and unsupported inputs use dense attention. Quest additionally requires a single allocated KV stream and prefix visibility. Approximate MTP verification does not recover the original dense model's distribution. Selection can change at the dense/sparse batch-size boundary.

## Measurements and tests

CPU validation on the pinned base plus all three patches:

- Original 12 representative-selector cases, with three cache/mask updates each.
- 24 Quest/SparQ reference cases across one/three/eight queries, full/pruned budgets and varied physical layouts; plus all-channel and zero-query SparQ cases. Independent scalar selection and attention checks include signed bounds, interior keys, GQA groups, unique selections, causal poisoning and full-budget equivalence.
- Incremental Quest metadata checks, including shrinking bounds after an overwritten page and unchanged untouched pages.
- 121 `GET_ROWS` backend cases, including strided F16/F32 inputs; 19 configuration cases.
- 64 server completions across dense/representative/Quest/SparQ, AR/MTP, accepted/rejected drafts, prefix reuse, reset and short-context fallback.
- Saved-state restoration checks in all four modes, across two prompt windows each.
- Benchmark runner: 16 measured fixture requests over two depths and eight arms. These random-weight fixture timings are harness checks, not model performance results.

The fixture runs use `--no-warmup`. The existing Saker MTP startup warmup failed on the tiny fixture even in dense mode; its nonfatal shortlist `invalid logits id` diagnostic also remains. The pre-existing Vulkan GEMM retune issue in the main README remains outside this change.

### Trained proxy quality

Qwen3.5-0.8B Q8_0, WikiText-2 test text, two independent 4096-token prefixes followed by 128 teacher-forced one-token decode scores each: **256 scored tokens per arm**. Every sparse arm left the first two full-attention layers dense; SparQ used 32 channels. Sparse activation was confirmed. Hashes and exact results are in [the measurement record](../reports/QUEST-SPARQ-QUALITY-20261021.json).

| Mode | Budget | Perplexity | Top-1 agreement with dense |
|---|---:|---:|---:|
| Dense | Full | 8.6799 | 100% |
| Representative | 512 | 8.9012 | 89.45% |
| Quest | 512 | 9.1055 | 92.97% |
| SparQ | 512 | 8.6502 | 97.27% |
| Quest | 2048 | 8.6461 | 98.44% |
| SparQ | 2048 | 8.6722 | 99.22% |

This small sample is a regression check, not evidence of a quality improvement. Slightly lower perplexity can arise from approximation on a small sample. It does not test Swift-27B, long-range retrieval, accounting accuracy or downstream tasks. No end-to-end speed claim follows from these measurements.

```bash
python3 /path/to/saker-llamacpp/scripts/quality-sparse.py \
  --evaluator build/bin/saker-quality \
  --model /path/to/Qwen3.5-0.8B-Q8_0.gguf \
  --corpus /path/to/wiki.test.raw --prefix 4096 --score 128 --windows 2 \
  --budgets 512 2048 --channels 32 --dense-layers 2 \
  --output-dir results/quality

python3 /path/to/saker-llamacpp/scripts/benchmark-sparse.py \
  --server build/bin/llama-server --model /path/to/Swift-27B-mtp.gguf \
  --prompt-file /path/to/real-long-document.txt \
  --depths 4096 16384 32768 --predict 128 --repetitions 3 \
  --selectors representative quest sparq --budget 2048 --channels 32 --dense-layers 2 \
  --output results/compare-2048.jsonl
```

The quality evaluator is CPU-only and deliberately evaluates small decode batches. Bulk-prefill perplexity would bypass these selectors. The throughput runner checks activation and evaluated context length, uses identical tokens and includes dense controls. Sparse attention still has context-dependent scoring, host mask processing, gathering and kernel-launch costs; it may be slower at some lengths.

## Primary references

- [Quest paper](https://arxiv.org/abs/2406.10774): channel-wise page bounds and page selection.
- [SparQ paper, including the GQA adaptation](https://arxiv.org/html/2312.04985v6): query-channel selection, temperature and token ranking.
- [Proxy model](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF).
- [Evaluation corpus archive](https://huggingface.co/datasets/ggml-org/ci/blob/main/wikitext-2-raw-v1.zip).

Implementation assistance: OpenAI Codex. This is a Saker fork experiment; no upstream llama.cpp submission is included.
