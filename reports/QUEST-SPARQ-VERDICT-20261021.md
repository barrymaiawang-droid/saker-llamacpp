# Quest / SparQ verdict — 2026-10-21

Branch `codex/quest-sparq-attention` (`b8936f00`), built from a pristine `9d57ce456` +
`saker-combined` + `saker-sparse-attention` + `saker-quest-sparq`, Vulkan on gfx1151.

## Verdict: not shippable. Slower at every tested depth, in both decode modes.

### Speed (predicted_per_second, n=3, real prompt tokens, `cache_prompt:false`)

| arm | 4096 | 8192 |
|---|---|---|
| AR dense | **19.411** | **18.933** |
| AR sparse (2048) | 17.782 | 17.657 |
| | **-8.4%** | **-6.7%** |
| MTP dense | **32.003** | **32.601** |
| MTP sparse (2048) | 27.111 | 24.911 |
| | **-15.3%** | **-23.6%** |

Sparse provably engaged: 775 `saker-sparse: active` markers in the sparse arm, 0 in dense.
Prefill unaffected (318/308 dense vs 326/310 sparse) — consistent with sparse being decode-only.

**The penalty grows with depth.** That is the decisive result: the design premise is that longer
contexts amortise selection cost, but the selection work scales with context while the KV saving is
capped by a fixed budget. At 2048 budget the saving is fixed, the overhead is not.

Draft acceptance gave no quality signal: dense 0.636/0.700 vs sparse 0.659/0.584.

### Quality

Both corpora are out-of-distribution for this model (dense PPL 8816 on wikitext, 22157 on project
source; independent `llama-perplexity` = 1403 on wikitext), so absolute PPL cannot rank modes and
PPL is non-monotonic in budget. The self-consistent metric is top-1 agreement with dense:

| budget | representative | quest | sparq |
|---|---|---|---|
| 512 | 0.256 | 0.238 | 0.273 |
| 2048 | 0.430 | 0.451 | 0.395 |

Direction is correct (more budget -> closer to dense) but at half the context selected, under half
the argmax tokens match. The branch's own 0.98-0.99 was measured on a confident 0.8B model; this
model's near-uniform entropy inflates divergence, so treat this as caution, not calibration.

## Three bugs found and fixed on the way

1. **`ggml_top_k` degenerates on ±INF input on Vulkan.** `saker_sparse_fill_row` used `±INFINITY`
   sentinels for forced pages (recent/sink protection, Quest partial pages). Vulkan returned the
   first k indices instead of ranking, silently dropping protection pages — i.e. losing the most
   recent context. Their `test-backend-ops` TOP_K passes 427/427 only because it never feeds
   infinities. Fixed with a finite `saker_sparse_force = 1e30f`.

2. **`test-saker-selectors` required an unreachable tolerance.** The pass0-vs-pass1 canary demanded
   absolute `1e-6`; measured 3.05e-05 on a value of -0.052 (0.06%, 1-2 F16 ulp). Masks were verified
   correct on both backends (7 aligned `-inf` entries), so this was not a leak. Changed to the test's
   own existing convention `0.002 * (1 + |expected|)`.

3. **MTP segfault — dangling memory context.** `llm_graph_input_attn_kv::can_reuse_saker` read
   `this->mctx`, which the hybrid reuse path never refreshes, so it pointed at a destroyed
   `llama_kv_cache_context` (SIGSEGV inside `llama_kv_cache::has_saker_quest`). Every sibling check
   takes the fresh context: `can_reuse_kq_mask(self_kq_mask, mctx->get_attn(), ...)`. Changed the
   signature to take it and both call sites to pass it. MTP then ran; that is what made the MTP
   measurement possible at all.

After fixes: `test-saker-selectors` PASS 24 cases on **CPU and Vulkan0**, ctest 2/2.

## Artifacts

- Speed: `talon-out/saker/sparq-bench-20261021/bench.jsonl` (AR), `sparq-mtp-20261021/bench.jsonl` (MTP)
- PPL: `sparq-ppl-20261021/` (wikitext), `sparq-ppl-id-20261021/` (project source)
- Fixes: `/home/hycris/saker-sparq-fix-vulkan-inf.patch`, `~/sparq-base` working tree
