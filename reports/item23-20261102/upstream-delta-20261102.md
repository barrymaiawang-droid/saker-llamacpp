# Item 23: what the pinned llama.cpp base is missing, and what that means for the pelican

**Date:** 2026-11-02 (offline review, no GPU or model access). **Scope:** the installed engine is
`ggml-org/llama.cpp` at `9d57ce456c94d241dde672b2db9cf18879766568` (2026-08-14, build b10438) plus
the Saker combined patch. Upstream `master` was `785322d17f2d51e5185cf996a1f4e46d87640841`
(2026-11-01) when this was written, 1,595 commits ahead of the pin. Everything below was read from
the upstream Git history and the handoff evidence archives; nothing was executed on node B.

## 1. What the production engine actually is

Reading the whole of `patches/saker-combined.patch` hunk by hunk: every Saker change outside the
Vulkan tile retune is gated on an environment variable that the item-23 launcher does not set or
explicitly removes (`SAKER_SHORTLIST`, `SAKER_SL_DEBUG`, `SAKER_SPEC_TRACE`, `SAKER_LMMQ_WT`,
`SAKER_L_WT`, `SAKER_LMMQ_WGD`, `SAKER_VK_TRACE`, `SAKER_PF_TRACE`, `SAKER_FORCE_QY`,
`SAKER_ROUND_TRACE`, `SAKER_ALLLOGITS`). With the shortlist off, `saker_mtp_shortlist_get()` returns
null, the MTP head uses the upstream `build_lora_mm`, `llama_context::decode/encode` take the
upstream `n_vocab`-wide readback branch, and `common/speculative.cpp` keeps upstream backend
sampling. The server-context and llama-bench hunks only add timers.

So the installed engine is **upstream llama.cpp 9d57ce456 plus the repaired BN256/WN128 Vulkan
tile**, nothing else. The retune-disabled arm in the audit (`SAKER_APU_MMQ=0`) is byte-for-byte
upstream behaviour, and it scores the same against the CPU reference as the retuned arm
(mean KL 0.002832 vs 0.002961). Any residual engine problem is therefore an *upstream* problem or a
model problem, not a Saker-introduced one.

## 2. The CPU comparison bounds the Vulkan error tightly

The published arms only recorded mean KL in `complete.json`; the full distribution is in each
`run.log` inside `q38gsq-engine-audit.zip`. Corrected Vulkan vs pure CPU over 1,279 positions:

| statistic | corrected Vulkan | retune disabled | Vulkan f32 diagnostic |
|---|---:|---:|---:|
| mean KL | 0.002961 | 0.002832 | 0.001687 |
| median KL | 0.000118 | | |
| 99.0% KL | 0.039686 | | |
| 99.9% KL | 0.110321 | | |
| max KL | 0.142817 | 0.148477 | 0.170650 |
| same top token | 98.436% | 98.593% | 99.062% |
| max top-1 probability delta | 12.8 points | | |

A mean of 0.003 could in principle hide one gross error every few hundred tokens; the maximum of
0.14 rules that out for this block. The shape (tiny median, heavy but bounded tail, 1.5% top-token
flips at near-ties) is ordinary f16-accumulation noise, and the f32 arm narrows it as expected.
**Vulkan arithmetic is faithful to the CPU implementation of the same graph.** No Vulkan commit in
section 5 can explain the remaining visual problem, because the CPU reference already does not.

## 3. What the audits cannot see

Every fidelity number compares two implementations of the **same compute graph** on the same
quantized weights. A defect in the graph definition itself, or a loss caused by the IQ3_S
quantization, is reproduced identically on CPU and Vulkan and scores KL 0. The only way to see such
a problem is a reference outside this graph: the upstream model code at a later commit, or a
higher-precision checkpoint. This review used the first; the second still needs node B.

## 4. Graph-level upstream changes for `qwen35` since the pin

| commit | PR | what it changes | relevance |
|---|---|---|---|
| `5fdfa6282` 2026-09-06 | #28068 | GDN q/k normalization: `ggml_l2_norm` is `x / max(sqrt(sum x²), eps)`; the reference (flash-linear-attention, and transformers after huggingface/transformers#40842) is `x * rsqrt(sum x² + eps)`. Replaced by `rms_norm(x, eps/n) * 1/sqrt(n)`. | **The only upstream-confirmed deviation of the pinned qwen35 graph from the reference model.** Affects every linear-attention layer. Backported as `patches/saker-gdn-l2norm-eps.patch`. Expected magnitude is small: the relative change is `eps / (2·Σx²)`, so it only matters where a q/k conv vector is nearly zero. It is a faithfulness fix, not a claimed fix for the pelican. |
| `839a572a4` 2026-10-21 | #30823 | reshape linear-attention output to 2-D instead of 3-D | layout only, no numerics |
| `f0c41e016`, `448147d42`, `1a3011cc0` 2026-10-06/07 | #30017, #30097, #30020 | nextn row cropping helpers, shared nextn flags, re-reserve the scheduler when nextn extraction flags change | plumbing/allocation; the pinned MTP graph crops the same rows |
| `866322481` 2026-08-28 | #27877 | `auto_fgdn`/`auto_flid` default false (fused GDN stays on) | selection policy; the installed build already runs the fused GDN op |

## 5. Speculative decoding and recurrent-state changes since the pin

| commit | PR | what it changes | does it bite the installed MTP2 profile? |
|---|---|---|---|
| `4453b535f` 2026-09-30 | #29019 | layer-input / unmasked NextN embeddings were handed to the draft in ubatch order, not original batch order | only when the batch allocator reorders tokens, i.e. several sequences; item 23 runs `-np 1` with `split_simple` ordering. Not a factor here; it would matter before raising slot count. |
| `d280808f5` 2026-09-30 | #29638 | stop accepting draft tokens after EOG | affects tokens *after* end of generation, which the server never emits; not page quality |
| `1fb7ef3e3` 2026-10-02 | #27694 | optional probabilistic (rejection-sampling) verification | opt-in feature. The pinned verifier samples from the target at each position and accepts the draft token only if it equals the sample, which is already exact in distribution. |
| `7df8b876b` 2026-10-29 | #31350 | server restores the spec checkpoint when the *draft* (not the rollback) exceeds `n_rs_seq` | with `--spec-draft-n-max 2`, `n_rs_seq = n_max = 2`, every target batch is ≤ 3 = `n_rs_seq + 1` tokens and every rollback ≤ 2, so the pinned condition and the fixed condition agree. Consistent with handoff finding 6 (192/192 top tokens agree under mixed rejection). |
| `ab3768cdc` 2026-10-30 | #31368 | recurrent memory refuses rollbacks past the snapshots the last ubatch actually wrote | same analysis: a 3-token batch writes snapshots 0..2, so no refused rollback would occur. Matters only if `n_max` is raised without raising `n_rs_seq`. |
| `fa3c2fab3` 2026-10-07 | #29923 | `test-recurrent-state-rollback` keeps the anchor token | upstream fixed the **test**, not the restore code. This independently confirms handoff finding 5: the test's dirty-restore mismatch was a test artefact. Do not patch restore code for it. |
| `08618ff8e`, `869034b4b` | #27530, #29799 | cleanup after failed restores; an invalid assert | error paths, not steady-state numerics |

Issue status (read 2026-11-02): ggml-org/llama.cpp#29493 is still open and `stale`, no fix PR;
#26695 was closed by its author without a fixing commit after the maintainer declined a slower
`seq_rm` fix, noting the MTP path never removes the anchor. Neither names a defect that the installed
profile exercises.

## 6. Vulkan backend changes since the pin

Correctness: `526247161` #28956 (mul_mat on a strided view of a larger buffer, e.g. KV cache slices),
`806eee984` #29591 (stale `prealloc_y` across FA and soft_max), `6c59c4007` #29988 (FA shared-memory
OOB write), `78ec4c378` #27413 (FA Q quantization in fp32), `2864937a7` #30283 (A prefetch clamp in
the new int8 coopmat1 matmul). Section 2 shows none of these engages measurably on this workload,
otherwise the CPU comparison would have caught it; they are still reasons to rebase.

Performance for exactly this hardware class: `70c4e1582` #27952 (int8 coopmat1 matmul for AMD RDNA3
and RDNA4), `bbd488c42` #28822 (IQ3_S MMQ kernels, the model's weight type), `b1ff4ca23` #28415
(IQ4_XS MMQ), `5c200e0c8` #29476 (GDN kernel tuning), `733f57a99` #27703 (strided f16 KV copy on
AMD). After a rebase the Saker BN256 retune must be re-measured against these; it may be redundant.

## 7. Server and template handling since the pin

`a71c00f0b` #30557 (qwen3-coder parser stops reasoning/content at template separators),
`e750b887a` #28174 (`preserve_reasoning` default on), `1b4554eff` #31271 (expose the template's
reasoning-effort levels), `9a3b91c68` #29486 (`completion_tokens` undercount with `n > 1`). None
changes which tokens the model produces for the pelican request.

## 8. Conclusion

1. The confirmed Vulkan tile defect is repaired, and the per-position distribution shows no second
   arithmetic defect on the audited block.
2. The installed engine equals upstream 9d57ce456 plus that repair. Its one upstream-confirmed
   graph deviation for this architecture (#28068) is now available as an incremental patch; its
   numerical effect is expected to be far below the f16 noise floor measured in section 2.
3. The remaining pelican failures (spoke length versus wheel radius, scenery axis, frame-rate
   dependent motion, proportions) come from one temperature-1 sample at the default (medium) effort,
   with about 250 tokens of visible reasoning before a 14 KB page. That is weak evidence of an engine
   defect. "Pelican riding a bicycle" is a known hard generative benchmark on which much larger
   models produce awkward geometry; a 27B model at 3.44 bits per weight is not expected to pass a
   strict geometric QA on one draw.
4. The decisive experiment has not been run: generate the unchanged request on the **CPU reference
   implementation** and QA it. If the CPU page fails the same checks, the Vulkan engine is
   exonerated for this task and what remains is model/quantization capability. The reproduction
   harness now has a `--backend cpu` arm for exactly this (section 9 of the handoff).
