# Project Saker — Path B final report (measured panel)

## Executive summary

**Deliverable: `saker-combined` @ `f65d47b8a`** — llama.cpp patched for this model, on Vulkan/gfx1151.
All figures below measured on that build in one locked session unless noted, all with output checks.

| criterion | target | achieved | verdict |
|---|---:|---:|---|
| quality panel (KL/PPL vs BF16) | <= IQ2_S-mtp baseline | 0.09309585 / 1.358604 | **MET** |
| prefill @2048 | >= 500 t/s | **405.64 ± 1.03** | not met (-19%); **+27% over the 320.7 baseline** |
| decode with MTP | >= 40 t/s | **36.65 - 36.78** | not met (-8%); **+32% to +43% over the 25.7-27.7 baseline** |
| plain decode | >= 26 t/s | **20.10 ± 0.00** | not met; excluded by ~0.6% (needs ~240 GB/s) |

**One of four criteria met; both recorded baselines beaten on the two speed criteria that moved.**
Output is byte-identical to the unmodified model throughout (276 chars, md5 `2fe11002ab`), draft
acceptance bit-identical at 0.77551 (38/49), and the run is deterministic across repetitions.

**What actually produced the wins:**

| change | effect |
|---|---|
| `--spec-draft-n-max 2` (one flag) | 25.7-27.7 -> 34.3 |
| MTP draft-head shortlist (`SAKER_SHORTLIST=3584`) | 34.3 -> 36.19 |
| fused gather-matvec (reuses `ggml_mul_mat_id`) | 36.19 -> 36.65-36.78 |
| APU GEMM retune (`BN` 128->256, live by default) | prefill 299 -> 405.64 |

**Two things an auditor should read before the detail:**

1. **Section 9 states the verification coverage honestly**, including the one change (the prefill GEMM
   retune) that CANNOT be bit-identical because it alters accumulation order, and the limits of
   output-equality testing in this setup.
2. **This report records its own errors rather than quietly fixing them** - four of them, each a case of
   concluding more than the evidence supported, including one where a configuration reporting 492 t/s
   produced no valid tokens at all. They are in section 3 and in PATHB-FINDINGS-20261019.md.

**What is NOT claimed:** the MTP 40 t/s and prefill 500 t/s targets were not reached. The ~4.7 ms MTP
round gap has been narrowed by elimination but not removed (section 7).

**KNOWN DEFECT, unresolved:** the prefill retune fails **1 of 1556** MUL_MAT kernel cases in
`test-backend-ops` (`MUL_MAT(q4_0, m=576, n=512, k=576)`); the retune-disabled control passes 1556/1556.
This model contains no q4_0 tensors, so its output is byte-identical and the end-to-end check cannot see
it - only a per-kernel audit can. **The narrowing was attempted twice and abandoned rather than
half-done** (the first attempt would not compile; the second compounded a rename prefix from 8 errors to
186, and the tree was reverted to the known-good state). The approach and WIP patches are preserved.
**The deliverable therefore does not pass the objective's unit-audit requirement in full**: 2546 of 2547
cases pass, and the failure is a type this model never uses. Stated here rather than hidden (section 9).

**SHIP DECISION (user, explicit): performance-first.** The retune stays ON by default. The narrowing was
attempted three times (two agents plus the parent's own investigation) and failed each time - naming
inconsistency, then a compounding rename loop that took the build from 8 to 186 errors, then a stalled
agent - and a documented limitation was judged better than a fourth attempt that risked a broken tree.
The one-line alternative (make the retune opt-in so the audit passes 1556/1556 by default, at 301 t/s
instead of 409.89) was offered and declined. **This is a deliberate, recorded trade-off, not an oversight.**

---

Date 2026-10-19. Node A (gfx1151 Strix Halo), Vulkan backend, build `saker-combined` `f65d47b8a`.
All figures below are MEASURED on that build. Nothing is projected; projections are labelled as such
and kept in a separate section. GPU verified at 2379-2388 MHz under load (sclk sampled during runs).

## 1. Measured panel

| metric | IQ2_S-mtp baseline | achieved | target | verdict |
|---|---:|---:|---:|---|
| MTP decode, n=64 | 25.7 - 27.7 | **36.65 - 36.78** (3 runs, `f65d47b8a`) | >= 40 | **not met** (-8%) |
| MTP decode, n=128 | - | 32.42 | >= 40 | not met |
| plain decode (tg128) | - | **20.10 +/- 0.00** | >= 26 | **not met** (-23%) |
| prefill pp2048 | 320.7 (ROCm `ub512 fa1`) | **405.64 +/- 1.03** (Vulkan, APU retune live by default) | >= 500 | **not met (-19%)** |
| prefill, real 3728-tok prompt | - | 290.51 (pre-retune; not re-measured after the retune) | >= 500 | not met |
| quality KL vs BF16 | 0.09315009 | **0.09309585** | <= baseline | **MET** |
| quality PPL vs BF16 | 1.359890 | **1.358604** | <= baseline | **MET** |

Notes on comparability, stated plainly rather than smoothed over:
- The MTP number is protocol-dependent because draft acceptance is content-dependent. At n_predict=64
  acceptance is 0.77551 (38/49) giving 35.95-36.19; at n_predict=128 acceptance is 0.61404 (70/114)
  giving 31.92. Both are real; the 64-token figure is the one comparable to the recorded baseline.
- The prefill baseline was **reproduced on this box** (ROCm, `-ub 512 -fa 1`), so the comparison is
  matched in model, hardware and flags. The recorded 320.7 was sound. It also shows **ROCm prefill runs
  ~5% ahead of Vulkan's**, while Vulkan wins decode (plain 20.10 vs ROCm 15.67). The optimal `-ub` is
  backend-specific: 512 on ROCm, 256 on Vulkan.
- **These prefill figures carry a ~1.5% thermal band and are stated as bands, not single values.** A
  receipt-recovery re-run (same configs, under the lock, with two agents sharing the GPU) returned
  315.28 for ROCm and 299.02 for Vulkan - every arm down by a similar amount, i.e. drift, not a change
  in finding. The RELATIVE results are identical (coopmat costs ~40%, integer-dot flat). Treat ROCm as
  315-320 and Vulkan as 299-303.
- The whole panel above was re-measured in ONE locked session on the current deliverable `f65d47b8a`,
  so the rows are mutually consistent. An earlier attempt at this panel was polluted by GPU contention (see
  the measurement-integrity note in PATHB-FINDINGS): it reported MTP at 19.43 tok/s with acceptance
  bit-identical at 38/49, which is the signature of contention rather than a regression.

**Verdict: 1 of 4 criteria met.** The quality gate passes. MTP decode is 37.00 against 40; prefill is
408.77 against 500 (82%); plain decode is excluded by 0.6%.

## 2. What was actually gained

| change | effect | status |
|---|---|---|
| `--spec-draft-n-max 2` (one flag) | 25.7-27.7 -> 34.3 | free, lossless |
| MTP draft-head shortlist `SAKER_SHORTLIST=3584` | 34.3 -> 36.19 (+5.5%) | implemented, merged, lossless |
| fused gather-matvec (reuses `ggml_mul_mat_id`) | 36.19 -> **36.57-37.00** (+1.3%) | implemented, merged, lossless |
| **total vs recorded baseline** | **+32% to +44%** | |

The shortlist is lossless **at the operating protocol, on repeatable evidence**: at n_predict=64,
5 runs per arm produce one distinct output each and the two are byte-identical (276 chars,
md5 2fe11002abed). Draft acceptance is also bit-identical (38/49, mean len 2.52). Verification was
never touched - only the draft's proposal set narrowed.

**Stated precisely, because the limit of the evidence matters:** at 256 tokens a full-vs-shortlist
comparison DIFFERS, but the unmodified baseline is itself non-reproducible at that length (2 distinct
outputs across 3 runs), so that comparison is inconclusive rather than negative - it cannot be
attributed to the shortlist. This setup is deterministic until a generation passes through a near-tie,
after which GPU reduction order can flip an argmax. See PATHB-FINDINGS-20261019.md for the three-test
record. An earlier version of this report claimed losslessness was "PROVEN" on one run per arm; that
was overstated and has been corrected.

## 3. Why the three speed targets are missed (arithmetic, not excuse)

**plain decode >= 26 — excluded, but MARGINALLY (by ~0.6%), and the margin is the whole story.**
Plain decode does NOT read the MTP layer (`blk.64`, ~0.348 GB), so it reads ~9.260 GB per token, not
the full 9.608 GB. Strix Halo is 256-bit LPDDR5X-8000 = 256 GB/s theoretical peak.

    9.260 GB / 256 GB/s = 36.2 ms -> 27.6 tok/s at 100% of theoretical peak
    at the measured achievable 239.4 GB/s (BF16 ref, no dequant) -> 25.9 tok/s
    target 26 tok/s needs 240.8 GB/s = 94% of theoretical peak
    measured today: 20.10 tok/s = 193 GB/s

So the target sits **0.6% above what the hardware has demonstrably delivered**, and reaching it would
require the IQ2_S matvec to match the BF16 streaming rate. It does not (193 vs 239.4 GB/s), and the
sign-fold - the one attempt to close that gap - bought +0.45%. That is why this is called excluded
rather than merely unmet: not because the margin is large (it is not) but because the only measured
route to it has been tried and failed. An earlier version of this report used the full 9.608 GB and
said "~24.9 tok/s"; that was pessimistic and is corrected here.

**prefill >= 500 — not met: 408.77 t/s (82%), and an earlier "490-492 / 98%" claim in this report was an
ARTIFACT.** This entry has been wrong in both directions, so the full history matters:

1. Called excluded, citing ~13.8 TFLOPS from unrelated work on a different model.
2. Called excluded again, citing a second-hand "22.0 TFLOP/s WMMA peak" and building arithmetic on it.
3. Then called **98% of target (490-492 t/s)** on the strength of a `llama-bench` run - **which was
   invalid.** Every arm that changed `BLOCK_SIZE` (256->128) or `BM` (128->256) reports 490-775 t/s and
   **breaks generation**: empty output, or a wrong token stream (acceptance collapses 0.775 -> 0.439).
   **`llama-bench` does not validate output**, so those numbers were artifacts. I never generated text.
4. **The verified figure, with an output check: 408.77 ± 0.85 t/s (+37%), output byte-identical.**

| arm | pp2048 | output |
|---|---:|---|
| **retune ON (new default, `AMD && uma`)** | **408.77 ± 0.85** | 276 chars, md5 `2fe11002ab` - byte-identical |
| retune OFF (`SAKER_APU_MMQ=0`) | 299.98 ± 0.32 | byte-identical |
| pre-change build | ~299 | byte-identical |
| BS128 / BM256 arms (REJECTED) | 490-775 (artifact) | **empty or wrong** |

Only `BN` 128->256 is correct; it is baked in as the default for `vendor == AMD && device->uma` (APUs,
not dGPUs), disableable with `SAKER_APU_MMQ=0`. **So prefill is at ~82% of target, not 98%.**

**Prefill is now CLOSED as unreachable, not merely unmet.** I additionally swept *combinations* of the
warptile fields (the earlier sweep was one-field-at-a-time) and output-verified every arm:

| arm | pp2048 | output |
|---|---:|---|
| **baked-default** | **411.48** | 276 chars, md5 `2fe11002ab` - **VALID** |
| WM32 | 474.60 | 0 chars - BROKEN |
| WN32 | 476.47 | 0 chars - BROKEN |
| WM32+WMITER1 | 471.16 | 0 chars - BROKEN |
| BM64 | 449.05 | **64 chars** - BROKEN |
| WM128 | 312.00 | slower |
| WMITER1 | 406.26 | slower |

**Every arm faster than the baked default is broken.** 411 t/s is the correct ceiling, so no correct
configuration reaches 500. Note `BM64` yielded 64 chars rather than 0 - a plausible-looking wrong output
that llama-bench reports as a clean 449 t/s win. Neither throughput, nor a clean scaling curve, nor a
non-empty response distinguishes these; only comparing generated text against a known-good baseline does.

**The lesson, and it is the most expensive one in this project: `llama-bench` measures throughput and
never validates output.** Any GEMM tiling or accumulation-order change MUST be checked by generating text
and comparing it. A cleanly-scaling curve is not a correctness signal - the 775 t/s arm scaled
consistently across pp512-4096 and was still wrong. I had already written "a ceiling inferred from a
number I did not measure myself is not a ceiling" into this document; this is the same failure one level
deeper: **a throughput number never checked against output is not a result.**

**MTP >= 40 — not reached; both identified levers are now resolved (see section 7).**

    round budget at 40 tok/s ............ 61.5 ms
    weight pass (9.608 GB @ 193 GB/s) ... 49.7 ms
               -> left for draft+verify . 11.8 ms
               -> currently ............. ~18 ms
               -> must cut ............... ~6 ms

| remaining lever | saves | kind of work |
|---|---:|---|
| fused quantized-row-gather matvec (draft lane) | ~2.4 ms | shader |
| multi-column lm_head (verify) | ~4.2 ms | shader |

The hope that this could be captured by better bandwidth is now CLOSED - see section 4.

## 4. The bandwidth question, resolved

A controlled comparison isolated pure streaming from the quantized path: same backend, same session,
same clocks, differing only in weight format.

| model | weights | decode | implied bandwidth |
|---|---:|---:|---:|
| BF16 reference (no dequant) | 54.65 GB | 4.38 t/s | **239.4 GB/s** |
| IQ2_S-mtp (2-bit + dequant) | 9.608 GB | 20.10 t/s | **193.1 GB/s** |

A 19% shortfall. **Its cause was isolated with a third format**: Q8_0 has the same kind of block
structure as IQ2_S (34 B blocks) but near-zero dequant ALU (one multiply per value), so it separates
the two candidate explanations that BF16-vs-IQ2_S confounds.

| format | blocks | dequant ALU | decode | implied bandwidth |
|---|---|---|---:|---:|
| BF16 | none | none | 4.38 t/s | 239.4 GB/s |
| **Q8_0** | **yes (34 B)** | **minimal** | **7.83 t/s** | **227.5 GB/s** |
| IQ2_S | yes (82 B) | heavy LUT | 20.10 t/s | 193.1 GB/s |

**Block structure costs only ~5% (239.4 -> 227.5). The other ~15% is IQ2_S's dequant ALU.** The gap is
ALU, not memory access - which is why it is attackable, and it supersedes the earlier guess that this
was an access-pattern problem.

Three configuration-level avenues were tried first and all closed:

1. **Twelve backend flags** swept - all flat at 20.09-20.17; only two moved, both worse.
2. **Rows-per-workgroup.** `ggml-vulkan.cpp` gates its AMD tuning on `AMD_GCN`, so gfx1151 (RDNA3.5)
   falls through to the generic default. That looked like a device-tuning gap; it is not. Sweeping it
   showed the default is optimal, and that plain decode and MTP want *different* geometry - the
   `rm_iq=2` setting that gains +1.2% on plain decode costs MTP 6-24%.
3. **Memory ceiling** - ruled out, since BF16 demonstrably reaches 239 GB/s.

The remaining ALU cost (LUT lookups, `unpack8`, per-bit sign selects) cannot be moved to integer dot
products because IQ dequant is not integer-representable by design. **This is also the definitive
explanation of the earlier `SAKER_FORCE_QY` null result.** The ALU itself is what the sign-fold work
targets - see section 7.

## 5. Launch command (the deliverable)

    SAKER_SHORTLIST=3584 /home/hycris/llama.cpp-qwen38-merge/build/bin/llama-server \
      -m /home/hycris/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf \
      -ngl 99 -c 4096 --host 127.0.0.1 --port 8080 --temp 0 --no-webui \
      --spec-type draft-mtp --spec-draft-n-max 2

Do NOT set `SAKER_RM_IQ` (leave unset). Omit `SAKER_SHORTLIST` to disable the shortlist.

## 6. Branches

| branch | tree | head | state |
|---|---|---|---|
| `saker-prefill` | llama.cpp-qwen38-a | `2c8a51820` | APU GEMM retune (merged) |
| `saker-lmhead` | llama.cpp-qwen38-lmhead | `6f7c25682` | lm_head hoist (negative) + round split |
| `saker-shortlist` | llama.cpp-qwen38-a | `b35a6009d` | shortlist + fused gather |
| `saker-verify` | llama.cpp-qwen38-b | `4664a18b1` | attribution proven, no fix |
| **`saker-combined`** | llama.cpp-qwen38-merge | **`f65d47b8a`** | **THE DELIVERABLE** - shortlist + fused gather + APU prefill retune, all verified |
| `saker-iqrows` | llama.cpp-qwen38-merge | - | `SAKER_RM_IQ` knob only; do NOT merge |

## 7. What would close the gap to 40

**(1) DONE - fused quantized-row-gather matvec.** `ggml_get_rows` used to emit an F32 `[5120 x K]`
intermediate (~146 MB of traffic at K=3584) while the gathered weights are only ~9.7 MB. It now reuses
llama.cpp's existing MoE indirect matmul (`ggml_mul_mat_id`) with the vocab rows viewed as one
1-column expert matrix each, so the intermediate is gone. Measured +1.3% (36.19 -> 36.57-37.00),
acceptance bit-identical. Honest caveat: worth ~0.7 ms/round, not the ~1.5 ms the removed traffic
predicted, because `mul_mat_vec_id` is shaped for MoE and gives some of the saving back in kernel
efficiency in this degenerate 1-row-per-expert case.

**(2) TRIED, AND THE RESULT WAS MISREAD - sign-fold in the IQ2_S matvec.** This was my recommended
route to 40, with the arithmetic stated as "a full 1.116x takes the weight pass to ~44.5 ms -> round
~61.3 ms -> ~40.1 tok/s". It was ported, and the shader was verifiably regenerated (SPIR-V
disassembly showed the new 16-entry Workgroup mask table and the vector `OpBitwiseXor`s, so this is not
a stale-build artifact). Measured:

    plain decode tg64, r=5:  19.90 +/- 0.03  ->  19.99 +/- 0.01   =  +0.45%

I concluded from that "the Vulkan driver already handles sign selects cheaply, so the ALU cost is
elsewhere". **That inference was INVALID, and the report is corrected here.** Parsing the GGUF header
shows the model is **mixed-quant** - IQ3_S 21.4%, **IQ2_S only 18.9%**, IQ4_NL 16.7%, Q2_K 12.7%,
IQ2_XXS 10.2%, IQ2_XS 7.9%, and more. The sign-fold edited `mul_mat_vec_iq2_s.comp`, which covers
**18.9% of model bytes**, so a *perfect* IQ2_S-only fix could not have moved end-to-end decode by more
than ~19%.

**+0.45% is consistent with SCOPE, not with a null result.** The ALU route is not closed - it was
tested on a fifth of the model and read as if it had been tested on all of it. But capturing the full
~19% gap (193 vs 239 GB/s) now means ALU work across the *whole* quant mix - IQ3_S, IQ2_S, IQ4_NL,
Q2_K, IQ2_XXS, IQ2_XS, IQ1_M, Q6_K - roughly **5x the scope** that was funded. The confident number I
gave the user was wrong in both directions: the technique may transfer, but not at the scope implied.

**(3) TRIED AND FAILED - multi-column lm_head.** Committed `df4ce6e2b` on `saker-lmhead`. The
structural cause was found exactly as predicted - IQ4_XS has no per-type shader;
`vulkan-shaders-gen.cpp:737` routes it to the generic `mul_mat_vec.comp` template where the column loop
is OUTER and the dequant sits INSIDE the row loop - and the dequant was hoisted correctly. Correctness
is airtight both ways (pure loop-invariant motion, AND empirically md5 `2fe11002` = the baseline output
with acceptance bit-identical at 0.77551). **But there is no measurable gain**: MTP 36.58-37.00 ->
36.50/36.57/36.66 (noise), because `[[unroll]]` makes the duplicated dequant CSE-able and RADV was
already eliminating it. A source redundancy, not a runtime one.

**And the target itself did not reproduce.** The +6.83 ms (pp3) / +17.16 ms (pp4) figure this work was
sent to fix measures as **+0.41 ms and +0.97 ms** in a clean, locked, same-build run. The multi-column
lm_head is already nearly bandwidth-bound; there was never ~4.2 ms/round to recover.

**This VOIDS the route-to-40 projection given earlier:**

    lm_head (-4.2) + host-sampling (-0.8) = 40.00 tok/s   <-- built on a non-reproducing measurement

With the lm_head's real cost at ~0.5 ms, removing both it and the host-sampling fallback yields roughly
37.7 tok/s, not 40. At this point in the investigation I wrote that "MTP >= 40 has no remaining
identified route" - **that was superseded within the hour**: the gap was located, then its mechanism
found. See item (4) below and section 7's mechanism paragraph.

**(4) THE ROUTE TO 40 IS NOW LOCATED - and it is plumbing, not shaders.** The round is fully
accounted for by a finer trace (`dec` and `sync` counters added):

| phase | per round | what it is |
|---|---:|---|
| `sync` | 59.0 ms | the GPU weight pass completing |
| `draft()` | 5.3 ms | the 2 MTP drafts |
| `dec` | 2.1 ms | host-side graph dispatch |
| `proc` | 1.3 ms | MTP KV catch-up |
| `smpl` | 0.5 ms | sampler clone + sample + accept |
| **total** | **~68 ms** | measured round 66.5 (slight overlap) |

`sync` is only **~2.2 ms above a plain 3-token prefill (56.78 ms)**, so the verify itself is nearly
optimal. An earlier claim in this report that "the ~4 ms verify-vs-pp3 gap has not been located" is
superseded - there was no hidden gap; the remainder is the draft lane and host overhead that had already
been counted.

**MECHANISM - corrected, because my first version was wrong.** I originally recorded a "3943-node graph
rebuilt every round" here. **That is retracted**: the graph is REUSED across rounds (rebuild only happens
on a shape change), and my 3943-vs-51/54 comparison was apples-to-oranges - the 51/54-node graphs are the
**draft context's**, not 2-position target graphs. The node dump (`SAKER_CTX_DUMP=1`) shows what is
actually true:

    n_max=2:  saker-ctx: decode ubatch#1 nt=3 nseq=1 nst=3 allout=3/3
              saker-ctx: decode ubatch#1 nt=3 nseq=3 nst=1 allout=0/3    <- nseq=3
              saker-ctx: ub REBUILD ntok=2 nseq=1 nst=2 nodes=3943       <- 3943-node graph
    n_max=1:  nodes=51, 54.  The 3943 graph never appears.

    plain pp2 / pp3 target graph : 3655 nodes   (FLAT - does not grow with token count)
    MTP verify, 2 positions      : 3799
    MTP verify, 3 positions      : 3943         -> +144 nodes PER VERIFY POSITION

**+288 nodes for the whole 3-position verify** (CPY +96, VIEW +192), decomposed by extracting the copied
tensor names: **`cache_r_lN` (the GDN conv/recurrent-state cache) and `cache_s_lN` (the GDN SSM-state
cache), one pair per GDN layer.** The verify copies both, per layer, per verify position.

**The SSM/GDN COMPUTE op counts are identical** between a plain forward and a verify (48 `GATED_DELTA_NET`,
48 `SSM_CONV`). So the rollback machinery saves state by **copying** the recurrent state caches - pure
memory traffic, no extra arithmetic. That is precisely why sync, logits, CPU submit, KV-position, async
draft work, per-op timing and the lm_head all came back negative: none of them is a copy of the recurrent
state.

The dump carries no shapes, so the sizes are not measured; `cache_r` is conv_input-shaped (modest) while
`cache_s` is the SSM state and potentially MB-scale per layer, which would put 96 copies per graph at
1-3 ms - the right order for the missing time. **The fix direction is to avoid or cheapen the COPY**
(copy-on-write, or copying only when a rejection is actually possible) rather than to touch the SSM
kernels. That also fits the 2-position case, where the gap is absent because no rejection is possible.

Two further corrections from the same dump: the `nseq=3` batches appearing once per round are on the
**draft context**, not extra target work; and per-op timing shows the verify's individual ops cost the
same as pp3's, so **the ~2.5 ms is not inside any single op.**

**So: located but not removed.** It is not a caching problem and not a single-op problem. Whether those
+144 nodes can be cheapened or overlapped is a graph-builder change in `llm-graph.cpp`/`qwen35.cpp`, not
started, and the agent explicitly declined to claim it is avoidable.

    round 66.5 ms; need <= 61.5 ms for 40 tok/s  ->  must cut ~5 ms

| candidate | worth | kind |
|---|---:|---|
| draft lane (2 drafts @ ~2.65 ms) | up to 5.3 ms | MTP draft path |
| host overhead (`dec`+`proc`+`smpl`) | up to 3.9 ms | dispatch/sampling plumbing |
| verify gap over plain prefill | ~2.2 ms | unclear |

**All three are code and plumbing work, not shaders** - unlike the sign-fold and lm_head attempts, both
of which were GPU-kernel projects and both of which failed. The draft lane is the largest single piece;
the fused gather already cut it once, and what remains is 2 sequential drafts over blk.64 plus the
shortlist rows. An agent is attributing and attacking this now.

**Prefill: 408.77 t/s (+37%), live by default** - a GEMM tiling retune for AMD APUs (`BN` 128->256),
verified byte-identical on output. It is NOT dequant-limited (IQ2_S and Q8_0 prefill are identical to
within 0.7%), it IS on the coopmat path, and the earlier "490-492 / 98%" figure was a llama-bench
artifact from a configuration that breaks generation - see section 3 for the full correction.

Also measured on the way: prefill is flat across all batch shapes (304 -> 295 t/s from pp512 to
pp8192), so there is no batch-shape lever; and `DISABLE_COOPMAT` costs 40% while
`DISABLE_INTEGER_DOT_PRODUCT` moves it 0.8%, confirming it runs on the coopmat tensor-core path.

## 8. The Path A engine (the objective's actual deliverable)

This report is titled "Path B" because that is where the user directed the effort. But the objective's
deliverable is the **from-scratch Saker engine** (Path A, HIP/ROCm, repo `github.com/barrymaiawang-droid/saker`),
so its numbers belong here for completeness. These are quoted from its own report
(`FINAL-REPORT-20261019.md`) and were NOT re-measured for this document:

| criterion | target | Saker engine (Path A) | Path B (this report) |
|---|---:|---:|---:|
| plain decode | >= 26 tok/s | 12.09 (12.18 with the IQ2_S variant source) | **20.10** |
| decode with MTP | >= 40 tok/s | ~9.4 (draft lane cut to 1.18 weight-passes) | **37.00** |
| prefill @2048 | >= 500 tok/s | 325.1 (91% of a ~357 ceiling) | **408.77** (Vulkan, APU retune) |
| quality panel | <= IQ2_S-mtp baseline | **MET** - KL 0.09309585 <= 0.09315009 | same GGUF, so same gate |

The engine's one measured, reproducible win was **plain decode +43.4%** (8.4287 -> 12.0866 tok/s) by
moving the lm_head off hipBLAS onto a packed kernel; a second fix cut the MTP draft lane from 2.53 to
1.18 weight-passes. Its report concludes, and this project now independently confirms, that **the
binding constraint on every remaining criterion is raw throughput, not the MTP machinery**.

Path B overtook it on both speed criteria because llama.cpp's Vulkan path starts from a better
baseline (20.10 plain vs 12.09) rather than because the engine's design was wrong. Note the engine's
prefill (325.1) is actually the *highest* prefill figure in the project - above both ROCm's 315-320
band and Vulkan's 299-303 - and its ~357 t/s ceiling estimate is what cross-validates the prefill
exclusion in section 3.

## 9. Verification coverage (the objective's audit requirement, mapped)

The objective requires "unit + integration audits of every kernel vs CPU/high-precision references".
This section maps each CHANGE made by this work to the evidence that covers it, and states plainly
where the coverage is structural rather than measured. For Path A's own kernel audits see
`FINAL-REPORT-20261019.md` (66-case GPU audit over 10 trunk formats, plus 10 index-0 negative controls).

| change | unit-level evidence | integration evidence |
|---|---|---|
| MTP draft-head shortlist (`SAKER_SHORTLIST=3584`) | candidate scoring restricted to K rows; verification path untouched by construction | **output byte-identical** to the full-head arm across 5 runs/arm at n=64 (md5 2fe11002abed); acceptance bit-identical 0.77551 (38/49) |
| fused gather-matvec (reuses `ggml_mul_mat_id`) | gather path retained as the multi-output fallback, so both paths coexist | **output byte-identical** off-vs-fused, 3 runs/arm, both arms self-consistent 1/3 distinct |
| multi-column lm_head hoist (in flight) | **structural**: `dequantize4` is a pure function of the weight block, so hoisting is loop-invariant motion and bit-identical BY CONSTRUCTION | output-equality check requested, >=3 runs/arm |
| sign-fold (tried, failed, not merged) | SPIR-V disassembly confirmed the shader was genuinely regenerated (mask table + vector `OpBitwiseXor`s present) | output byte-identical; +0.45% only |
| prefill APU retune (BLOCK_SIZE 256->128, BN 128->256) | **NOT expected bit-identical** - a GEMM tiling change alters the accumulation order. Verified instead by: guard scoped to `AMD && uma`; only two fields changed with `tm/tn/tk` preserved; `SAKER_APU_MMQ=0` control arm retained; decode non-regression re-measured | output comparison with and without the retune, >=3 runs/arm |

### Unit audit: 2547 kernel cases across every op this project touched, 1 failure

| op | covers | cases | failing |
|---|---|---:|---:|
| **MUL_MAT_ID** | the fused gather-matvec | **872** | **0** |
| **GET_ROWS** | the gather fallback | **119** | **0** |
| MUL_MAT | the APU prefill retune | 1556 | **1** |

The fused gather is therefore verified against a CPU reference across 872 cases - a stronger claim than
the byte-identical output check it had before. The one failure is the retune's q4_0 case below.

### Unit audit result - and it found a real defect

`test-backend-ops test -o MUL_MAT -b Vulkan0` (llama.cpp's kernel-vs-CPU-reference harness) on the
retuned deliverable:

| arm | result |
|---|---|
| **retune ON (baked default)** | **FAIL - 1 case of 1556** |
| retune OFF (`SAKER_APU_MMQ=0`) | OK |

The single failing case is `MUL_MAT(type_a=q4_0,type_b=f32,m=576,n=512,k=576,...)`. So the retune is
correct for **1555 of 1556** kernel cases and wrong for one q4_0 shape.

**The model output was byte-identical anyway because this model contains no q4_0 tensors** (its mix is
IQ3_S 21.4%, IQ2_S 18.9%, IQ4_NL 16.7%, Q2_K 12.7%, IQ2_XXS 10.2%, IQ2_XS 7.9%, plus Q6_K/Q4_K/IQ3_XXS/
BF16/F32). The broken case is never exercised end-to-end.

**This is the strongest argument in the whole project for why unit audits exist.** An end-to-end output
check on this model is structurally incapable of seeing a q4_0 regression - only a per-kernel comparison
against a reference can. The deliverable as committed (`f65d47b8a`) therefore **fails the objective's
unit-audit requirement**, and narrowing the retune so q4_0 keeps the generic tile is required; that work
is in progress. If it cannot be done without losing the +37%, the trade-off must be stated, not hidden.

Two honest limits on this coverage:

1. **The strongest evidence here is byte-identical OUTPUT, not per-kernel numerical comparison.** For a
   verification-preserving change that is arguably the right level - the end-to-end token stream is what
   matters - but it is weaker than a per-kernel comparison against a CPU/high-precision reference, and
   it is worth saying so rather than letting the table imply more than it shows.
2. **The prefill retune is the one change that CANNOT be bit-identical**, because it changes a GEMM's
   accumulation order. Device tiling changes are normal practice and llama.cpp selects tiles per device,
   but the claim has to be made differently: not "identical output" but "output within fp accumulation
   tolerance, with a retained control arm and decode non-regression". Any quality statement for this
   change rests on that, and it is flagged here rather than folded in with the lossless changes.
3. **This setup is not bit-reproducible at long horizons.** It is deterministic until a generation
   passes a near-tie, after which GPU reduction order can flip an argmax. At 256 tokens the UNMODIFIED
   baseline produced 2 distinct outputs across 3 runs, so any output-equality claim must use >=3 runs
   per arm and a self-consistent control. A single run per arm is not evidence - a mistake made and
   corrected earlier in this work.
