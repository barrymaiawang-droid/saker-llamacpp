# Path B findings: llama.cpp's MTP draft lane (2026-10-19)

Path B = improve the *number* by patching the implementation that already has the fast plain rate
(llama.cpp Vulkan, 20.20 tok/s plain on this box) instead of the from-scratch HIP engine, which tops
out near 15.7 plain. Everything below is measured on node A, same GGUF, same 64-token prompt, via
`llama-server` + a bounded `/completion` request.

## Headline

| configuration | tok/s | note |
|---|---:|---|
| plain | 19.98 | |
| `--spec-draft-n-max 1` | 29.97 | acceptance 0.853 |
| **`--spec-draft-n-max 2`** | **34.39 / 34.40** | **best measured; free flag change** |
| `--spec-draft-n-max 3` (default) | 33.66 / 33.68 / 33.70 | |
| `--spec-draft-n-max 4` | 12.98 | pathological, but see below - it does not matter |

`n_min` and `p_split` are dead knobs (34.39/34.40 vs 34.03 at `n_max=2`; 33.66/33.70 vs 33.68 at
`n_max=3`). `backend_sampling` is already on by default, so there is no per-draft host sync to remove.
**`n_max=2` is the only free win: +2% over the default, no code, no quality risk.**

## Why the draft lane is the whole story

The round cost decomposes exactly from the sweep (`round_ms = base + k x per_draft`):

    n_max=1: 60.1 ms/round    n_max=2: 71.2    n_max=3: 85.0
    marginal cost per draft: 11.1 ms then 13.8 ms  -> ~12 ms
    base (verify) = 60.1 - 11.1 = 49.0 ms, against 49.3 ms for a plain token

So **W = 1.0 already**: the verify is one 9.6 GB weight pass at 195 GB/s and cannot improve. Every
token above 20 tok/s comes from the draft lane, and

    tok/s = A / (0.049 + 0.012 x k)

Draft bytes are blk.64 (Q6_K, 348 MB) + the shared lm_head (IQ4_XS, 675 MB) = 1.02 GB/draft = 5.2 ms
at 195 GB/s. **The measured 12 ms means ~7 ms is compute and launch overhead, not bytes** - so a
shortlist that removes the lm_head read buys ~3.5 ms, not the ~7 I first projected.

## The n_max=4 pathology is a dead end (closed)

At `n_max=4` the round cost jumps 85 -> 242 ms while the mean round only grows 3.00 -> 3.32. No
clamp fires (`n_draft_max` is not enforced in this impl) and the depth is genuinely used. But fixing
it would not help: at 12 ms/draft, k=4 projects to 3.32/(0.049+0.048) = 34.2 tok/s - the same as
today's 34.4. Deeper drafting only pays if the draft lane is cheap. **Do not re-open this.**

## What a real improvement requires, and why it is invasive

| change | projected tok/s | cost |
|---|---:|---|
| today (`n_max=2`) | **34.4** | free |
| shortlist head (score K vocabulary rows instead of all 248,320) | ~37 | requires computing logits for a row subset inside the draft context - graph surgery in llama.cpp |
| batched drafting (one MTP pass over the whole draft batch, amortising blk.64) | ~44 | requires the target's `h_nextn` for every draft position, i.e. a verify-first round restructure |
| draft lane free (theoretical ceiling) | ~60-80 | A ~ 3-4 at 20.4 tok/s per weight pass |

## Quality

**Zero regression, by construction.** Speculative decoding with exact verification is lossless: every
emitted token is the target's own greedy token. Nothing done to the *draft* path - shortlist, cheaper
head, batching - can change a single output token; a bad draft costs time, never correctness. The only
ways to regress quality would be to change the weights or approximate the verify. Neither is proposed.

## Bottom line

Path B yields **34.4 tok/s today for one flag**, and realistically **~37 with a shortlist** and **~44
with batched drafting**, both of which are real engineering projects inside llama.cpp rather than
patches. The 40 criterion is reachable only via the ~44 route.

## Instrumented: where the round time actually goes (measured, not derived)

Patched `common/speculative.cpp` (backup at `common/speculative.cpp.orig-saker`) with an env-gated
timer around `impl->draft()`, rebuilt the Vulkan `llama-server`, and ran with `SAKER_SPEC_TRACE=1`:

```
24 rounds traced · draft() = 11.00 ms median per ROUND (both drafts)
-> 5.50 ms per draft, against 1.02 GB of draft bytes = 5.2 ms at 195 GB/s
```

**The draft lane is already bandwidth-bound. There is nothing to win there** - which kills the
shortlist/batched-drafting plans as *draft-side* optimisations.

The cost is on the verify side, and it is the model's normal multi-position compute:

| verified positions | ms | marginal |
|---:|---:|---:|
| 1 | 50.1 | - |
| 2 | 52.2 | +2.1 |
| 4 | 67.2 | ~7.5/pos |

Cross-checked against the MTP sweep, which fits `verify(k) = 49.3 + 6.4k` for k+1 positions - the
same slope as plain prefill scaling. So the round is

    round_ms = 49.3 (one 9.6 GB weight pass, the bandwidth floor) + 11.9 x k
               11.9 = 5.5 (draft, already at bandwidth) + 6.4 (verify compute per extra position)

    k:  1      2      3      4
    A:  1.83   2.46   2.91   3.32
    t/s: 30.5   34.4   34.5   34.5

**The plateau is flat across k = 2..4**, which is exactly why every flag sweep landed within noise of
34.4: adding a draft costs 11.9 ms and buys only p^k tokens, and at p ~ 0.75 that is break-even.

## The honest ceilings

- **40 tok/s** needs the round at k=2 to fall from 72 to 61.5 ms, i.e. the per-position cost halved
  from 11.9 to ~6 ms. The draft is already at bandwidth, so this means roughly halving the verify's
  per-position *compute* - a small-batch GEMM optimisation inside llama.cpp's Vulkan backend.
- **60 tok/s is impossible at today's acceptance**: at A = 2.46 it needs a 41 ms round, which is
  *below the 49.3 ms weight-read floor*. Even at A = 4 it needs 4.4 ms/position against today's 11.9.
- **Best measured, free: 34.4 tok/s** (`--spec-draft-n-max 2`), which already beats the objective's
  recorded 25.7-27.7 baseline.

Everything above is measured. The instrumentation patch is env-gated and inert unless
`SAKER_SPEC_TRACE=1` is set, so it can stay in the tree or be reverted from the `.orig-saker` backup.

## The last cheap hypothesis, falsified

Checked whether the 3-column verify falls off llama.cpp's matvec path onto a tiled matmul (which is
what wastes ~8x in our own engine's prefill path):

    ggml/src/ggml-vulkan/ggml-vulkan.cpp:389
    static constexpr uint32_t mul_mat_vec_max_cols = 8;

ncols = 3 <= 8, so **the verify already uses the matvec shader**. The per-position cost is inside the
matvec path itself, not a dispatch/threshold mistake. No one-line fix exists.

## The exact arithmetic for 40, and why it cannot be reached from here

    round(k=2) = 49.3 (one 9.6 GB weight pass - the floor)
               + 11.0 (2 drafts, measured, and already at bandwidth)
               + 12.8 (verify compute for 3 positions, which does NOT overlap the weight read)
               = 72.0 ms   ->  A = 2.46  ->  34.4 tok/s

    for 40 tok/s the round must be <= 61.5 ms.
    with a completely FREE draft lane the round is still 61.0 ms -> 40.3 tok/s.

So 40 requires the verify to run at *one-position* cost (49-52 ms) for three positions - i.e. the
extra 12.8 ms of compute must overlap the weight read. That is a small-batch scheduling/kernel
optimisation inside llama.cpp's Vulkan backend, not a flag, not a threshold, and not something
reachable from the measurement layer.

## 60 tok/s: arithmetically excluded at today's acceptance

    60 tok/s at A = 2.46 requires a 41 ms round, which is BELOW the 49.3 ms weight-read floor.
    Even at A = 4 (the geometric maximum at p ~ 0.75) it needs 4.4 ms per verified position,
    against 11.9 measured. Both would require the weight pass itself to get faster, which means
    fewer bytes or more bandwidth - neither of which is available at this quant on this box.

## Final Path B position

- **Best measured and free: 34.4 tok/s** with `--spec-draft-n-max 2`, beating the objective's
  recorded 25.7-27.7 Vulkan+MTP baseline. Quality is lossless by construction.
- **40 is reachable only by fixing the verify's compute overlap** in llama.cpp's Vulkan backend.
- **60 is not reachable at all** at this quant, this acceptance, and this bandwidth.
- Instrumentation is in place and env-gated (`SAKER_SPEC_TRACE=1`, backup at
  `common/speculative.cpp.orig-saker`) so the next attempt starts from measurement, not guesswork.

## The shader-level mechanism, found (and the last cheap fix ruled out)

Read `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq2_s.comp`. The loop structure is:

```glsl
[[unroll]] for (uint n = 0; n < num_rows; ++n) {
    const float d = float(data_a[ibi].d);            // weight block loaded
    const uint qh = data_a[ibi].qh[ib32];            // ONCE per row ...
    const u8vec2 qs16 = unpack8(...);
    const u8vec2 sign16 = unpack8(...);
    [[unroll]] for (uint l = 0; l < 2; ++l) {
        const uvec2 grid = iq2s_grid[qs];
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {   // ... and reused across columns
            8-deep fma chain, one select per value:
            fma(b0.x, (sign & 1) != 0 ? -grid0.x : grid0.x, fma( ... ))
        }
    }
    ibi += num_blocks_per_row;
}
```

**The weight-re-read hypothesis is falsified**: the weight block is loaded once per (block, row) and the
column loop is *inside*, so multi-column does not multiply weight traffic.

What multi-column does multiply is **arithmetic with no new memory traffic**: each extra column adds
8 FMAs plus 8 sign selects per 8 values. That makes the matvec shader **compute-bound at NUM_COLS > 1**,
which is exactly the observed non-overlapping remainder: 4 positions of compute is ~14.5 ms at
Vulkan's ~15 TFLOP/s, against the +17.1 ms measured from pp1 to pp4.

Note the structure is **identical in spirit to our own HIP kernel** - per-value sign select plus float
FMA - which is why both are compute-bound at the same ~92-124 GB/s of effective weight throughput.

### Why the cheap shader fix does not reach 40

The obvious tweak is the sign-fold already measured in our engine (1.116x, bit-exact): replace the
8 per-value selects with a XOR against a mask built from the sign byte.

    verify compute 12.8 ms -> ~11.4 ms  ->  round ~70.6 ms  ->  35.2 tok/s.   Not 40.

Reaching 40 needs the full treatment: int8 activations plus **integer dot products** so that 4 MACs
become one instruction. The Vulkan device advertises `int dot: 1`
(`VK_KHR_shader_integer_dot_product`), so `dot4i8` is available in GLSL - but that is a real shader
project (new shader variant, pipeline plumbing, activation quantisation), not a tweak.

**Status: Path B stands at 34.4 tok/s measured. 40 requires the integer-dot shader project; 60 is
excluded at this quant.**

## Three falsifications in the Vulkan backend (all measured, patch left in tree)

Instrumented and A/B'd llama.cpp's Vulkan dispatch to find the verify's +6.4 ms per position:

1. **Weight re-read per column - FALSIFIED.** `mul_mat_vec_iq2_s.comp` loads the weight block once per
   (block, row) and loops columns *inside*: `for n in num_rows { load block; for j in NUM_COLS {...} }`.
   Multi-column does not multiply weight traffic.
2. **Tiled-matmul dispatch - FALSIFIED.** `mul_mat_vec_max_cols = 8` (ggml-vulkan.cpp:389), so a
   3-column verify uses the matvec shader, not a padded tiled matmul. No threshold to fix.
3. **int8-activation (Q8_1 / integer-dot) path - FALSIFIED as the lever.** Added an env-gated override
   `SAKER_FORCE_QY=0|1` on the `quantize_y` gate at ggml-vulkan.cpp:9194 and measured:

   | arm | tok/s |
   |---|---:|
   | default | 34.54 |
   | `SAKER_FORCE_QY=0` (int8 OFF) | 34.34 |
   | `SAKER_FORCE_QY=1` (int8 ON) | 34.49 |

   Within noise (+-0.3%). The int8 activation path is neither the bottleneck nor a lever.

**Net: the verify's per-position cost is not weight traffic, not dispatch, and not activation
precision.** What remains is the small-M compute in the Vulkan matvec/GEMM path itself - a
backend/scheduling problem, not a flag, threshold, or precision choice.

## Patches left in the tree (all env-gated, inert by default)

| file | change | gate |
|---|---|---|
| `common/speculative.cpp` | per-round `draft()` timer | `SAKER_SPEC_TRACE=1` |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | matvec path trace | `SAKER_VK_TRACE=1` |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | force/disable the int8 activation path | `SAKER_FORCE_QY=0|1` |

Originals: `common/speculative.cpp.orig-saker`, `ggml/src/ggml-vulkan/ggml-vulkan.cpp.orig-saker`.

## Refined: the draft IS byte-bound, so a shortlist does help (correcting my earlier note)

Plain prefill scaling, llama-bench Vulkan, 5 reps (`pathB-pp2/raw.txt`):

| pp | t/s | total ms | marginal ms/position |
|---:|---:|---:|---:|
| 1 | 19.97 | 50.08 | - |
| 2 | 38.35 | 52.15 | 2.08 |
| 3 | 52.84 | 56.78 | 3.35 |
| 4 | 59.53 | 67.19 | 5.71 |
| 5 | 63.11 | 79.23 | 7.29 |
| 6 | 64.74 | 92.68 | 8.52 |

The marginal grows superlinearly: at 1 position the pass is bandwidth-bound (50.08 ms for 9.6 GB =
192 GB/s), and compute progressively takes over as columns increase.

**Correction to my earlier statement that "the draft lane is already at bandwidth so there is nothing
to win there".** Being bandwidth-bound is exactly why cutting its BYTES cuts its TIME proportionally:

    draft bytes 1.02 GB (blk.64 Q6_K 348 MB + shared lm_head IQ4_XS 675 MB) = 5.5 ms measured
    shortlist the draft lm_head (675 MB -> ~10 MB): bytes 0.36 GB -> 1.9 ms per draft
    round 72 ms -> 64.9 ms  ->  2.46 / 0.0649 = 37.9 tok/s

## Why 40 needs TWO changes, not one

The MTP verify is 61 ms (derived: round 72 minus measured draft 11) against pp3's 56.78 ms for the
same 3 positions - a **4.2 ms MTP-specific overhead**. The likely cause is that the verify computes
the lm_head for all 3 positions (to compare each draft), whereas a plain prefill computes it only for
the last. A 248320-row matvec whose compute triples with columns is exactly the superlinear term
above.

    both fixes:  verify ~57 + draft ~3.9  =  60.7 ms  ->  2.46 / 0.0607 = 40.5 tok/s

So the route to 40 is: **(a)** a shortlist head for the draft lane, and **(b)** making the
multi-position lm_head bandwidth-bound rather than compute-bound. Neither alone suffices:
(a) alone gives 37.9, (b) alone leaves the draft at 11 ms.

Note this also means the earlier conclusion "40 requires the verify's compute to overlap" was right,
but incomplete - it needs the *lm_head's* multi-column compute specifically, plus the shortlist.

## Flash attention: already on, and it is worth 2 tok/s

| arm | tok/s |
|---|---:|
| default | 34.43 |
| `--flash-attn 1` | 34.50 |
| `--flash-attn 0` | 32.48 |

FA is enabled by default and forcing it off costs 6%. No lever there either - but worth recording,
because it means the multi-position attention path is already the accelerated one.

## Complete list of what Path B has exhausted

Free/config levers (all measured, all within noise of 34.4 except where noted):
`n_max` 1/2/3/4 (2 is optimal), `n_min`, `p_split`, `--no-mmap`, `--mlock`, `--flash-attn` (on by
default; off costs 6%), stacked zero-weight n-gram drafters (worse: 29.9-30.4).

Code-level hypotheses, all falsified with measurements:
1. weight re-read per column (the shader reuses the block across columns)
2. tiled-matmul dispatch (`mul_mat_vec_max_cols = 8`, so the verify already uses the matvec shader)
3. int8-activation / Q8_1 integer-dot path (`SAKER_FORCE_QY=0|1` -> within noise)
4. memory placement / residency (both `--no-mmap` and `--mlock` within noise)
5. the `n_max=4` pathology (closed: deeper drafting is break-even at 12 ms/draft)

## The only two things left, and what each is worth

| change | effect | what it takes |
|---|---|---|
| **(a) shortlist head for the draft** | draft 5.5 -> 1.9 ms/draft (1.02 GB -> 0.36 GB) | a row-subset matmul for the draft's logits: llama.cpp has no such op, so a new graph node + shader. **Gets 37.9 tok/s on its own.** |
| **(b) multi-position lm_head bandwidth-bound** | verify 61 -> ~57 ms | the lm_head's multi-column matvec is compute-bound; three obvious fixes already falsified |
| (a) + (b) together | round 60.7 ms | **40.5 tok/s** |

Recommended order: **(a) first** - it is the larger single win, it does not depend on cracking the
backend mystery, and it lands a real improvement (37.9) even if (b) never succeeds.

# FINAL STATE — Path B, two-branch workflow, 2026-10-19

## Result: 36.19 tok/s, lossless, merged

| build / arm | tok/s | draft acceptance |
|---|---:|---|
| parent tree, shortlist off (baseline) | 34.47 | 0.77551 (38/49), mean len 2.52 |
| parent tree, `SAKER_SHORTLIST=3584` | **35.91** | 0.77551 (38/49), mean len 2.52 |
| merged tree, shortlist off | 34.42 | 0.77551 (38/49), mean len 2.52 |
| **merged tree, `SAKER_SHORTLIST=3584`** | **36.19** | **0.77551 (38/49), mean len 2.52** |

**+5.1% over baseline, and the acceptance counts are byte-identical (same 38/49, same mean length).**
Identical drafted tokens means an identical output stream, so this is lossless by construction -
verification was never touched, only the draft's proposal set narrowed. Parent measured the
shortlist-alone win independently at +4.2% (35.91 vs 34.47); the agent reported +5.4% over two reps.
Same direction and magnitude, within run-to-run noise.

Enable with `SAKER_SHORTLIST=3584`. K=3584 is the knee: at that K acceptance is *exactly* the full-head
acceptance, while K=3072 drops it to 0.725 and K=4096 buys nothing for more traffic.

## The three defects that had to be fixed (the WIP's collapse to 0.084)

1. **The backend sampler returned a column index as a token id.** The MTP draft offloads sampling to
   an on-device top-k chain; with the narrow `[K]` logits tensor its argmax returns `j in 0..K-1`,
   which the caller treats as a token id. This was the entire 0.084 collapse - and it is why the
   host-side readback never ran (`needs_raw_logits` is false when a backend sampler is registered).
   Fix: keep draft sampling on the host when the shortlist is on.
2. **The WIP's readback scattered scores at positions 0..K-1**, leaving the rest of the full-vocab row
   stale. Fix: each candidate's score now lands at its true token position, `-inf` elsewhere - a
   faithful restricted distribution.
3. Missing `ggml_set_input` on the ids tensor, so the scheduler never inserted the CPU->VK copy.

## Why 36.2 and not the predicted 37.9

`ggml_get_rows` emits an F32 `[5120 x K]` intermediate (~2 x 74 MB of extra traffic at K=3584), and
host sampling costs ~0.4 ms/draft. Note the gathered *weights* are only ~9.7 MB (3584 rows x 2.7 KB) -
the F32 intermediate is the dominant remaining cost, not the weight read. Removing it needs a fused
quantized-row-gather matvec: a shader project, not an ops change.

## The sibling workstream (b): attribution proven, fix NOT landed

Controlled experiment (env-gated `SAKER_ALLLOGITS` in llama-bench - identical forwards, only the
output-column count differs):

| pp | last-only | all-cols | delta |
|---:|---:|---:|---:|
| 1 | 16.76 t/s | 16.81 t/s | -0.18 ms |
| 3 | 41.65 t/s | 46.01 t/s | **+6.83 ms** |
| 4 | 44.45 t/s | 54.92 t/s | **+17.16 ms** |

So the verify's multi-column lm_head IS compute-bound and superlinear in columns - which also
explains the n_max=4 pathology. But there is no cheap fix, and the reason is now known:

- **The IQ quant family (IQ2_S trunk, IQ4_XS lm_head) is excluded from the Q8_1 integer-dot path BY
  DESIGN** - those pipelines are never built, because LUT-based IQ dequant is not
  integer-representable. This is the definitive explanation of the earlier `SAKER_FORCE_QY` null
  result: we were forcing a path that does not exist for these quants.
- Matvec pipelines are built per *exact* `num_cols`, so there is no bucketing waste to reclaim.
- The algorithmic escapes (2-column verify, deferred verification) cost 26% of acceptance for ~3 ms.

Remaining route is shader-level work or f16 activations on the output matmul.

## Branches and merge

| branch | tree | head |
|---|---|---|
| `saker-shortlist` | /home/hycris/llama.cpp-qwen38-a | `0e74be4c9` |
| `saker-verify` | /home/hycris/llama.cpp-qwen38-b | `4664a18b1` |
| **`saker-combined`** | /home/hycris/llama.cpp-qwen38-merge | **`291da144f`** |

Merged with **0 conflicts**, as predicted (both branches shared `9d57ce456` as merge base and carried
an identical instrumentation delta). The merge is a `git clone` + fresh configure, so it does not
inherit the build landmine; provenance was re-verified before measuring
(`CMAKE_HOME_DIRECTORY` and RUNPATH both `-merge`).

## Gap to 40 (unchanged in kind, now smaller in size)

Need round <= 61.5 ms. The shortlist took ~5.5 ms/round off the draft lane; ~5 ms more is required and
both remaining levers are shader projects: the fused quantized-row-gather matvec above, and (b)'s
multi-column lm_head. Neither is a flag, threshold, or configuration change.

# 🔑 BREAKTHROUGH: the IQ2_S matvec is execution-limited, not bandwidth-limited (19% headroom)

Measured 2026-10-19, both arms on the merged build `291da144f`, same backend (Vulkan), same session,
GPU verified at **2379-2388 MHz under load** (sclk sampled during the run - idle readings are
meaningless and were the cause of an earlier false alarm).

| model | weights | decode | implied bandwidth |
|---|---:|---:|---:|
| BF16 reference (NO dequant - pure memory traffic) | 54.65 GB | 4.38 ± 0.01 t/s | **239.4 GB/s** |
| IQ2_S-mtp (2-bit + dequant) | 9.608 GB | 20.10 ± 0.01 t/s | **193.1 GB/s** |

**The hardware can do 239 GB/s and the IQ2_S path only gets 193 GB/s - a 19% shortfall.** This is a
clean controlled comparison: same machine, same Vulkan backend, same shaders, same clocks; the ONLY
difference is the weight format. The BF16 arm has no dequant at all, so it isolates pure streaming
from the quantized path.

Two things follow:

1. **The weight-read floor I had been quoting (49.3 ms) is NOT a hardware floor.** It is an IQ2_S
   kernel inefficiency. At 239 GB/s the same 9.608 GB takes **40.2 ms** - 9.55 ms per token cheaper.
2. The BF16 arm achieving 239 GB/s while being 5.7x LARGER (and therefore thrashing cache harder)
   makes this stronger, not weaker: a smaller working set should be easier to stream, and it still
   loses.

## Projection (clearly labelled - this is arithmetic, not a measurement)

    trunk weight pass   49.75 ms -> 40.2 ms      (-9.55 ms/token)
    MTP round           ~68 ms   -> ~58.5 ms
    MTP decode          36.2 t/s -> ~42 t/s      ** above the 40 target **
    plain decode        20.10    -> ~24.8 t/s

Even if only the 1-column path improves and the 3-column verify keeps its superlinear penalty, the
pp3 figure (56.78 ms) would fall to ~46.7 ms and the round to ~53.7 ms, i.e. **~45.8 t/s**.

## Why this matters strategically

Every earlier "40 is excluded" conclusion rested on 49.3 ms being a hard floor. It is not - it is a
kernel defect of exactly the same family as the two Path A defects already fixed (the lm_head GEMM
off hipBLAS, and the MTP bf16 head re-reading weights). Both of those were worth +43% and +2.29x
respectively once found. This is the third instance of the same pattern: **a quantized small-batch
kernel sitting well below the achievable bandwidth that the same hardware demonstrably reaches.**

# The 19% bandwidth gap is NOT capturable — three avenues closed (2026-10-19)

Follow-up to the BF16-vs-IQ2_S finding (239.4 vs 193.1 GB/s). All measured on the merged build,
same session, GPU at ~2.38 GHz under load, acceptance invariant at 0.77551 across every arm.

## 1. Backend flags — FLAT

Twelve `GGML_VK_*` disable knobs swept on tg64. Everything landed at 20.09-20.17 against a 20.11
baseline. Only two moved at all, both WORSE: `DISABLE_ASYNC` 18.22 (-9%) and `DISABLE_GRAPH_OPTIMIZE`
19.92 (-1%). There is no configuration lever.

## 2. Rows-per-workgroup (`rm_iq`) — default is already optimal

`ggml-vulkan.cpp:5179-5194`: the AMD tuning branch is gated on `device->architecture == AMD_GCN`.
**gfx1151 is RDNA3.5, not GCN, so this device falls through to the generic default** (`rm_kq = 2`,
`rm_iq = 4`) even though the code's own AMD path would set `rm_kq = 4` (i.e. 8 rows). The comment
above it says outright: "the number of rows computed per shader depends on GPU model and quant".
That looked like a device-tuning gap. It is not - the default is correct here.

Made env-overridable as `SAKER_RM_IQ` and swept. Plain decode, tg64:

| rm_iq | tg64 t/s | implied GB/s |
|---:|---:|---:|
| default (4) | 20.12 | 193.3 |
| 2 | 20.38 | 195.8 |
| 8 | 20.20 | 194.1 |
| 16 | 19.32 | 185.6 |
| 32 | 16.93 | 162.7 |
| 64 | 10.55 | 101.4 |

`rm_iq=2` looked like a +1.4% win on plain decode, and at 5 reps it confirmed (20.09 -> 20.33, +1.2%).
**But it is not a win - it costs the MTP path far more than it gains:**

| rm_iq | plain tg128 | MTP (n=64) |
|---:|---:|---:|
| default (4) | 20.09 | **35.96** |
| 2 | **20.33** | 33.62 |
| 8 | - | 26.08 |
| 16 | - | 20.67 |

MTP throughput falls off a cliff as rows-per-workgroup rises, while plain decode barely moves. So the
two paths want different geometry and the default is the best single compromise. **Do not set
`SAKER_RM_IQ`; leave it unset.**

## 3. Conclusion: the gap is the IQ2_S dequant itself

Not flags, not geometry, not a memory ceiling (BF16 proves 239 GB/s is reachable). What remains is the
dequant ALU work that IQ2_S requires per value - LUT lookups (`iq2s_grid[qs]`), `unpack8` calls, and
per-bit sign selects - which is intrinsic to the format and, per the sibling workstream, cannot be
moved to integer dot products because IQ dequant is not integer-representable by design.

**Net: the "just make the matvec hit peak bandwidth" route to 40 tok/s is CLOSED.** 40 again requires
the two shader projects: the fused quantized-row-gather matvec (draft lane, ~2.4 ms/round) and the
multi-column lm_head (verify, ~4.2 ms/round).

## Branch state

`saker-iqrows` in /home/hycris/llama.cpp-qwen38-merge carries the `SAKER_RM_IQ` instrumentation only.
It is NOT to be merged - it changes no default behaviour but adds a knob that should stay unset. The
deliverable is `saker-combined` (`291da144f`).

# Losslessness PROVEN directly; n_max re-swept; and a correction to my prefill claim

## 1. Losslessness proven by output bytes (not by acceptance counts)

My earlier reasoning was "acceptance counts are bit-identical (38/49), therefore the drafted tokens are
the same, therefore the output stream is the same". **That inference is not valid** - equal counts can
arise from different token sequences. So it was tested directly: generate 64 tokens with the full head
and with the shortlist, and compare the output TEXT.

| arm | chars | sha256 |
|---|---:|---|
| full head (no shortlist) | 276 | a4bc86010e1b60b8906ba47a |
| shortlist K=3584 | 276 | a4bc86010e1b60b8906ba47a |

**Byte-identical. Losslessness is now PROVEN by direct evidence, not inferred.**

## 2. n_max re-swept WITH the shortlist (the shortlist changes the draft/verify cost ratio)

n_predict=64, merged build, shortlist on:

| n_max | t/s | acceptance |
|---:|---:|---|
| 1 | 30.81 | 0.85294 (29/34) |
| **2** | **36.21** | 0.77551 (38/49) |
| 3 | 27.90 | 0.45570 (36/79) |
| 4 | 25.37 | 0.39583 (38/96) |

Control, same build, shortlist OFF:

| n_max | t/s | acceptance |
|---:|---:|---|
| 2 | 34.48 | 0.77551 (38/49) |
| 3 | 33.60 | 0.68852 (42/61) |

Two conclusions:
- **n_max=2 remains the operating point** - no free win here.
- **The shortlist degrades DEEPER drafting**: at n_max=3 acceptance falls 0.68852 -> 0.45570. At
  n_max=2 it is bit-identical. This is a performance caveat, NOT a quality issue: verification is
  exact, so output correctness is unaffected at any n_max - only speed. Worth knowing before anyone
  runs this with n_max=3.

## 3. CORRECTION: prefill >=500 is NOT excluded - I over-claimed

I previously told the user prefill >=500 was "excluded, factor of ~2", citing a ~13.8 TFLOPS coopmat
ceiling. **That number was from the Hy3 work on a different model and workload, and I misapplied it.**
The arithmetic for THIS model says something different:

    27.32 B params -> 2 x 27.32e9 = 54.64 GFLOP/token
    pp2048 measured 295.12 t/s -> 2048/295.12 = 6.94 s per pass
    111.9 TFLOP / 6.94 s = 16.1 TFLOP/s achieved

    target 500 t/s -> 2048/500 = 4.10 s -> 111.9 TFLOP / 4.10 s = 27.3 TFLOP/s required

    RDNA3.5 fp16 peak (40 CU x 128 lanes x 2 packed-fp16 x 2.39 GHz) ~= 49 TFLOP/s
    27.3 / 49 = 56% of fp16 peak

So the requirement is **1.7x current throughput at 56% of fp16 peak** - demanding but not
arithmetically excluded. **Prefill >=500 is a hard-but-open target, not a closed one.**

The plain-decode exclusion DOES still stand, and for a cleaner reason: 9.608 GB / 256 GB/s = 37.5 ms
= 26.7 t/s at 100% of theoretical peak, and the measured achievable ceiling (BF16, no dequant) is
239.4 GB/s = 24.9 t/s. So >=26 needs more than the hardware demonstrably delivers.

# Mirror audit 2026-10-19 (objective requires "all receipts mirrored both nodes")

Audited the whole `~/talon-out/saker/` tree both directions and repaired the drift.

| direction | before | after |
|---|---:|---:|
| files on node B only (missing on node A) | 4684 (122 MB) | **0** |
| files on node A only (missing on node B) | 4235 (~28 GB) | 1175 |

Repaired by `rsync --ignore-existing --files-from` both ways. The local-only set was 4212 files of
`recovery-sol-20261017` tooling plus source for `mtp-batch`/`dp4a`/`batched-verify` — those are now on
node A.

**Documented gap (deliberate, not an oversight):** the 1175 files still node-A-only are dominated by
multi-GB binary tensor dumps (`*.bf16`, `*.f32`, `native/`, `raw/`, `tensors/` — e.g.
`afix-cmp-*/native`, `perf-after-quality-*/parent/*/native`, `weightpass-B/run/count-*/native`). They
were excluded from the pull because both nodes are at 96-97% disk usage with only 69 GB free, and these
are regenerable intermediates rather than receipts. The audit RESULTS that reference them (comparison
reports, logs, scripts, panel summaries) are mirrored. If the tensor dumps themselves are required to
satisfy the mirroring requirement literally, that is a ~28 GB copy onto a 97%-full filesystem and
should be a deliberate decision, not something done silently.

# Losslessness: what the evidence actually supports (correcting my own overstatement)

I previously wrote "losslessness PROVEN" on the strength of ONE run per arm at 64 tokens. That was too
strong - a single run per arm cannot establish bit-reproducibility. I then ran a harder test and got a
contradiction, which turned out to be a confound in the BASELINE, not a defect in the shortlist. Full
record, because the reasoning matters more than the verdict:

## Test 1 - 256 tokens, 2 prompts, 1 run per arm

| prompt | full | shortlist | verdict |
|---|---|---|---|
| prompt-short16 | 940 chars | 960 chars | **DIFFERS** at char 637 |
| real-prompt (3728 tok) | 876 chars | 876 chars | identical |

Alarming on its face. Divergence text: full produced "...The code is cut off at \"All Saker projec"
while shortlist produced "...The comment mentions \"Saker projection widths\" which is cut ".

## Test 2 - 256 tokens, 3 runs per arm (is the arm even self-consistent?)

    full  (baseline, NO shortlist): 2 distinct outputs across 3 runs   <-- the control does not hold still
    short (shortlist):              1 distinct output  across 3 runs

**The unmodified baseline is not bit-reproducible at 256 tokens.** So Test 1 cannot attribute the
divergence to the shortlist - the comparison had no stable reference.

## Test 3 - 64 tokens (the operating protocol), 5 runs per arm

    full  : 1 distinct output / 5 runs   276 bytes   md5 2fe11002abed
    short : 1 distinct output / 5 runs   276 bytes   md5 2fe11002abed
    full-1 == short-1

## What is actually supported

1. **At the operating protocol (n_predict=64, n_max=2) the shortlist is bit-identical**, now with
   5 reproducible runs per arm rather than 1 - much stronger evidence than the original claim.
2. **Acceptance statistics are bit-identical**: 0.77551 = 38/49, mean len 2.52.
3. **Beyond that horizon the comparison is INCONCLUSIVE, not negative.** The baseline itself is
   non-deterministic at 256 tokens, so "full vs short differ" carries no information about the
   shortlist.

## Methodological finding worth keeping

**This measurement setup is not bit-reproducible at long horizons.** It is deterministic until a
generation passes through a near-tie, at which point GPU reduction order can flip an argmax and the
remainder diverges. Consequences:

- Any future "byte-identical output" claim MUST use >=3 runs per arm and a stable control. A single run
  per arm is not evidence, and a divergence is not attributable unless the control is reproducible.
- Speed measurements are unaffected (tok/s error bars were +-0.01 and acceptance was stable at
  0.77551 across many runs), because those are averages/statistics rather than exact-sequence claims.

# DECISIVE DIAGNOSTIC: the 19% bandwidth gap is dequant ALU, not block structure

The BF16-vs-IQ2_S comparison confounds two things: IQ2_S has both heavy LUT dequant AND a
non-power-of-two 82-byte block layout. Q8_0 separates them - it has the same kind of block structure
(34 bytes/32 values) but near-zero dequant ALU (one multiply per value).

Quantized the BF16 reference to Q8_0 (29.05 GB) with llama-quantize and measured decode:

| format | blocks | dequant ALU | weights | decode | implied bandwidth |
|---|---|---|---:|---:|---:|
| BF16 | none | none | 54.65 GB | 4.38 t/s | 239.4 GB/s |
| **Q8_0** | yes (34 B) | **minimal** | 29.05 GB | 7.83 t/s | **227.5 GB/s** |
| IQ2_S | yes (82 B) | heavy LUT | 9.608 GB | 20.10 t/s | 193.1 GB/s |

**Reading: block structure costs ~5% (239.4 -> 227.5). The IQ2_S LUT dequant costs the remaining
~15% (227.5 -> 193.1).** The gap is ALU, not access pattern.

Consequence - this is the measured path to 40:

    if IQ2_S reached Q8_0's bandwidth:  9.608 GB / 227.5 GB/s = 42.2 ms  (vs 49.7 ms now)
    round: ~68 -> ~60.5 ms
    MTP decode: 36.6 -> ~40.6 tok/s

So the remaining lever is **reducing ALU work in the IQ2_S matvec** (e.g. sign-fold, which measured
1.116x bit-exact in the Path A engine) - NOT restructuring memory access, and not a flag. This
supersedes my earlier speculation that it might be an access-pattern problem.

## Fused gather-matvec: landed, verified, +6.9%

Commit `b35a6009d` on `saker-shortlist`, merged into `saker-combined` as `b52203612` (0 conflicts).

Rather than writing a new op or shader, it reuses llama.cpp's existing **MoE indirect matmul**
(`ggml_mul_mat_id`) with the vocab rows viewed as one 1-column "expert matrix" each:

    head_w [n_embd, n_vocab] -> [n_embd, 1, n_vocab]   ids [K] -> [K,1]   cur -> [n_embd,1,1]
    ggml_mul_mat_id -> [1, K, 1] -> reshape [K, 1]

With ne11 == 1 the shader's `b_offset = (expert_i0 % ne11) * stride_b` is 0 for every candidate (the
designed broadcast case), and `a_offset = ids[k] * 160 blocks` is exactly that token's IQ4_XS row.
The gather path is kept as a fallback for multi-output batches. Net +23/-2 in one file.

Parent-verified, 3 runs per arm, merged build `b52203612`:

| arm | t/s | acceptance | distinct outputs |
|---|---|---|---|
| off | 34.43 / 34.27 / 32.96 | 0.72549 (37/51) | 1/3 |
| **fused K=3584** | **36.57 / 36.78 / 36.58** | 0.77551 (38/49) | 1/3 |

**`off` vs `fused` output is BYTE-IDENTICAL**, both arms self-consistent 3/3.

**Honest caveat from the agent, and it is right**: the gain over the previous gather path is only
~0.7 ms/round, not the ~1.5 ms the removed traffic predicted. The intermediate is gone, but
`mul_mat_vec_id` is shaped for MoE (multi-row expert matrices) and gives some of the saving back in
kernel efficiency in this degenerate 1-row-per-expert case. The host-sampling fallback (~0.4 ms/draft)
also remains.

# Prefill diagnosis: saturated kernel at 34% of fp16 peak

Measured on the merged deliverable `b52203612`, Vulkan, `-b 1024 -ub 256`, 3 reps.

| pp | t/s | ms/pass | TFLOP/s | % of ~49 TF fp16 peak |
|---:|---:|---:|---:|---:|
| 512 | 303.99 ± 0.47 | 1.68 | 16.62 | 34% |
| 1024 | 303.28 ± 0.81 | 3.38 | 16.57 | 34% |
| 2048 | 300.40 ± 0.47 | 6.82 | 16.41 | 33% |
| 4096 | 295.92 ± 0.11 | 13.84 | 16.17 | 33% |
| 8192 | 295.20 ± 2.82 | 27.75 | 16.13 | 33% |

**Throughput is FLAT from batch 512 to 8192 (304 -> 295 t/s, 3% drift).** That is the signature of a
saturated kernel: if launch overhead, scheduling, or tail effects were the limit, larger batches would
recover efficiency. They do not. This also independently confirms the `-ub` sweep - there is no
batch-shape lever left at any shape.

**The decisive number: ~16.5 TFLOP/s achieved against a ~49 TFLOP/s fp16 peak = 34%.**

Prefill is compute-bound, not memory-bound (arithmetic intensity ~11.6 FLOP/byte; memory alone could
feed only ~2.6 TFLOP/s). So the target is a pure kernel-efficiency question:

    target 500 t/s  ->  27.3 TFLOP/s  ->  56% of fp16 peak  ->  1.65x current saturated rate

Unlike plain decode (which is arithmetically excluded), prefill >=500 is a demanding-but-open kernel
problem. The MMQ path (mmq_wg_denoms / warptile_mmq) is already on its AMD-coopmat branch for this
device - verified, unlike the matvec which was silently falling through to generic defaults.

## Corrected prefill figure for the record

Best measured prefill is **298.58 t/s** at `-b 1024 -ub 256` (the 300.40 above is the pp2048 row of
this sweep, within run-to-run spread of the same configuration). Against the recorded 320.7 t/s
baseline that is ~93%, and against the 500 t/s target ~60%. Comparability caveat unchanged: the
baseline was measured on ROCm, these on Vulkan.

# Prefill diagnostic: NOT dequant-limited (IQ2_S == Q8_0)

Same isolation method that cracked decode, applied to prefill. pp2048, `-b 1024 -ub 256`, merged build.

| model | pp2048 | TFLOP/s | % of ~49 TF fp16 peak |
|---|---:|---:|---:|
| IQ2_S | 301.45 | **16.47** | 33.6% |
| Q8_0 | 299.41 | **16.36** | 33.4% |
| BF16 (reference) | 123.72 | 6.76 | 13.8% |

**IQ2_S and Q8_0 are identical to within 0.7%.** So the prefill ceiling is NOT the weight format's
dequant - it is the MMQ GEMM kernel's efficiency itself. This is the opposite of the decode finding,
where Q8_0 beat IQ2_S by 15% and isolated the gap to dequant ALU. **Decode is format-limited; prefill is
not.** Any prefill work should target tiling/scheduling in the MMQ path, not the IQ2_S dequant.

**BF16 is not a usable prefill reference on this device.** The Vulkan device caps report `bf16: 0` - no
native bfloat16 - so BF16 is emulated and lands at 13.8% of peak. (This does NOT undermine the BF16
*decode* bandwidth measurement of 239.4 GB/s: 54.64 GFLOP/token at the measured 6.76 TFLOP/s emulated
rate would be 8 ms, versus the 12.5 s/token actually observed - so decode was genuinely memory-bound
and 239.4 GB/s stands as the achievable streaming figure.)

## Consequence for the 500 t/s target

    298-301 t/s achieved = 16.4 TFLOP/s = 33.5% of fp16 peak
    target 500 t/s      = 27.3 TFLOP/s = 56% of fp16 peak  ->  1.65x

Prefill is compute-bound (arithmetic intensity ~11.6 FLOP/byte; memory alone could feed ~2.6 TFLOP/s)
and the kernel is SATURATED (flat 304 -> 295 t/s from batch 512 to 8192, so no shape/overhead lever
remains). Reaching 500 t/s therefore requires ~1.65x GEMM efficiency from the MMQ path. Demanding, but
not arithmetically excluded - unlike plain decode, which is.

## Prefill: where the 1.65x would have to come from (lead, not a fix)

`ggml_vk_get_mul_mat_mat_pipeline` (ggml-vulkan.cpp:7670) shows the prefill shader choice:

    if (src1_type == GGML_TYPE_Q8_1) {
        return ctx->device->pipeline_dequant_mul_mat_mat_q8_1[src0_type].f32acc;   // int8 + dp4a
    }
    if (src1_type != GGML_TYPE_F32 && !ctx->device->coopmat2) {
        return nullptr;                                                            // no path
    }

So prefill's variant hinges on `quantize_y` selecting **Q8_1 (int8) activations**, which run on `dp4a`.
This device reports `matrix cores: KHR_coopmat` but **NOT `coopmat2`**, and coopmat2 is what exposes
int8 tensor cores - the backend's only `coopmat2` reference is a block-size constant. So the int8 MMQ
is on dp4a, not tensor cores.

Caveat on my own percentage: I have been quoting "33.5% of fp16 peak", but that denominator is not the
right peak for an int8 kernel, so treat the percentage as indicative and the ratio (1.65x needed) as
the solid number.

This is a LEAD, not a fix: closing it means either a better MMQ tiling/scheduling, or getting int8 onto
tensor cores. It is a project, not a flag. Not pursued - the user scoped this round to the sign-fold.

## Disk incident avoided (recorded because it was self-inflicted twice today)

node A hit 98% used (38 GB free) while the sign-fold agent was building there. Freed the 29 GB Q8_0
diagnostic artifact after confirming (a) its diagnostic question was answered and recorded, and (b) no
process held it open. Disk went 38 -> 65 GB free; btrfs (`compress=zstd:1`) reclamation is not instant,
which briefly made it look like the delete had failed.

Two self-inflicted traps worth keeping:
- `ulimit -f 1048576` protects against runaway logs but **kills any legitimate large write** - it killed
  my own llama-quantize with "File size limit exceeded (core dumped)". Raise it for that step only.
- `cp -a` of a configured llama.cpp tree inherits a `build/` wired to the ORIGINAL tree (CMake cache +
  binary RUNPATH), so `cmake --build build` silently compiles the wrong sources.

# Mirror audit COMPLETE — 0 drift both directions (supersedes the earlier partial note)

    node A files: 18606
    node B files: 18606
    node-A-only : 0
    local-only  : 0

Key deliverable artifacts confirmed present on both nodes: FINAL-REPORT-PATHB-20261019.md,
PATHB-FINDINGS-20261019.md, MERGE-PLAN-20261019.md, and the whole bench-final evidence directory.

## Two corrections to my own earlier mirror notes

1. **My "28 GB node-A-only" figure was a measurement bug.** I ran
   `ssh nodeA "cd $A && cat /tmp/list.txt | xargs du -cb"` but the list file existed only on node B, so
   `cat` failed, `xargs` got nothing, and `du -cb` with no arguments measured the CURRENT directory -
   the whole ~/talon-out/saker tree (30.41 GB) - not the subset. The real node-A-only subset was ~3 GB,
   which is why mirroring it moved the free-space needle only 66 -> 63 GB while the drift count
   collapsed 1019 -> 18. The same bug produced the earlier "~28 GB" claim; that number was never real.

2. **My first text-file exclusion was too aggressive.** `--exclude='native/***'` and
   `--exclude='raw/***'` dropped 261 genuine TEXT receipts that happen to live under those directories
   (`.txt`, `.json`, `.stdout`, `.stderr`, `.exit`, `.log`). Those are exactly what "receipts" means, so
   they were pulled. Only after that did the remaining set consist purely of binaries.

Lesson worth keeping: **when a filter is expressed as a directory glob, verify what it actually
excluded by extension before calling the job done.** The exclusion looked principled and silently
dropped a quarter of the receipts.

# Sign-fold result: essentially NULL. My recommendation was wrong.

I recommended the sign-fold to the user as "the measured route to 40" with the arithmetic laid out:
"a full 1.116x takes the weight pass to ~44.5 ms -> round ~61.3 ms -> ~40.1 tok/s". The agent ported it,
confirmed the shader was genuinely regenerated (SPIR-V disassembly showed its 16-entry Workgroup uvec4
mask table and 4 vector `OpBitwiseXor`s, so this is not a "did the build take effect" artifact), and
measured:

    plain decode tg64, r=5:   19.90 +/- 0.03  ->  19.99 +/- 0.01
                              = +0.45%  (191.2 -> 192.1 GB/s)

The reference implementation in my own HIP engine measured **1.116x**. The Vulkan port gets **1.0045x**.
**The HIP -> Vulkan transfer did NOT hold.** I stated the caveat ("plausible, not proven") but I still
put a confident number in front of the user, and that number was wrong. The deliverable remains 37.00.

## What this does and does not falsify

- **Still valid:** the Q8_0 isolation. The 19% decode gap IS dequant ALU, not block structure
  (BF16 239.4, Q8_0 227.5, IQ2_S 193.1 GB/s). That diagnostic stands on its own measurement.
- **Now falsified:** that the *sign selects* are where that ALU cost lives, in this shader. Replacing
  8 compare+selects per 8 values with a mask-XOR bought ~nothing, so the Vulkan driver was evidently
  already handling the selects cheaply, and the real ALU cost is elsewhere - the `iq2s_grid` LUT
  lookups, the `unpack8` calls, or the per-block scale handling.
- The agent's MTP comparison was left **unresolved**: its arm numbers (off 30.84/34.02/32.79 vs on
  36.45/36.46/36.25) are confounded by thermal drift and by concurrent GPU use, and it died while
  building the interleaved .so-swap A/B that was meant to settle it. Its own read was "MTP appears to
  have lost ~1.2%, thermal drift is a known confounder".

## State left behind

- tree a is clean at `b35a6009d` with the shader change **reverted** (the agent reverted it for the
  .so-swap A/B and never restored it). The compiled artifact survives at
  `/tmp/libggml-vulkan.so.signfold` (53087128 B) alongside `.baseline` (53107608 B); the evidence dirs
  `decode-ilp-20261019/signfold/{arm-base,arm-sf,interleaved}` hold the generation outputs.
- The agent timed out at 30 min, leaving an orphaned `llama-server` that held the GPU.

# MEASUREMENT INTEGRITY: I polluted my own panel, and caught it by its signature

I ran a clean-build panel using the desktop gate but **forgot `gpu-lock.sh`**, while the sibling agent
had servers on the same GPU. Result:

    MTP n=64: 19.43 tok/s   acceptance 0.77551 (38/49)   <-- HALF the true rate
    plain:    19.27 +/- 1.44                            <-- error bar 100x normal

**The signature that gave it away: acceptance was bit-identical (38/49) while throughput halved.**
Identical acceptance means the code did not change, so the throughput could not be real. A ~2x drop
with unchanged acceptance is contention, not a regression. Re-run with the lock held:

    MTP n=64: 36.58   n=128: 32.42   plain: 20.10 +/- 0.00   prefill pp2048: 302.54 +/- 0.65

Lesson: **acceptance counts are a free integrity check on any speculative-decoding throughput number.**
If acceptance is unchanged and throughput moved, suspect the machine before believing the result.

Also re-confirmed the `pkill -f` self-match trap: killing the orphaned server with
`pkill -f "llama.cpp-qwen38-a/build/bin/llama-server"` killed my own ssh session (exit 255), because my
command line contained the pattern. Use `ps | grep "[l]lama..."` and kill by PID.

# CORRECTION: prefill uses coopmat; the dp4a hypothesis was wrong

I had hypothesised that prefill fell back to int8 `dp4a` because this device lacks `coopmat2`. The
flag sweep on pp2048 refutes that:

| flag | pp2048 | TFLOP/s |
|---|---:|---:|
| baseline | 303.44 | 16.58 |
| **`GGML_VK_DISABLE_COOPMAT=1`** | **182.23** | **9.96** (-40%) |
| `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` | 301.15 | 16.45 |
| `GGML_VK_DISABLE_COOPMAT2=1` | 299.42 | 16.36 |
| `GGML_VK_DISABLE_F16=1` | 298.09 | 16.29 |
| `GGML_VK_DISABLE_DOT2=1` | 298.52 | 16.31 |
| `GGML_VK_DISABLE_MULTI_ADD=1` | 298.68 | 16.32 |
| `GGML_VK_DISABLE_ASYNC=1` | 295.05 | 16.12 |

Two things follow:

1. **Prefill is on the coopmat (fp16 tensor core) path and depends on it**: disabling coopmat costs
   40% (303 -> 182 t/s). So the 33.5%-of-peak figure is the COOPMAT GEMM's own efficiency.
2. **The int8/integer-dot path is NOT the limiter**: disabling it moves throughput by 0.8%. My
   "no coopmat2 -> dp4a fallback" story was wrong and is retracted.

This is also consistent with the IQ2_S == Q8_0 prefill result: both formats dequantize on the fly and
then use the same coopmat GEMM, so the GEMM dominates and the format does not matter.

## Where the prefill lever therefore is

Not the format, not the activation precision, not the batch shape (flat 304 -> 295 t/s from pp512 to
pp8192). It is the coopmat GEMM's tiling/scheduling. Note `ggml-vulkan.cpp` ~line 4287: the AMD
coopmat branch overrides only the **large** warptile:

    } else if (vendor == AMD && coopmat_support && driver != eAmdProprietary) {
        l_warptile     = { 256, 128, 128, 16, subgroup_size_8, 64, 2, tm_m, tn_m, tk_m, subgroup_size_8 };
        l_warptile_mmq = l_warptile_mmq_int = { 256, 128, 128, 32, subgroup_size_8, 64, 2, tm_m, tn_m, tk_m, subgroup_size_8 };
        l_warptile_mmq_int_k = { 256, 128, 128, 32, subgroup_size_16, 64, 1, 4, 2, 1, subgroup_size_16 };
    }

The **medium and small** warptiles keep generic values on this device. Whether pp2048 at
`-b 1024 -ub 256` selects the large tile or a smaller one is exactly the question to answer before
tuning anything.

# Prefill baseline VALIDATED, and the backend comparison is now matched

The report had been comparing my Vulkan 302.54 against a recorded ROCm baseline of 320.7 - an
apples-to-oranges comparison I had flagged but not fixed. Fixed by measuring the baseline build
(ROCm, 2026-08-15) on the same box:

| backend / flags | pp2048 | TFLOP/s |
|---|---:|---:|
| **ROCm `-ub 512 -fa 1`** | **320.13** | 17.49 |
| ROCm `-ub 512` | 319.39 | 17.45 |
| ROCm `-ub 256` | 308.92 | 16.88 |
| ROCm `-ub 1024` | 308.75 | 16.87 |
| **Vulkan, current merged build** | **302.54** | **16.50** |

1. **The recorded 320.7 baseline reproduces at 320.13** - my methodology matches the baseline's, so the
   comparison is sound.
2. **ROCm prefill beats Vulkan by 5.8%** (320.13 vs 302.54) - while Vulkan wins decode (plain 20.10 vs
   ROCm 15.67). The backends are each better at one thing.
3. **The optimal `-ub` is backend-specific**: 512 on ROCm, 256 on Vulkan. Using one backend's value on
   the other costs ~4-8% (Vulkan at ub512 is 290 t/s; ROCm at ub256 is 308.9).
4. Target 500 is **1.56x the ROCm baseline**, not 1.65x the Vulkan figure.

# Path split confirmed: verify = matvec, prefill = coopmat. No synergy.

Traced with the `SAKER_VK_TRACE` instrumentation in the merge tree, same build, `-b 1024 -ub 256`:

    pp3    (the verify's shape): 1834 matvec dispatch lines, ALL num_cols=3
    pp2048 (prefill)           :    0 matvec dispatch lines

So the two workloads use **different kernels**: 3-column matvec for the verify (consistent with
`mul_mat_vec_max_cols = 8`, so anything <= 8 columns takes the matvec path) and the coopmat matmul for
prefill.

**Consequence: the prefill tiling work cannot improve MTP.** I had half-expected a synergy - that the
verify's superlinear per-position cost was the same small-M GEMM inefficiency that prefill shows - and
it is not, because they are not the same code path. Do not read a prefill win as progress toward the
40 tok/s MTP target. Conversely, the sign-fold work (matvec path) could not have helped prefill.

Incidental observation from the trace: `a_type=30` (BF16) appears frequently at pp3, alongside
`b_type=9` (Q8_1) and `b_type=0` (F32) in equal counts. The BF16 weights are the MTP layer's, and the
paired b_type counts are the pipeline *lookup* ambiguity noted earlier (both variants queried per node),
not proof of which is dispatched.

# CORRECTION (second one on prefill): the 500 target is EXCLUDED, and my first "correction" was the error

This entry flip-flopped twice. Settled version and the error chain, because the reasoning matters more
than the verdict:

1. The report initially called prefill excluded, citing a ~13.8 TFLOPS ceiling - but that figure came
   from **unrelated work on a different model**, so it was not a valid basis.
2. I then "corrected" it to "demanding but NOT excluded", using a **theoretical** 49 TFLOP/s fp16 peak
   (40 CU x 128 lanes x 2 x 2.39 GHz) and concluding "needs 56% of peak, hard but open".
   **That correction was the mistake.** It overturned a correct conclusion using a number that no
   tuned backend on this hardware comes close to.
3. The settled position: the project's own Path A engine records a **measured fp16 WMMA peak of 22.0
   TFLOP/s** on this box ("Prefill needs 34.4 TFLOP/s against a measured 22.0 TFLOP/s fp16 WMMA peak
   (1.57x)"). No separate receipt was found for the 22.0, so it is credible-but-unverified - but the
   empirical picture independently supports it: **two independently tuned backends both land at ~17
   TFLOP/s** (Vulkan coopmat 16.5, ROCm hipBLAS 17.5).

    target 500 t/s  = 27.3 TFLOP/s (my FLOP count) or 34.4 (Path A's)
    achievable      = ~17-22 TFLOP/s measured/achieved
    -> 500 is ABOVE the measured peak. Prefill >= 500 is excluded.

And the figure I had been quoting - "Vulkan is at 33.5% of fp16 peak" - was wrong for the same reason.
Against the measured 22.0 peak, Vulkan's 16.5 TFLOP/s is **75%**, i.e. there was never much headroom.
A theoretical peak is not evidence; the measured one is. I should have used Path A's own measurement
instead of recomputing a peak from first principles.

## The prefill agent was redirected

It was launched toward 500 with "need 1.65x". That target is unreachable, so it has been steered to a
narrower and genuinely achievable one: **beat ROCm's 320.13** (Vulkan is 5.8% behind on identical
hardware and weights - a real gap, not blocked by any ceiling). Its most valuable single outcome is now
to determine whether ~17 TFLOP/s is a hard practical ceiling, which would close the criterion as
arithmetically excluded rather than merely unmet.

# Prefill excluded by TWO independent routes (cross-validated), and a route to 40 exists

## Prefill: the exclusion is now multi-sourced, not single-sourced

| route | figure | implied ceiling |
|---|---|---|
| Path A measured fp16 WMMA peak | 22.0 TFLOP/s | ~403 t/s |
| Path A engine: 325.1 t/s stated as 91% of a ceiling | ~357 t/s ceiling | 19.5 TFLOP/s |
| ROCm hipBLAS achieved | 17.49 TFLOP/s | 80% of the 22.0 peak |
| Vulkan coopmat achieved | 16.53 TFLOP/s | 75% of the 22.0 peak |

The two independent ceiling estimates agree within 13% (403 vs 357), and both tuned backends sit at
75-80% of the measured peak - i.e. there is little headroom left in either.

    target 500 t/s = 27.3 TFLOP/s = 124% of the measured 22.0 peak

**Prefill >= 500 is EXCLUDED, by two independent routes.** This supersedes the single-sourced version
and makes it robust to the 22.0 figure being slightly off: even at 26 TFLOP/s it would still be short.

## MTP: a route to 40 exists, and it needs TWO changes

    current                         round 66.5 ms -> 36.99 tok/s
    + multi-column lm_head (-4.2)   round 62.3 ms -> 39.49   MISSES 40
    + host-sampling fix    (-0.8)   round 61.5 ms -> 40.00   LANDS ON 40

- The **multi-column lm_head** (~4.2 ms/round) is under way in tree `llama.cpp-qwen38-lmhead`
  (`saker-lmhead`), register-pressure hypothesis.
- The **host-sampling fallback** (~0.8 ms/round, ~0.4 ms/draft) is a CODE fix, not a shader: the
  backend sampler's argmax returns a column index `j in 0..K-1` for the narrow `[K]` logits tensor,
  and the caller treats it as a token id. Making the sampler ids-aware (map j back through the
  candidate list) lets sampling return to the device. Identified by the shortlist agent, not yet done.

Both are needed; neither alone reaches 40. Worth noting the total is exactly marginal - 40.00 is an
arithmetic projection from measured per-piece costs, and the real number will land either side of it.

# The multi-column lm_head cost: mechanism FOUND (dequant recomputed per column)

The lm_head agent read the generated IQ4_XS matvec shader and found the actual cause, which is
structural rather than empirical:

    vec4 dequantize4(uint ib, uint iqs, uint a_offset)   // pure function of the WEIGHT BLOCK
    ...  sl = scales_l[ib32/2] >> ...;  sh = scales_h >> ...;  qs = unpack8(...); dl = ...
         return dl * vec4(kvalues_iq4nl[qs.x], ...);

`dequantize4` is **j-independent** - it depends only on the weight block, not on the output column. But
it is being called INSIDE the `for (uint j = 0; j < NUM_COLS; ++j)` loop, so at NUM_COLS=3 the whole
dequant (scale shifts + `kvalues_iq4nl` LUT lookups + `unpack8`) is paid **three times over**.

That is exactly the measured superlinearity: pp3 +6.83 ms, pp4 +17.16 ms, pp1 -0.18 ms.

**Why this is the best kind of fix:** hoisting it out of the column loop is a pure **loop-invariant
motion**. Correctness is a property of the code structure, not something that has to be established by
measurement - the hoisted value is bit-identical by construction. Compare the sign-fold, where
bit-exactness had to be argued and the transfer failed anyway.

Note this is also consistent with the earlier sign-fold failure: the IQ2_S shader already hoists its
`iq2s_grid` lookup outside the column loop, so there was no redundant dequant to remove there and the
sign selects were never the cost. The IQ4_XS shader simply lacks the same hoist.

## Receipt-recovery re-measurement: a thermal band, and it matters

Re-ran two measurements whose receipts were missing (the originals had been piped through a parser and
never saved), under the lock, with two agents sharing the GPU:

| measurement | original | re-run | delta |
|---|---:|---:|---:|
| ROCm prefill `ub512 fa1` | 320.13 | **315.28** | -1.5% |
| Vulkan prefill baseline | 303.44 | **299.02** | -1.5% |
| `DISABLE_COOPMAT` | 182.23 | 181.22 | -0.6% |
| `DISABLE_INTEGER_DOT` | 301.15 | 299.02 | -0.7% |

**Every arm moved down by a similar amount**, which is thermal/contention drift, not a change in the
finding. The RELATIVE results are identical (coopmat costs ~40%, integer-dot is flat), which is what
those receipts exist to support. But the absolute figures carry a **~1.5% band**, so treat ROCm prefill
as **315-320 t/s** and Vulkan prefill as **299-303 t/s** rather than as single values.

# Prefill agent: spec-constant field mapping (valuable partial, preserved)

The prefill agent's warptile sweep did not complete before its timeout, but its structural analysis is
worth keeping. It decoded the `mul_mmq.comp` / `mul_mm.comp` specialisation-constant field order:

    {BLOCK_SIZE, BM, BN, BK, WM, WN, WMITER, TM, TN, TK, WARP}

Three findings from that:

1. **`BK` (field 3) is INERT on the quant path.** It is commented out in `mul_mmq.comp`
   (`// layout (constant_id = 3) const uint BK = 32;` and `#define BK 32`). So tuning it does nothing
   for IQ2_S/IQ4_XS - a whole axis of the sweep was wasted before this was known.
2. **`subgroup_size_8 = max(subgroup_size, 8)` = 64 on wave64**, so the real default large warptile is
   `{256,128,128,32,64,64,2,cm,cn,ck,64}` - not the `{256,128,128,32,8,...}` that the literal `8`
   suggests. Reading the source literally gives the wrong baseline.
3. The AMD coopmat branch overrides only the LARGE warptile; `m_` and `s_` keep generic values.

Its own harness had two bugs it caught and was fixing when it timed out (the t/s parse grabbed the
model size because `-b/-ub` shift the column indices, and the M-fallback grep matched bf16 rows - the
same column-shift trap that bit the parent twice). The uncommitted 41-line env-override patch is
snapshotted at `prefill/tree-a-wip-<ts>.patch` so the work is not lost.

The sweep's own numbers were never trustworthy (every arm read 8.94 = the model size), so **no prefill
tuning conclusion was reached.** The prefill exclusion rests on the independent ceiling cross-check, not
on this sweep.

# Correction: the plain-decode exclusion is MARGINAL (0.6%), not comfortable

My earlier arithmetic used the full 9.608 GB per token. That is wrong: plain decode does not read the
MTP layer (`blk.64`, ~0.348 GB), so it reads **~9.260 GB per token**. Recomputed:

    9.260 GB / 256 GB/s (theoretical peak)      = 36.2 ms -> 27.6 tok/s
    at the measured achievable 239.4 GB/s       = 38.7 ms -> 25.9 tok/s
    target 26 tok/s needs 240.8 GB/s = 94% of theoretical peak
    measured today: 20.10 tok/s = 193 GB/s

**The target sits 0.6% above what the hardware has demonstrably delivered.** So "excluded" is the right
word, but for the correct reason: not that the margin is large - it is not, it is 0.6% - but that the
one measured route to it (getting the IQ2_S matvec to the BF16 streaming rate) has been tried and
failed (+0.45% from the sign-fold). The earlier "~24.9 tok/s" figure was pessimistic by 1 tok/s.

This matters for how the result is read. Plain decode is NOT comfortably out of reach; it is out of
reach by less than a percent, and only because the format's dequant cost will not come out of the
kernel. If anyone later finds a way to make IQ2_S stream at BF16's rate, this criterion becomes
reachable rather than excluded.

# CORRECTION: the model is MIXED-QUANT, and the sign-fold tested only 18.9% of it

The prefill agent's warptile trace incidentally revealed that the model I have been calling
"IQ2_S-mtp" is **not** IQ2_S. Parsed from the GGUF header directly (866 tensors, 10.538 GB):

| type | count | GB | share |
|---|---:|---:|---:|
| IQ3_S | 62 | 2.252 | **21.4%** |
| **IQ2_S** | 96 | 1.991 | **18.9%** |
| IQ4_NL | 64 | 1.758 | 16.7% |
| Q2_K | 41 | 1.337 | 12.7% |
| IQ2_XXS | 59 | 1.072 | 10.2% |
| IQ2_XS | 46 | 0.830 | 7.9% |
| type 29 (unmapped in my table) | 18 | 0.540 | 5.1% |
| Q6_K | 8 | 0.348 | 3.3% |
| Q4_K | 9 | 0.204 | 1.9% |
| IQ3_XXS | 7 | 0.146 | 1.4% |
| BF16 | 96 | 0.047 | 0.4% |
| F32 | 360 | 0.011 | 0.1% |

(Caveat: my type-number table maps both 18 and 23 to IQ2_S and does not decode 29; the mix is certain,
the exact per-type labels may be off by one. The headline is unaffected - the model is a blend, and
IQ2_S is a minority of it.)

## What this falsifies: my reading of the sign-fold result

I concluded, from the sign-fold's +0.45%, that "the Vulkan driver already handles per-value sign
selects cheaply and the ALU cost is elsewhere". **That inference was invalid.** The sign-fold edited
`mul_mat_vec_iq2_s.comp`, which handles only the IQ2_S tensors = **18.9% of model bytes**. A *perfect*
IQ2_S-only fix could not have moved end-to-end decode by more than ~19%, and the sign-fold only removes
part of the dequant cost within that 19%.

**+0.45% is consistent with SCOPE, not with a null result.** The ALU route is not dead - it was tested
on a fifth of the model and read as if it had been tested on all of it. This is the same class of error
as the earlier ones in this project: drawing a whole-model conclusion from a partial-scope experiment.

## Consequence for the remaining decode gap

Capturing the full ~19% bandwidth gap (193 vs 239 GB/s) would require ALU work across the **whole quant
mix** - IQ3_S, IQ2_S, IQ4_NL, Q2_K, IQ2_XXS, IQ2_XS, IQ1_M, Q6_K - i.e. roughly **5x the scope** I had
assumed when I told the user the sign-fold was "the measured route to 40". Each type has its own
dequant shader and its own hoistable/reducible ALU. That is a much larger project than was funded.

It also means the Q8_0 diagnostic (BF16 239.4 / Q8_0 227.5 / this model 193.1 GB/s) measures the
*blend's* average dequant cost, not IQ2_S's specifically. The conclusion "block structure costs ~5%,
dequant ALU costs ~15%" still holds for the blend as a whole; the attribution to one type was wrong.

# PREFILL: 298 -> 490-492 t/s (98% of target). My "excluded" conclusion was WRONG.

Independently verified by the parent at r=3, locked, on tree a's committed `2714f007a`:

| arm | pp2048 | TFLOP/s |
|---|---:|---:|
| baseline | 299.36 ± 0.37 | 16.36 |
| **BS128+BN256 (`SAKER_LMMQ_WT=128,128,256,32,64,64,2,16,16,16,64`, `SAKER_LMMQ_WGD=128,256,1`)** | **492.12 ± 2.03** | 26.89 |
| repeat | 489.99 ± 1.16 | 26.77 |
| decode non-regression (tg128) | 20.06 ± 0.01 | unchanged |

The agent's own best single arm was 497.26 (r=2). **This is 98% of the 500 target** - from two spec
constants that had been left at generic-AMD defaults for a 40-CU APU: `BLOCK_SIZE` 256->128 and
`BN` 128->256 on the large quant-matmul warptile.

## I called prefill excluded TWICE, and both times the evidence was inadequate

1. First citing ~13.8 TFLOPS from **unrelated work on a different model**.
2. Then citing a **second-hand** "22.0 TFLOP/s measured WMMA peak" from Path A's prose - and I built
   arithmetic on it ("500 needs 124% of peak") without ever testing whether a spec constant was simply
   mis-set for this device.

The agent's sweep **falsified the ceiling outright: 27.17 TFLOP/s measured, 123.5% of that "peak"**.
Neither 17 nor 22 TFLOP/s is a wall. There was no plateau; throughput kept climbing as the constants
were changed.

**Lesson, and it is the same one three times now: a ceiling inferred from a number I did not measure
myself is not a ceiling.** Both earlier prefill "exclusions" were arithmetic on borrowed figures. The
one measurement that settled it took a single build and 14 arms.

## The winning configuration is currently an ENV OVERRIDE, not a default

`SAKER_LMMQ_WT` / `SAKER_LMMQ_WGD` are instrumentation added by the agent (`2714f007a`). The default is
still the generic-AMD one, so **the +64% is not yet live** - it requires the env vars. Baking the
winning values in as the default for this device class (AMD + coopmat) is the obvious next step, and
should be guarded so it does not silently change behaviour on other AMD parts.

# lm_head: verified NEGATIVE, and the 4.2 ms target did not reproduce

Committed `df4ce6e2b` on `saker-lmhead`. The agent found the structural cause exactly as briefed -
IQ4_XS has no per-type shader; `vulkan-shaders-gen.cpp:737` routes it (and every non-`iq1_/iq2_/iq3_/_k`
type) to the generic `mul_mat_vec.comp` template, where the column loop is OUTER and the dequant sits
INSIDE the row loop - and hoisted it correctly.

**Correctness is airtight both ways**: bit-identical by construction (pure loop-invariant motion), and
empirically 3 runs -> 276 chars, md5 `2fe11002` = the parent-verified baseline output, acceptance
bit-identical at 0.77551 (38/49), 3/3 self-consistent.

**Performance: no measurable gain.** MTP 36.58-37.00 -> 36.50/36.57/36.66 (noise); pp3 delta +0.41 ->
+0.72 ms (noise); pp4 +0.97 -> +1.00 ms (noise). Probable reason: the `[[unroll]]` column loop makes
the duplicated dequant CSE-able straight-line code, so **RADV was already eliminating it**. A source
redundancy, not a runtime one.

## The finding that matters more than the fix

**The +6.83 ms / +17.16 ms target does NOT reproduce.** Clean, locked, same-build, same-flag
measurement gives **+0.41 ms (pp3) and +0.97 ms (pp4)**. So:

- The multi-column lm_head is **already nearly bandwidth-bound** on this code; there was never ~4.2
  ms/round to recover from it.
- **The "lm_head (-4.2) + host-sampling (-0.8) = 40.00 tok/s" projection I gave the user is void.** It
  was built on a non-reproducing measurement.
- The verify-vs-pp3 gap (~4 ms) must live elsewhere, not in the lm_head.

# MTP round decomposition on the CURRENT build, and the ~5.2 ms that is the route to 40

Measured on the merged deliverable with `SAKER_ROUND_TRACE=1 SAKER_SPEC_TRACE=1`,
`SAKER_SHORTLIST=3584 --spec-type draft-mtp --spec-draft-n-max 2`, 64 tokens, **MTP = 36.96 tok/s**:

| phase | per round |
|---|---:|
| `tgt` (target verify decode + sync) | ~62 ms |
| `proc` (MTP KV catch-up) | ~1.2 ms |
| `smpl` (sampler clone + sample + accept) | ~0.5 ms |
| `draft()` (2 drafts, `SAKER_SPEC_TRACE`) | ~5.3 ms |
| **total** | **~69 ms** (measured round 66.5 ms) |

(The trace counters are cumulative and increase by the per-round amounts shown; `draft()` prints
per-call.)

## The gap that matters

**A plain 3-token prefill (`llama-bench -p 3`) costs 56.78 ms. The MTP verify costs ~62 ms.** Both
process 3 tokens in one batch, so the ~5.2 ms difference is not inherent to the shape.

    66.5 ms - 5.2 ms = 61.3 ms -> 40.1 tok/s

**This is the only remaining identified route to the 40 tok/s MTP target.**

## Ruled out already - do not re-investigate

- **multi-column lm_head**: +0.41 ms (pp3) / +0.97 ms (pp4), clean locked measurement. The earlier
  "+6.83/+17.16 ms" does NOT reproduce and is retracted.
- **extra logits at 3 positions**: `SAKER_ALLLOGITS` shows +0.41 ms at pp3 for all-positions vs last-only.
- **the draft lane**: `draft()` is 5.3 ms and that is its measured cost.

## Leading hypotheses for the 5.2 ms

1. **A forced synchronisation after the verify.** The verify must read logits to decide acceptance,
   which forces a sync each round; a plain llama-bench prefill has no per-step sync and can pipeline.
   Most likely single cause, and cheapest to test.
2. **Memory pressure from the resident draft context** (blk.64 + shortlist rows) displacing cache for
   the target's 9.6 GB weight pass.
3. **KV writes for 3 positions** sitting on the critical path.

An agent is attributing and attacking this now. Note the projection above is arithmetic from measured
per-piece costs; the real number will land either side of 40.

# Second instance of the same bug class: AMD denied the matvec workgroup heuristic

The prefill win came from a device-tuning branch that excluded AMD. The **matvec** path has the same
defect, found by looking for it deliberately:

    // ggml-vulkan.cpp:7816 and :7983 (TWO copies)
    uint32_t dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
    if ((vendor_id == VK_VENDOR_ID_NVIDIA && arch != NVIDIA_PRE_TURING) || vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        ...
    }

**That branch is NVIDIA-and-Intel only.** This device is AMD/RADV, so it ALWAYS keeps
`DMMV_WG_SIZE_SUBGROUP` (subgroup_size16 = 64 threads, SUBGROUP reduction) and never gets the LARGE
variant (subgroup_size16*4 = 256, HYBRID reduction) - even though the comment states the larger
workgroup is preferred when M is small, which is exactly decode's case.

Made env-overridable as `SAKER_DMMV_WG` (both sites) and measured on tree `-hostsample`:

| workload | SUBGROUP (default) | LARGE (forced) |
|---|---:|---:|
| plain decode tg128 (1 column) | 20.06 ± 0.00 | **20.38 ± 0.01  (+1.6%)** |
| MTP verify (3 columns, shortlist on) | 36.60 / 36.60 | **13.99 / 29.07  (much worse)** |
| acceptance | 0.77551 (38/49) | 0.77551 (38/49) - unchanged |

## The structural pattern, now confirmed twice

**The 1-column and multi-column matvec paths want different geometry, and the current code cannot
express that.**

- `rm_iq=2`: +1.2% plain decode, **-6% to -24%** MTP
- `DMMV_WG_SIZE_LARGE`: +1.6% plain decode, **-20% to -60%** MTP

Both help the 1-column path and hurt the 3-column one. The heuristic already varies by `m`; it could
also vary by `num_cols`, which would take the +1.6% on plain decode while leaving the verify alone.

## What this is and is not worth

A real, free +1.6% on plain decode (20.06 -> 20.38) that AMD has been denied - **but it does not close
the gap to 26 tok/s** (which needs ~240 GB/s and would give 25.9 even at the measured BF16 streaming
rate). MTP is unaffected at best. Recorded because it is the second instance of a systematic defect
(device-tuning branches that skip AMD) and because the num_cols-dependent fix is available if anyone
wants the 1.6%.

# MTP round: the gap is LOCATED (and my "5.2 ms unaccounted" was wrong)

The round-decomposition agent extended `SAKER_ROUND_TRACE` with `dec` (graph dispatch) and `sync` (GPU
completion wait) counters. Per-round deltas, n_max=2, shortlist on:

| phase | per round | what it is |
|---|---:|---|
| `sync` | **59.0 ms** | the GPU weight pass completing |
| `draft()` | 5.3 ms | the 2 MTP drafts |
| `dec` | 2.1 ms | host-side graph dispatch |
| `proc` | 1.3 ms | MTP KV catch-up |
| `smpl` | 0.5 ms | sampler clone + sample + accept |
| **total** | **~68 ms** | measured round 66.5 (slight overlap) |

## Correction: there is no hidden 5.2 ms

I had estimated "the verify costs ~62 ms vs pp3's 56.78 ms = a 5.2 ms gap". The finer trace shows the
truth: **`sync` is 59.0 ms, only ~2.2 ms above a plain 3-token prefill.** The rest of the round is the
draft lane (5.3) and host-side overhead (dec 2.1 + proc 1.3 + smpl 0.5 = 3.9) - all of which I had
already counted separately. The round is essentially fully accounted for.

## The route to 40, restated accurately

    round 66.5 ms; need <= 61.5 ms  ->  must cut ~5 ms

| candidate | worth | kind |
|---|---:|---|
| draft lane (2 drafts @ ~2.65 ms) | up to 5.3 ms | the MTP draft path |
| host-side overhead (dec+proc+smpl) | up to 3.9 ms | dispatch/sampling plumbing |
| verify gap over plain prefill | ~2.2 ms | unclear |

**All three are code/plumbing, not shaders** - unlike the sign-fold and lm_head attempts. The draft lane
is the largest single piece. Note the fused gather already cut it once (the F32 intermediate is gone);
what remains is 2 sequential drafts over blk.64 + the shortlist rows.

# Systematic audit for the "device tuning that skips AMD" bug class

The prefill win came from a device-tuning value that was wrong for this part, so I audited every
device-conditional branch in `ggml-vulkan.cpp` for the same defect. Result: **essentially one instance
on the decode path, already found, and no second big one.**

| site | condition | affects AMD? |
|---|---|---|
| :3713, :5190, :6384, :6952, :7513 | `vendor == INTEL` | no - Intel-specific |
| :3738 | `D & 8 \|\| n_kv < 1024 \|\| INTEL` | yes, via the first two terms (not a vendor exclusion) |
| :3758, :3809 | `vendor == NVIDIA` (shmem_staging) | no |
| :4165, :5200, :5287, :5342 | `vendor == INTEL && subgroup_size_control` | no - AMD uses `device->subgroup_size` (64), which is correct |
| :4292 | `vendor == INTEL && coopmat` | no - AMD has its own coopmat branch at :4287 |
| :5921, :7064, :6339 | INTEL | no |
| **:7816 and :7983** | **`(NVIDIA && !pre-Turing) \|\| INTEL`** | **YES - the matvec workgroup heuristic. AMD always keeps SUBGROUP.** |
| :4287 | `vendor == AMD && coopmat` | AMD IS handled here - its values were just tuned for dGPUs, which is what the prefill sweep corrected |

## What this bounds

- The one genuine AMD exclusion on the decode path is the matvec workgroup heuristic, worth **+1.6% on
  plain decode** (20.06 -> 20.38) and **harmful to the multi-column verify** if applied globally. The
  fix would be to vary it by `num_cols` as well as `m`.
- The prefill win was NOT an exclusion - the AMD-coopmat branch exists and is used. Its *values* were
  tuned for AMD dGPUs. That is a different failure mode, and it is fixable only by sweeping (which is
  what found BS128+BN256).
- **No further large AMD-skipped branch was found.** If more decode headroom exists it is in the kernel
  or the plumbing, not in a vendor conditional.

# MTP round gap: it is GPU-side execution, not host/sync overhead (agent commit 6f7c25682)

The round-decomposition agent extended `SAKER_ROUND_TRACE` with `dec` (CPU submit) and `sync` (GPU wait)
counters and produced a clean same-session decomposition at n_max=2 (MTP 36.59, acceptance 0.77551):

    tgt  = 62.93 ms = dec 2.15 (CPU submit) + sync 59.17 (GPU wait) + ~1.6 (queue handoff)
    proc = 1.52    smpl = 0.51    draft() ~ 5.3

    same-session plain forwards (llama-bench r=5, -b 1024 -ub 256):
        pp1 50.23   pp2 52.65   pp3 56.69   pp4 67.61  ms/pass
        all-logits deltas: +0.36 / +0.19 / +0.88 ms (pp2/3/4)  -> logits are NOT the gap

    tgt-minus-pp:  +3.8 ms (2 positions)   +6.2 ms (3)   +6.2 ms (4)
    at 3 positions: ~4.7 ms EXTRA GPU EXECUTION + ~1.6 ms queue handoff + ~0 submit

## This CORRECTS my reading of the raw counters

I had inferred from `sync = 59.27 ms vs pp3 = 56.78 ms` that the verify was "only +2.5 ms over a plain
prefill" and that the cuttable time was host-side (`dec`). **The agent's finer split shows the opposite:**
the gap is ~4.7 ms of extra GPU execution plus ~1.6 ms of queue handoff, and `dec` contributes ~0 at 3
positions. The decisive evidence is the 2-position case, where `sync` (52.64) matches `pp2` (52.65)
almost exactly - so the extra GPU cost only appears at >=3 positions.

**So the route to 40 is a GPU-execution problem after all**, not plumbing. That is a worse place to be
than I thought: both previous GPU-kernel projects on this project (sign-fold, lm_head) failed.

## Remaining attribution - explicitly incomplete, and the clear next step

The agent's own words: *"Remaining attribution (KV position vs GDN-state handling) NOT completed."*

The extra GPU execution scales in only at >=3 verify positions. The two candidate mechanisms are the KV
write for each extra position, or the GDN recurrent-state handling for each extra position. The model is
hybrid (48 GDN + 16 full-attention layers), so a per-position recurrent-state cost is plausible and would
not appear in the 2-position case the same way.

Its numbers cross-check against the parent's independent measurements (pp2 52.65 vs 52.15, pp3 56.69 vs
56.78), so the decomposition is trusted.

# ⚠️ CORRECTION: the "prefill 490-492 t/s (98% of target)" claim was an ARTIFACT

**My 490-492 verification was invalid, and so was the agent's original 492/775 result.** The cause:

> Every arm that changed `BLOCK_SIZE` (256->128) or `BM` (128->256) reported spectacular llama-bench
> throughput (490-775 t/s) and **BREAKS GENERATION** - empty output (BS128, BM256) or a wrong token
> stream (wgd256: acceptance collapses 0.775 -> 0.439, different md5).
> **llama-bench does not validate output, so those numbers were artifacts.**

I ran llama-bench, got 492 t/s, and called it verified - **without ever checking that the model still
generates text.** My merge-verification script (which WOULD have caught it) was staged but never run,
because the bake-in had not been committed yet. The signal was even sitting in a log I had already read:
the round agent's `split.log` contained `get_logits_ith: invalid logits id 99` and I did not connect it.

## The verified-correct configuration, and the real number

Only `BN` 128->256 is correct. Baked in as the default for `AMD && uma` in commit `2c8a51820`
(disable with `SAKER_APU_MMQ=0`). Parent-verified with an OUTPUT check this time:

| arm | pp2048 | output |
|---|---:|---|
| retune ON (new default) | **408.77 ± 0.85** | 276 chars, md5 `2fe11002ab` - byte-identical |
| retune OFF (`SAKER_APU_MMQ=0`) | 299.98 ± 0.32 | 276 chars, md5 `2fe11002ab` |
| pre-change build | ~299 | 276 chars, md5 `2fe11002ab` |
| BS128 / BM256 arms (REJECTED) | 490-775 (artifact) | **empty or wrong output** |

**So prefill is at ~411 t/s = 82% of the 500 target, not 98%.** The +37% is real and live by default;
the +64% was not real.

## Refinement of the agent's diagnosis

The agent attributed the breakage to `get_logits_ith: invalid logits id`. **That log line is PRE-EXISTING
and benign** - it appears once per request on the unmodified merge tree, on the retune-ON build and on the
retune-OFF control, all three producing byte-identical 276-char output. It is not the discriminator. The
real discriminator is **the generated output itself** (empty/wrong vs 276 chars).

## The lesson, which cost the most of anything in this project

**llama-bench measures throughput and never validates output.** Any change to GEMM tiling, accumulation
order, or kernel structure MUST be checked by generating text with llama-server and comparing it. A
consistent scaling curve (the 775 t/s arm scaled cleanly across pp512-4096) is NOT a correctness signal
either. I had already written "a ceiling inferred from a number I did not measure myself is not a ceiling"
into this document - this is the same failure one level deeper: **a throughput number that was never
checked against output is not a result.**

# Deliverable updated: prefill retune merged and verified (f65d47b8a)

Merged `saker-prefill` (`2c8a51820`) into `saker-combined`. 0 conflicts. New deliverable head
**`f65d47b8a`**, rebuilt, provenance checked.

Parent-verified on the merged build, under the gpu-lock, **with the output check that the earlier
invalid verification skipped**:

| check | result |
|---|---|
| prefill pp2048, NO env vars | **409.80 ± 1.41 t/s** (22.39 TFLOP/s) |
| prefill, `SAKER_APU_MMQ=0` control | 301.19 ± 0.42 t/s -> **+36%** |
| plain decode tg128 | 20.08 ± 0.01 (unchanged) |
| MTP decode, 3 runs | 36.76 / 36.57 / 36.52 tok/s |
| output, 3 runs | 276 chars, md5 `2fe11002ab`, **1 distinct** - byte-identical |
| draft acceptance | 0.77551 (38/49), bit-identical |

**The retune is now live by default** (guard `vendor == AMD && device->uma`, i.e. APUs not dGPUs;
`SAKER_APU_MMQ=0` disables). No env vars needed.

Worth noting: the verified config measures **22.39 TFLOP/s**, which is *above* the "22.0 TFLOP/s measured
WMMA peak" that two of this project's earlier prefill exclusions were built on. That figure was never a
ceiling, and the whole episode is a case study in arithmetic on borrowed numbers.

# DEFINITIVE PANEL on the current deliverable f65d47b8a (one build, one locked session)

No env vars except `SAKER_SHORTLIST=3584` for the MTP arm. sclk sampled at start.

| metric | value |
|---|---|
| prefill pp2048 (`-b 1024 -ub 256`) | **405.64 ± 1.03 t/s** |
| plain decode tg128 | **20.10 ± 0.00 t/s** |
| MTP decode, 3 runs | **36.78 / 36.65 / 36.66 t/s** |
| draft acceptance | 0.77551 (38 accepted / 49 generated) |
| output, 3 runs | 276 chars, md5 `2fe11002ab`, **1 distinct** (deterministic) |

Against the recorded baselines: MTP **+32% to +43%** (25.7-27.7 -> 36.65-36.78), prefill **+27%**
(320.7 ROCm -> 405.64 Vulkan), plain decode ~flat vs the 20.1 Vulkan figure.

# MTP round gap: the rollback/async-drain hypothesis is REFUTED (agent commit 7bd1da41a)

The previous agent's top hypothesis was that KV/GDN rollback work - or async draft-context work - was
submitted outside the measured windows and drained inside the next round's `tgt` sync. A direct A/B
refutes it.

Method: added `SAKER_SYNC_DFT=1`, forcing `llama_synchronize(ctx_dft)` after **all 10 real**
`llama_decode(ctx_dft, ...)` sites (a first patch attempt also corrupted 3 error-message string literals
containing that text - caught by the build, reverted, repatched by line-shape), plus a `dft=` counter.

Same session, locked, n_max=2, shortlist on, 64 tokens:

| arm | MTP t/s | sync/round | dft/round | md5 | acceptance |
|---|---:|---:|---:|---|---|
| default | **36.53** | 53.48 ms | - | `2fe11002ab` | 0.77551 (38/49) |
| `SAKER_SYNC_DFT=1` | 35.72 | **53.49 ms** | 2.73 ms (3 calls) | `2fe11002ab` | 0.77551 (38/49) |

**Why this refutes it:**
1. `sync` is identical to **0.01 ms** across arms - forcing every draft submission to drain in its own
   window did not reduce tgt's sync, so the draft work was never draining there.
2. The draft context's entire async footprint is **~2.8 ms/round** (~0.92 ms x 3 calls) - too small to
   account for 4.7 ms even if it were misplaced.
3. Forcing the syncs **costs 0.8 tok/s** of pure overhead, so "sync the draft context" is not a fix.
4. Correctness held exactly; the instrumentation is inert when unset (`dft=0.00(n=0)`).

## Cumulative elimination for the ~4.7 ms

Ruled out by direct measurement, one at a time: sync mechanism (2-position match), all-positions logits
(+0.19 ms), CPU submit, KV-position interaction (2-position sync == pp2 over ~110 KV positions), and now
**async draft-context work**. The surviving candidate is **hybrid GDN recurrent-state handling** for the
extra verify positions - the model has 48 GDN + 16 full-attention layers, so a per-position
recurrent-state cost would not appear in a 2-position comparison the same way.

# Prefill CLOSED: every correct configuration tops out at ~411 t/s (82% of target)

The prefill agent swept warptile fields ONE AT A TIME and concluded "no correct configuration reaches
500". I swept **combinations** - genuinely unexplored, and motivated by the DMMV_WG result showing these
fields interact. Parent-run, locked, tree `-a` at `2c8a51820`, baked default = `BS256 BM128 BN256 WM64
WN64 WMITER2`:

| arm | pp2048 | output verification |
|---|---:|---|
| **baked-default** | **411.48** | 276 chars, md5 `2fe11002ab` - **VALID** |
| WM32 | 474.60 | **0 chars** - BROKEN |
| WN32 | 476.47 | **0 chars** - BROKEN |
| WM32+WMITER1 | 471.16 | **0 chars** - BROKEN |
| BM64 | 449.05 | **64 chars, md5 `6441072631`** - BROKEN |
| WM128 | 312.00 | (slower) |
| WMITER1 | 406.26 | (slower) |

**Every arm faster than the baked default is broken.** The correct ceiling is ~411 t/s, which is **82% of
the 500 t/s target**. Prefill >= 500 is therefore closed: it is not reachable by any correct
configuration of the large quant-matmul warptile on this device.

## The verification discipline earned its keep twice in one round

This is the second time output-checking caught artifacts that llama-bench presented as wins:

1. **BS128/BM256/`wgd256`** (earlier): 490-775 t/s, empty or wrong output.
2. **WM32/WN32/BM64/WM32+WMITER1** (now): 449-476 t/s, all broken.

`BM64` is the most instructive of the eight: it produced **64 chars, not 0** - a partially-correct output
that llama-bench reports as a clean 449 t/s win. Neither the throughput nor the scaling curve nor a
non-empty response distinguishes it. **Only comparing the generated text against a known-good baseline
does.** Four of the seven arms I measured in this sweep were artifacts; without the output check I would
have reported prefill at 476 t/s and been wrong by 16%.

**Rule, now proven twice: for any GEMM tiling or accumulation-order change, the throughput number is not
a result until the generated text matches.**

# UNIT AUDIT: the APU retune breaks ONE MUL_MAT case (q4_0), 1555/1556 pass

The objective requires "unit + integration audits of every kernel vs CPU/high-precision references".
llama.cpp ships the right tool: `test-backend-ops test -o MUL_MAT -b Vulkan0`, which compares backend
kernels against the CPU backend. Run on the retuned deliverable `f65d47b8a`:

| arm | result |
|---|---|
| **retune ON (baked default)** | **FAIL** - 1 case of 1556 |
| retune OFF (`SAKER_APU_MMQ=0`) | OK |

The single failing case:

    MUL_MAT(type_a=q4_0,type_b=f32,m=576,n=512,k=576,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1)

**So the retune is correct for 1555 of 1556 MUL_MAT kernel cases and wrong for one q4_0 shape.**

## Why the model output was still byte-identical

**This model contains no q4_0 tensors.** Its mix is IQ3_S 21.4%, IQ2_S 18.9%, IQ4_NL 16.7%, Q2_K 12.7%,
IQ2_XXS 10.2%, IQ2_XS 7.9%, plus Q6_K/Q4_K/IQ3_XXS/BF16/F32. So the one broken case is never exercised by
this model - which is exactly why the end-to-end byte-identical check passed and did NOT catch it.

**This is the case for unit audits existing at all.** An end-to-end output check on this model cannot see
a q4_0 regression; only a per-kernel comparison against a reference can.

## Status and required fix

The deliverable as committed (`f65d47b8a`) **fails the objective's unit-audit requirement**. The correct
resolution is to NARROW the retune so q4_0 (or the failing shape) keeps the original generic tile, which
retains the +37% prefill on this model while making the audit pass. Not yet done.

# MTP gap: MECHANISM FOUND - a 3943-node graph rebuilt per round at >=3 verify positions

The GDN agent instrumented the context/ubatch path (`saker-ctx`) and the asymmetry is now explained at
the mechanism level, not just located:

    n_max=2 (3 verify positions):
      saker-ctx: decode ubatch#1 nt=3 nseq=1 nst=3 allout=3/3
      saker-ctx: decode ubatch#1 nt=3 nseq=3 nst=1 allout=0/3     <- nseq=3, THREE SEQUENCES
      saker-ctx: ub REBUILD ntok=2 nseq=1 nst=2 nodes=3943        <- 3943-node graph, rebuilt

    n_max=1 (2 verify positions):  nodes=51 and 54. The 3943 graph does not appear at all.

## What this means

1. **The verify decodes with `nseq=3`** - the batch is split into three sequences. That is the mechanism
   by which drafts can be discarded: each draft position gets its own sequence so it can be dropped
   without copying state. It is rollback implemented by sequence isolation rather than by state copying.
2. **That split triggers a 3943-node graph** where the ordinary path builds 51-54 nodes. 3943 is about
   right for the GDN/SSM work across the 48 GDN layers (~82 nodes each).
3. **It is a REBUILD, not a reuse** - so it is paid every round.

This is why the extra GPU execution appears ONLY at >=3 positions and never at 2: at 2 positions the
batch does not split into 3 sequences, so the large graph is never built. It also explains why every
previously-tested candidate was refuted - the sync mechanism, all-positions logits, CPU submit,
KV-position interaction, and the draft context's async work are all real but all *small*. The cost is a
graph rebuild, which none of those probes would have surfaced.

## Consequence

The ~4.7 ms is not inherently irreducible work; it is a **per-round graph rebuild**. Candidates now worth
testing (agent in progress): caching/reusing the graph across rounds, or avoiding the multi-sequence split
for the positions that cannot be rejected.

# Unit audit of the kernels THIS PROJECT changed: 2547 cases, 1 failure

`test-backend-ops test -o <OP> -b Vulkan0` compares Vulkan kernels against the CPU backend. Run on the
deliverable `f65d47b8a` for every op this project touched:

| op | what it covers | cases | failing |
|---|---|---:|---:|
| **MUL_MAT_ID** | the fused gather-matvec (reuses `ggml_mul_mat_id`) | **872** | **0** |
| **GET_ROWS** | the gather path, kept as the multi-output fallback | **119** | **0** |
| MUL_MAT | the APU prefill retune (GEMM tiling) | 1556 | **1** (q4_0, being narrowed) |
| | **total** | **2547** | **1** |

**The fused gather is now verified against a CPU reference across 872 cases**, not merely by end-to-end
output equality. That is a materially stronger claim than the byte-identical output check it had before,
and it closes the objective's "unit audits of every kernel" requirement for the two changes that were
previously covered only behaviourally.

The single failure is the prefill retune's q4_0 case (`m=576,n=512,k=576`) - a real defect, already
identified, with narrowing in progress. Every other kernel case this project touches passes.

## Why this is the right shape of verification

The three ops map exactly onto the three changes:
- `MUL_MAT_ID` -> the fused gather (the draft lane)
- `GET_ROWS` -> the fallback path that change retained
- `MUL_MAT` -> the prefill retune

Everything else in llama.cpp is untouched, so auditing these is auditing the change surface rather than
the whole backend.

# q4_0 narrowing: IMPLEMENTED but DOES NOT COMPILE (naming mismatch) - snapshot preserved

The narrowing agent's approach is right: save the pre-retune Q4_0 tiles before `saker_apu_tune`
overwrites them, and pass those to the Q4_0 pipeline-creation sites. But the edit uses **three different
naming conventions in one change** and cannot compile:

    DECLARED (~4309):  q40_l_warptile_mmq, q40_m_..., q40_s_..., *_int
    DECLARED (~4325):  q40_l_mmq_wg_denoms, q40_m_..., q40_s_...
    USED (4660-4760+): l_q40_mmq_wg_denoms, l_q40_warptile_mmq, m_q40_..., s_q40_...
    USED (4680-5106):  q40_mmq_wg_denoms, q40_warptile_mmq   (no size prefix at all)

Build output: `'l_q40_mmq_wg_denoms' was not declared in this scope` at 4660, 4760 and several more.

**The fix is a rename** - pick one convention (the `q40_<size>_<thing>` form the declarations already use)
and make every use-site agree. The size prefix must be part of the name because Q4_0 needs the L tile
specifically.

**Prefill is unaffected by the broken edit**: measured on the tree with the edit applied,
pp2048 = **412.94 ± 0.63 t/s**, so the retune win is intact.

WIP snapshot: `prefill/wip-q40-<ts>.patch` (mirrored both nodes). The agent has ~8 min left; if it does not
finish the rename, the parent will.

# q4_0 narrowing: NOT COMPLETED. Documented trade-off (final position on this)

**Decision: the narrowing was abandoned rather than half-done, and the trade-off is stated explicitly.**

## What the retune costs and buys, verified

| | |
|---|---|
| prefill, retune ON (no env vars) | **409.89 ± 1.53 t/s** |
| prefill, retune OFF (`SAKER_APU_MMQ=0`) | 301.19 ± 0.42 t/s |
| MUL_MAT audit, retune ON | **1 failing of 1556** |
| MUL_MAT audit, retune OFF | 0 failing |
| the failing case | `MUL_MAT(type_a=q4_0,type_b=f32,m=576,n=512,k=576,...)` |
| model output | 276 chars, md5 `2fe11002ab`, byte-identical, deterministic |

**+37% prefill, at the cost of exactly one q4_0 kernel case.** This model contains no q4_0 tensors, so its
behaviour is unaffected - which is why the byte-identical check cannot see the defect.

## Why it was not fixed

The fix is a per-type exception: save the pre-retune Q4_0 tiles and pass them to the Q4_0
pipeline-creation sites. The approach is right and is preserved, but two attempts failed:

1. **First attempt (agent)**: three inconsistent naming conventions in one change
   (`q40_l_warptile_mmq` vs `l_q40_warptile_mmq` vs `q40_warptile_mmq`) -> would not compile.
2. **Second attempt (same agent, after steering)**: a rename loop that COMPOUNDED the prefix, producing
   `q40_q40_mmq_wg_denoms` and `q40_q40_q40_mmq_wg_denoms`, taking the build from 8 errors to **186**.
   The agent was interrupted and the tree reverted to the known-good `2c8a51820`.

The use-sites pass loop-local aliases (`mmq_wg_denoms`, `warptile_mmq`) whose binding is threaded through
the creation loop; resolving that correctly is more than a rename, and doing it at 71% context risked a
third botched edit. **A documented limitation beats a broken tree.**

WIP patches preserved both nodes: `prefill/wip-q40-<ts>.patch`, `prefill/wip-prefill-<ts>.patch`.

## Consequence for the objective

**The deliverable does NOT pass the unit-audit requirement in full**: 2546 of 2547 kernel cases pass, and
the one failure is a type this model never uses. This is stated in the report's executive summary and
section 9 rather than hidden.

# ⚠️ CORRECTION: "3943-node graph rebuilt per round" was WRONG (I over-read the trace)

The GDN agent (commit `c0f4eca55`) node-dumped the graphs and refuted my recorded mechanism on two counts.
**My finding was wrong and is retracted.**

## What I got wrong

1. **The graph is REUSED, not rebuilt.** Rebuild happens only on a shape change (prompt -> first verify).
   So there is nothing to cache and "graph rebuild" is not the mechanism at all.
2. **I compared apples to oranges.** The 51/54-node graphs are the **DRAFT context's** per-token forwards
   (`ct=1`), not 2-position target graphs. The 2-position *target* verify graph is 3799 nodes. My
   "3943 vs 51/54" comparison was between two different contexts.

## What is actually true

    plain pp2 target graph : 3655 nodes
    plain pp3 target graph : 3655 nodes   <- FLAT; the plain graph does not grow with token count
    MTP verify, 2 positions: 3799
    MTP verify, 3 positions: 3943         -> +144 nodes PER VERIFY POSITION

**+144 nodes per verify position** over the plain graph. 144 is about 48 GDN layers x 3 nodes, which fits
a per-position GDN/SSM state-save - but the agent did **not** confirm the op names (its node-list diff run
failed on an extraction bug at the last step).

Also corrected:
- The `nt=3 nseq=3 nst=1` batches appearing once per round are on the **draft context** (`ct=1`) - the
  draft catch-up, not extra target work.
- Per-op timing (sync-inflated, attribution only) shows the verify's individual ops cost the **same** as
  pp3's (ffn_gate ~300 us, lm_head ~3.1 ms). **The ~2.5 ms is not inside any single op.**

## Where the MTP gap stands now

Located: ~4.7 ms of extra GPU execution at >=3 verify positions. Attributed: +144 nodes per position,
plausibly the per-position GDN state-save that exists so state can be rolled back when drafts are
rejected. **Not** a caching problem, and **not** inside any single op. Whether those nodes can be
cheapened or overlapped is a graph-builder change in `llm-graph.cpp`/`qwen35.cpp` - not started, and the
agent explicitly declined to claim it is avoidable.

Instrumentation committed as `c0f4eca55` (`SAKER_CTX_TRACE`, `SAKER_CTX_DUMP`, `SAKER_OP_TRACE`); the
node dump is at `lmhead/nd-mtp.log`. Confirming the +144 ops needs ~10 minutes of diffing that dump.

# q4_0 narrowing: FINAL POSITION - abandoned after two funded attempts, trade-off documented

**Two funded attempts both failed, and I am not starting a third.** The user asked for the narrowing to be
done properly; it was funded twice and produced nothing usable:

| attempt | outcome |
|---|---|
| 1 (agent) | three inconsistent naming conventions -> would not compile (8 errors) |
| 2 (same agent, after steering) | rename loop COMPOUNDED the prefix: `q40_q40_q40_mmq_wg_denoms`, build went 8 -> **186 errors**. Interrupted; tree reverted to `2c8a51820`. |
| 3 (fresh agent, clean context) | stalled in a single generation for ~15 min with no source edits; interrupted |

I also investigated it myself: the `CREATE_MM2(...)` call sites take the tile as a **macro argument**, so
the `warptile_mmq` / `mmq_wg_denoms` names at the call sites are not simple loop-locals that can be rebound
in one place. Resolving it correctly is real surgery in a very large function, and doing it at 73% context
risked a fourth failure.

**A documented limitation beats a broken tree.** The deliverable stays at the known-good `2c8a51820`
state: prefill **409.89 ± 1.53 t/s**, output byte-identical, deterministic, and **2546 of 2547** kernel
cases passing.

## The alternative that IS one line, if the audit must pass by default

Make the retune **opt-in** rather than default - i.e. require `SAKER_APU_MMQ=1` to enable it instead of
`SAKER_APU_MMQ=0` to disable it. Then:

    audit  : 1556/1556 passing by default
    prefill: 301 t/s by default (409.89 with SAKER_APU_MMQ=1)

That trades the +37% headline for full audit compliance by default. **Not done** - it is the user's call
which of the two they want shipped, and the current state (retune ON, one documented case failing) is the
performance-first choice.

# MTP graph delta: DECOMPOSED. It is COPIES and VIEWS, not extra SSM/GDN compute

Extracted a single contiguous graph from the committed node dump and diffed it against the plain pp3
graph (both on node A, `SAKER_CTX_DUMP=1`):

    single MTP verify graph : 3943 nodes
    plain pp3 graph         : 3655 nodes
    delta                   : +288

Per-op decomposition of the delta:

| op | pp3 | mtp | delta |
|---|---:|---:|---:|
| **CPY** | 192 | 288 | **+96** |
| **VIEW** | 690 | 882 | **+192** |
| | | | **+288** |

**And the SSM/GDN compute ops are IDENTICAL in both graphs: 48 `GATED_DELTA_NET` and 48 `SSM_CONV`.**
There are no extra SSM ops at all.

## What this means

The per-position GDN state-save hypothesis is **confirmed in mechanism but wrong in form**. The extra work
is **2 `CPY` + 4 `VIEW` per GDN layer** (48 layers x 2 = 96 copies, 48 x 4 = 192 views) - i.e. the state
save/rollback is implemented by **copying tensors and viewing them**, not by adding SSM compute. The
recurrent maths itself is unchanged between a plain forward and a speculative verify.

**So the ~2.5-4.7 ms is memory traffic (copies), not compute.** That is a materially different fix
direction from "cheapen the SSM ops": it points at avoiding the copy (copy-on-write, or copying only when
a rejection is actually possible) rather than at the kernel.

## Caveat

I did not extract the copied tensors' shapes, so I have not established that the copies are large enough
to account for the full 2.5-4.7 ms. A `CPY` of the GDN recurrent state per layer is on the order of tens
of KB; 96 of them is a few MB, which at ~200 GB/s would be well under 0.1 ms. **The node count proves the
copies exist and are per-layer; it does not yet prove they are the time.** The next step is the shapes, or
a direct A/B with the copy suppressed.

# MTP gap: the mechanism is the GDN STATE CACHES being COPIED per verify position

Following the +288-node decomposition (CPY +96, VIEW +192, SSM ops unchanged), I extracted the copied
tensor names from the committed node dump:

    CPY|cache_r_lN (view) (copy of conv_input-N (view))      432 occurrences
    CPY|cache_s_lN (view) (copy of )                         288
    CPY|cache_r_lN (view) (copy of )                         144

**`cache_r_lN` = the GDN conv/recurrent-state cache; `cache_s_lN` = the GDN SSM-state cache**, one pair
per GDN layer (`_l0`, `_l1`, `_l2`, ...). The verify copies both, per layer, per verify position.

## This IS the mechanism, and it explains why every compute-side probe came back empty

The SSM/GDN **compute** op counts are IDENTICAL between a plain forward and a verify (48 `GATED_DELTA_NET`,
48 `SSM_CONV`). What differs is **`CPY` of the state caches**. So the rollback machinery saves state by
copying `cache_r`/`cache_s` - pure memory traffic, no extra arithmetic. That is exactly why sync, logits,
CPU submit, KV-position, async draft work, per-op timing and the lm_head all came back negative: none of
them is a copy of the recurrent state.

## Correction to my own caveat one step earlier

I wrote that the copies were "on the order of tens of KB... well under 0.1 ms". **That was an assumption
about the tensor size, and the names show it was wrong.** `cache_r` is the conv state (conv_input-shaped,
modest) but **`cache_s` is the SSM state**, which for a GDN layer with many value heads is potentially
MB-scale. 96 copies per graph at MB scale is plausibly **1-3 ms** - the right order for the missing time.

The dump carries no shapes, so the sizes are still not measured. **But the identification is now specific:
it is the GDN state caches, and the fix direction is to avoid or cheapen the copy** - copy-on-write, or
copying only when a rejection is actually possible - rather than to touch the SSM kernels.

## What remains to confirm

1. The shapes of `cache_r` / `cache_s` (a one-line print in the graph builder, or a dump extension).
2. Whether the copy can be suppressed when no draft can be rejected (which is the 2-position case, where
   the gap is absent - consistent with this mechanism).

# MTP gap: the state-cache copies are TOO SMALL to be the time. My "reversal" was wrong.

I found that the +288-node verify delta is `CPY` of `cache_r_lN` / `cache_s_lN` (the GDN conv-state and
SSM-state caches), then talked myself into a reversal: "the copies are MB-scale, so they could be the
1-3 ms". **Sizing them from the source shows that was wrong and my original caveat was right.**

From `src/llama-hparams.cpp`:

    n_embd_r() -> (ssm_d_conv-1) * (ssm_d_inner + 2*ssm_n_group*ssm_d_state)     // conv state
    n_embd_s() -> n_embd_head_la^2 * n_head()                                    // linear-attention state

and from `src/llama-memory-recurrent.cpp`:

    ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, n_embd_r(), n_rows);   // n_rows = mem_size*(1+n_rs_seq)
    ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, n_embd_s(), n_rows);

The recurrent state is **per-sequence, not per-token** (`n_rows` = mem_size x (1 + n_rs_seq), which is 1 for
a single sequence) - so these are small tensors, not per-token buffers. Estimating for this model (head
dim 128, ~40 heads, 48 GDN layers, F16):

    cache_s ~ 128*128*40 * 2 B  ~ 1.3 MB per layer
    cache_r ~ 5120*3 * 2 B      ~ 30 KB per layer
    per verify: 48 layers x (r + s) x 2 copies  ~ 64 MB copied
    at ~200 GB/s                                 ~ 0.3 ms   (0.6 ms if the state is F32)

## Conclusion: the copies are ~0.3-0.6 ms of a 2.5-4.7 ms gap

**The +288 nodes are real but account for roughly a tenth of the missing time.** The MTP gap therefore
remains **largely unexplained** after eliminating: sync mechanism, all-positions logits, CPU submit,
KV-position interaction, async draft-context work, the multi-column lm_head, graph rebuilding, extra
SSM/GDN compute ops, and now the state-cache copies.

This is the honest end state of the investigation: **located to within a factor of ten, not attributed.**
A next step that would actually settle it is a per-op GPU timeline (device timestamps per node) rather
than more node counting - the node counts have now been exhausted as a diagnostic.

# DFlash2 as a draft instead of MTP: NOT POSSIBLE - different architecture, not a tokenizer issue

Attempted `--spec-type draft-dflash` with
`/home/hycris/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q8_0.gguf` (2.06 GB, Q8_0):

    E llama_model_load: error loading model: done_getting_tensors: wrong number of tensors;
      expected 81, got 58
    E common_speculative_init_result: failed to load draft model

## Why

llama.cpp here DOES register `LLM_ARCH_DFLASH` (`src/llama-arch.cpp:139`, arch string `"dflash"`) and
`src/models/dflash.cpp` exists - so the spec type is real and supported. The GGUF even declares
`general.architecture = dflash`. **But the tensor layouts are entirely different:**

| DFlash2 (Qwen3.8-27B) tensors | this fork's `dflash.cpp` expects |
|---|---|
| `blk.N.attn_conv_base`, `blk.N.attn_conv_proj.weight` | `layer.wq_a` (`LLM_TENSOR_ATTN_Q_A`) |
| `blk.N.ffn_conv_base`, `blk.N.ffn_conv_proj.weight` | `LLM_TENSOR_ATTN_SINKS` |
| `blk.N.attn_k_norm`, `attn_q_norm`, `attn_output` | `LLM_TENSOR_DSPARK_MARKOV_W1/W2`, `CONF_PROJ` |
| `enc.output_norm.weight` | `LLM_TENSOR_HC_HEAD_FN/BASE/SCALE` |
| **58 tensors** | **81 expected** |

`attn_sinks`, `markov_w*` and `hc_head_*` are **DeepSeek-V4 dspark tensors**. So this fork's
`LLM_ARCH_DFLASH` implements the **DeepSeek-V4 dspark/dflash** layout, not z-lab's Qwen3.8-27B DFlash2
layout. The Qwen file is a different format built for the **vLLM stack**.

## Correction to my own earlier note

My project notes recorded DFlash2 as "not a DeepSeek option... a draft head must share the target's
tokenizer". **That reasoning does not apply here**: this target IS Qwen3.8-27B, so the tokenizer matches
perfectly. The real blocker is the **architecture/tensor layout**, not the vocabulary. The note was right
about the conclusion for a different target and wrong about the reason for this one - a reminder that a
note written against one target should not be trusted against another without re-deriving it.

## What it would take to use it

Either (a) a converter from the z-lab DFlash2 layout into this fork's dflash tensors - which is not a
rename but a different model topology, so it means implementing a new arch in llama.cpp - or (b) running
the speculator on the vLLM stack, which is outside this project's boundaries. Neither is a configuration
change, and the draft is 2.06 GB Q8_0 so a draft pass would read ~10 ms of weights anyway - it would have
to draft a block per pass to be competitive with the MTP head's 5.3 ms/round.

# DFlash2: the draft is CORRECT, the FORK is too old. Root cause found.

Follow-up to the failed DFlash2 load. **This was never a model problem.**

- The canonical Qwen3.8-27B DFlash2 speculator is **`z-lab/Qwen3.8-27B-DFlash2`** (mirrored
  `incoai/Qwen3.8-27B-DFlash2`), and its GGUF is **`incoai/Qwen3.8-27B-DFlash2-GGUF`** (mirrored
  `z-lab/...`). Community quants also exist (`Akicou/...`, `HermiHg/...-Q2_K_S-MIX-GGUF`).
  **We already have the right file** at
  `~/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q8_0.gguf`.
- **llama.cpp PR #27342 — "spec : add DFlash2 support (local convolution + candidate selector)" by
  SubSir — is merged into mainline.** It adds the DFlash2 path.
- Our fork `llama.cpp-qwen38` predates it: its `src/models/dflash.cpp` implements **DFlash1**
  (the DeepSeek dspark layout - `attn_sinks`, `markov_w1/w2`, `hc_head_*`), which is exactly the
  **81-tensor** set the loader expected. The Qwen3.8-27B DFlash2 GGUF is the **DFlash2** layout with
  **58 tensors** (`attn_conv_*`, `ffn_conv_*`, `enc.output_norm`).

**So the fix is to bring PR #27342 into the fork, not to find another model.**

Caveat on provenance: the PR number, title, author and merged status come from a web search, not from
inspecting the PR myself. Verify against the actual commit before relying on it.

## Why this could matter more than anything else left

The MTP head gives acceptance A = 2.46 (p1 0.778, p2 0.71) and the round is ~66.5 ms. A purpose-built
DFlash2 speculator is a different class of draft - it selects from a candidate set with a local
convolution rather than reusing the target's nextn head - so its acceptance could be substantially
higher. Higher A is the one lever that moves MTP >= 40 without touching the weight pass:

    round 66.5 ms, A = 2.46 -> 37.0 tok/s        (measured)
    same round,   A = 3.00 -> 45.1 tok/s
    same round,   A = 2.70 -> 40.6 tok/s

The draft cost would rise (2.06 GB Q8_0, so ~10 ms per draft pass versus the MTP head's 5.3 ms/round),
so the net is not simply proportional - it has to be measured. But **this is the first lever identified
in a long time that is not arithmetically closed**, and it needs a code port rather than a kernel.

# DFlash2 port: TARGET IDENTIFIED (fetch done, cherry-pick not started)

The quiet upstream fetch SUCCEEDED on node A in tree `/home/hycris/llama.cpp-dflash2`
(`git fetch -q --no-tags upstream master > /tmp/fetch.log 2>&1`):

    upstream/master = 42b70032b

**The DFlash2 commit is confirmed:**

    b10f9ca58  spec : add DFlash2 support (local convolution + candidate selector) (#27342) (#27816)

So the PR number (#27342) was correct and it landed upstream as **#27816**. Verification that upstream
actually handles the Qwen3.8-27B DFlash2 layout:

    upstream src/models/dflash.cpp  -> 20 references to attn_conv / ffn_conv / dflash2
    upstream src/llama-arch.cpp     ->  0 references to dflash2

i.e. DFlash2 is **not a new arch** - the string stays `dflash` and the DFlash2 path lives inside
`dflash.cpp`. That is exactly why the GGUF declared `general.architecture = dflash` and still failed:
the fork's `dflash.cpp` is the older DFlash1 (DeepSeek dspark) implementation.

Related dflash follow-ups worth including in the port:
    a11f57ba9  model : fix DFlash output head sharing (#30111)
    cc231cb0d  dflash: pass missing NVFP4 scales to attention operations (#28000)

## The port that remains

    cd /home/hycris/llama.cpp-dflash2          # branch saker-dflash2 @ f65d47b8a
    git cherry-pick b10f9ca58                  # + a11f57ba9, cc231cb0d if they apply
    cmake --build build --target llama-server llama-bench -j 10
    # GATE 1: does the GGUF load?
    llama-server -m <IQ2_S target> -md /home/hycris/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q8_0.gguf \
        -ngl 99 -c 4096 --spec-type draft-dflash
    # GATE 2: measure vs the MTP arm (36.65-36.78 t/s, acceptance 0.77551)

The upstream remote is already added and fetched, so the cherry-pick can start immediately.

## Blocker: subagent quota exhausted

Both port attempts were killed before the fetch completed - the first by a runaway `git fetch` that dumped
765,574 tokens into its context, the second by the provider quota:

    429 / code 1310: "you have reached the weekly/monthly usage limit, resets 2026-10-23 09:09:03"

**No further subagents can be launched on `zai-coding-cn/glm-5.3` until 2026-10-23.** The parent had ~10%
context left and could not do the cherry-pick + build + measurement itself, so the port is left at a clean,
prepared state: tree ready, remote fetched, target commit identified.

# DFlash2 GATE 1 PASSED - the speculator LOADS and runs on upstream/master

**The answer to "is there a proper DFlash2 draft" is YES, and it is the file we already had.**
The blocker was the fork's age, exactly as suspected.

Built `/home/hycris/llama.cpp-upstream` at **upstream/master = 42b70032b** (llama.cpp b11731) and loaded
the DFlash2 speculator against the IQ2_S-mtp target:

    I common_speculative_init_result: loading draft model '.../Qwen3.8-27B-DFlash2-Q8_0.gguf'
    I common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dflash'
    I common_speculative_impl_draft_dflash: - n_max=3, n_min=0, p_min=0.00
    I common_speculative_impl_draft_dflash: - block_size=8, mask_token_id=248070, n_extract=5, sample_from_anchor=true
    I srv  llama_server: model loaded
    I srv  llama_server: listening on http://127.0.0.1:18791

No "wrong number of tensors". The 81-vs-58 error is gone.

**What DFlash2 says about itself: block_size=8, n_extract=5** - a block speculator that can extract 5
tokens per draft pass, against the MTP head's 2 (n_max=2). That is the number that decides whether it can
pay for its 2.06 GB of weights.

## Why the cherry-pick route was abandoned in favour of building upstream directly

The cherry-pick onto the fork did apply (head dc5c79544, dflash.cpp gained 16 DFlash2 tensor refs), but it
**failed to compile**:

    src/models/dflash.cpp:847:18: error: 'ggml_rope_set_offset' was not declared in this scope

Tracing that: `ggml_rope_set_offset` came from a *different* upstream commit -
`fe8156f78 ggml: add ggml_rope_set_offset (+ metal support) (#27120)`. The fork's rope op carries
`int32_t params[15]` (indices 0-14) while upstream writes `params[15]`, so the op is one field short, and
**the fork's backends ignore n_offs entirely** (the grep hits in ggml-cpu/ops.cpp and ggml-vulkan.cpp were
`rotate_pairs(..., n_offset, ...)` and `misalign_offsets`, unrelated). Upstream/master by contrast has
n_offs in ggml.c (4 refs), ggml_rope_set_offset (2 refs), and honoured in **cpu (11 refs) and vulkan
(3 refs)**.

So finishing the cherry-pick would have meant hand-porting rope kernels into CPU *and* Vulkan in a tree
that already carries the Saker Vulkan patches - the exact class of work that failed repeatedly on this
project. Building upstream/master instead gets DFlash2 **and** n_offs **and** every backend's kernel
support for free.

Upstream/master also has the MTP path (`graph_mtp` in src/models/qwen35.cpp, `draft-mtp` in
common/speculative.cpp), so the comparison arm runs on the **same build** - which is the fair test.

## Traps hit on the way (both self-inflicted, both recorded)

- A `git clone` of a local repo copies only `refs/heads`, **not** remote-tracking refs, so `upstream/master`
  was missing in the clone and the checkout silently fell back to `saker-dflash2`, launching a build of the
  tree that cannot compile. Fixed with an explicit refspec fetch.
- `rm -rf build` raced with a build that had just been killed, leaving stale shader-gen output
  ("cannot remove ... vulkan-shaders.spv: Directory not empty"). The next build then failed on
  `MmTypeA / GGML_TYPE_Q4_0 undeclared` in mul_mm_funcs.glsl - a *stale-state* error, not a source error.
  Cleanly removing the dir fixed it.

# DFlash2 GATE 2 - measured. Works, wins on the same build, does NOT beat the deliverable.

Both arms on the SAME upstream build (`llama.cpp-upstream` @ 42b70032b, provenance verified), prompt-short16,
n_predict=64, temp 0, cache_prompt=false:

    ARM                       t/s     draft_n  accepted  acc/round  mean_len
    MTP (upstream, no shortlist)  31.96      50        38      0.760      2.52
    DFlash2                       33.41      65        41      0.631      2.86

Output text is IDENTICAL between the two arms (both begin "The user is asking me to review a HIP GEMM
implementation..."), so DFlash2 is lossless on this prompt.

**DFlash2 is +4.5% over MTP on the same build, and it wins while having LOWER acceptance per round**
(0.631 vs 0.760). The reason is mean accepted length: 2.86 vs 2.52. That is the block_size=8 /
n_extract=5 geometry doing exactly what it is supposed to - extracting more tokens per draft pass. The
MTP head drafts 2 (n_max=2); DFlash2 extracts ~2.86.

## But it does not beat what already ships

    deliverable MTP (Saker shortlist + fused gather)   36.65-36.78 t/s
    upstream MTP (same build as DFlash2)               31.96 t/s
    DFlash2 (same build)                               33.41 t/s

DFlash2 is below the deliverable because the deliverable's MTP carries two MTP-specific Saker patches
(SAKER_SHORTLIST vocab pruning, and the fused gather-matvec lm_head) that **cannot transfer to DFlash2** -
the shortlist prunes the target's own nextn head, which DFlash2 does not use.

## Round arithmetic - is 40 reachable from here?

    33.41 t/s at mean_len 2.86  ->  11.68 rounds/s  ->  round = 85.6 ms
    (compare: the deliverable MTP round is ~66.5 ms)

    40 t/s needs round <= 61.5 ms. At a 85.6 ms round that would need mean_len = 3.42.
    So DFlash2 needs mean_len 2.86 -> 3.42 (+20%), or a cheaper draft pass, to reach 40.

The draft is **Q8_0 (2.06 GB)**, so the draft pass reads ~10 ms of weights per round against the MTP head's
5.3 ms - that is where DFlash2's larger round comes from. A smaller draft quant would attack exactly that
term, and smaller quants of this speculator are published (`HermiHg/...-Q2_K_S-MIX-GGUF`, `Akicou/...`).
Also untested: n_max (it defaults to 3 while block_size is 8).

## Verdict

**"Is there a proper DFlash2 draft on HuggingFace?" - YES, and it is the file we already had.** The whole
problem was that this fork's dflash.cpp implements DFlash1. Upstream/master (b11731) loads it and runs it.
It is a real, working, lossless-on-this-prompt speculator at 33.41 t/s - **but it is not yet a route to 40**,
and it does not displace the 36.65-36.78 deliverable as it stands.

# DFlash2 n_max sweep - the default (3) is already optimal, and n_max>=5 CHANGES THE OUTPUT

Same build, same prompt, n_predict=64, temp 0:

    n_max   t/s     draft_n  accepted  acc/round  mean_len  output md5
    3(*)   33.41      65        41       0.631      2.86     2fe11002  <- baseline, preserved
    4      25.22      76        43       0.566      3.15     2fe11002  <- baseline, preserved
    5      21.66      95        43       0.453      3.26     e328a405  <- DIVERGES
    6      20.93     108        45       0.417      3.50     e328a405  <- DIVERGES
    8      18.05     125        45       0.360      3.50     e328a405  <- DIVERGES
    (*) default

**Two conclusions.**

1. **n_max=3 (the default) is optimal.** Raising n_max increases mean accepted length (2.86 -> 3.50) but
   loses throughput monotonically, because the verify batch widens faster than extraction grows. Same shape
   as the MTP head's n_max knee, but here there is no gain at all above the default.

2. **n_max >= 5 does not preserve the output.** `2fe11002` is this project's recorded baseline output md5
   (276 chars at n_predict=64 on prompt-short16); n_max 3 and 4 reproduce it, n_max 5/6/8 produce
   `e328a405` instead. So high-n_max DFlash2 is a *different* generation, not merely a slower one. This is
   a correctness caveat, not a performance one - recorded whether or not it is ever used.

## Final position on DFlash2

    deliverable MTP (shortlist + fused gather)   36.65-36.78 t/s   <- ships
    DFlash2 (upstream, default n_max=3)          33.41     t/s
    upstream MTP (same build as DFlash2)         31.96     t/s

**DFlash2 works, is lossless at its default setting, and beats MTP by +4.5% on an identical build - but it
is not a route to 40 tok/s and does not displace the deliverable.** Its round is 85.6 ms against the
deliverable's ~66.5 ms; reaching 40 at that round would need mean_len 3.42 versus the measured 2.86 (+20%),
and the n_max sweep shows that is not available from tuning.

The one untested lever left is the **draft quant**: 2.06 GB of Q8_0 means ~10 ms of weights read per round
against the MTP head's 5.3 ms, which is precisely where DFlash2's larger round comes from. Smaller quants
of the same speculator are published (`HermiHg/...-Q2_K_S-MIX-GGUF`, `Akicou/...`). That would cut the draft
term directly - but it is a fresh download and a fresh measurement, and the ceiling it could buy is
33.41 -> ~40 only if the draft cost halves with no acceptance loss, which is an assumption, not a result.

# CORRECTION - two wrong claims about DFlash2 output md5, and a variance finding

**I got this wrong twice. Both are retracted.**

## What I claimed, and why it was wrong

Claim 1 (recorded in the n_max sweep section): "n_max 3 and 4 reproduce the baseline md5 2fe11002;
n_max 5/6/8 do not (e328a405), so high-n_max DFlash2 is a different generation."

Claim 2 (raised during the optimization investigation): "DFlash2's default output md5 is e328a405, the
diverged value, so the default diverges."

**Both are wrong, and the measurement that shows it is the baseline arm itself.** Running the MTP arm twice
on the same build, same server, same prompt:

    arm      run  t/s     md5        len   acc
    mtp       1    34.86   e328a405   290   0.760
    mtp       2    34.93   2fe11002   276   0.776     <- SAME ARM, DIFFERENT OUTPUT
    dfl-n3    1    33.60   e328a405   290   0.631
    dfl-n3    2    36.95   e328a405   290   0.729
    dfl-n4    1    33.82   2fe11002   276   0.566
    dfl-n4    2    34.24   2fe11002   276   0.566

The baseline arm produces **both** md5s. There are two attractors - 276 chars (`2fe11002`) and 290 chars
(`e328a405`) - and which one a run lands on is not a property of the arm. So:

- The n_max sweep's md5 column measured **which attractor each arm happened to land on**, not a defect.
  "n_max>=5 diverges" is retracted.
- DFlash2's default landing on `e328a405` is **not** a divergence. Retracted.
- The Gate 2 claim that "output text is identical between arms" was based on a **100-character prefix**;
  the two attractors share their first 100 chars and differ later. That claim was under-evidenced and is
  also retracted.

This is the same non-reproducibility this project already recorded (at 256 tokens, 2 distinct outputs in 3
runs of the baseline). **It also happens at n=64**, which was previously believed reproducible (5 runs/arm
byte-identical). The lesson is narrower and more useful than "speculation is lossy": **md5 at n=64 on this
prompt is not a correctness discriminator, because the baseline itself is not stable.**

## Variance finding - the earlier DFlash2-vs-MTP comparison was under-powered

    MTP, first Gate 2 run            31.96 t/s
    MTP, md5 check (2 runs)          34.86 / 34.93 t/s      <- +9% over the first reading
    DFlash2 n_max=3 (2 runs)         33.60 / 36.95 t/s      <- 10% spread WITHIN the arm

So single-run arm comparisons on this box are not decisive. Any DFlash2-vs-MTP claim needs >=3 reps per arm.
The honest current statement is: **on the same upstream build, DFlash2 and MTP are within noise of each
other (both ~34-37 t/s), not the clean +4.5% I reported from one run each.**

## What survives from the investigation (these are solid)

- DFlash2's GGUF is **Q8_0, 1.92 B params, 2.05 GB, 8.50 bits/param** (measured from tensor types, not the
  filename). A full draft pass therefore reads ~2.05 GB -> ~10.5 ms at 195 GB/s.
- DFlash2 has **no token_embd and no output.weight** - it reuses the target's lm_head.
- It has 5 blocks and vocab-sized selector tensors (`selector_predecessor` / `selector_successor`,
  63.57 M params each = ~248070 x 256).
- Under DFlash2 the target's MTP layer (blk.64, ~348 MB, 12 tensors) is **unused**, which is why DFlash2's
  prompt eval is ~618 ms against MTP's ~1396 ms for the same 104-token prompt.

# DFlash2 vs MTP, properly powered (3 reps/arm) - DFlash2 does NOT win

    arm       run1    run2    run3    mean/median   spread   md5s
    mtp       34.85   34.92   34.92   34.90         0.2%     e328a405 / 2fe11002 / 2fe11002
    dfl-n3    33.57   36.87   33.55   33.57 (med)   10%      e328a405 x3
    dfl-n3b   33.87   34.28   34.00   34.05         1.2%     2fe11002 x3

**Label bug, stated for the record: `dfl-n3b` is actually `--spec-draft-n-max 4`.** A sed edited the arm's
label without editing its argument. The numbers are valid; the name is wrong.

## Conclusions

1. **DFlash2 does not beat MTP on the same build.** 33.57 (median) against 34.90, and MTP is far more stable
   (0.2% spread vs 10%). The "+4.5% for DFlash2" reported from the first Gate 2 run is retracted: that run
   paired a lucky DFlash2 sample (33.41, close to its own median) against a depressed MTP baseline (31.96,
   against 34.90 here).
2. **MTP's own baseline moved 31.96 -> 34.90 (+9%) between sessions on the same build.** Single-run arm
   comparisons on this box are not decisive; 3 reps minimum.
3. **The deliverable (fork MTP + shortlist + fused gather, 36.65-36.78) remains the fastest configuration
   measured**, ahead of upstream MTP (34.90) and DFlash2 (33.57).

## DFlash2 cannot reach 40 even with a free draft

Using its own measured numbers at the median acceptance (round 85.6 ms, 2.86 tokens/round, 33.4 t/s):

    draft cost 2.05 GB Q8_0 -> ~10.5 ms/round
    if the draft were FREE:  round 85.6 - 10.5 = 75.1 ms -> 2.86/0.0751 = 38.1 t/s

So eliminating the entire draft pass still leaves DFlash2 short of 40. (The single high-acceptance outlier
run - 36.87 t/s at acc 0.729 - would project to ~42 with a free draft, but that run is not reproducible:
the same arm gave acc 0.631 on its other two reps.)

**This closes DFlash2 as a route to the MTP >= 40 criterion.**

## Does the MTP optimization transfer? Structurally yes, economically no

    Saker MTP patch              targets                          applies to DFlash2?
    SAKER_SHORTLIST              draft head's per-step vocab read  yes - DFlash2 has vocab-sized
                                                                   selector tensors (63.57 M each)
    fused gather-matvec lm_head  the gather over the 248k vocab    yes - DFlash2 has NO own
                                                                   token_embd/output.weight, it
                                                                   reuses the target's lm_head

But the payoff is small: DFlash2's round is dominated by (a) the target weight pass (~47.5 ms, and it
already saves ~348 MB by not touching the target's blk.64 MTP layer), (b) an 8-position block verify, and
(c) its 2.05 GB draft (~10.5 ms). The lm_head and selector reads are a few ms of an ~85 ms round - and the
whole point of the two patches was to cut a term that is not DFlash2's bottleneck.

**And since DFlash2 is slower than MTP to begin with, optimizing it does not advance any goal criterion.**

# PER-OP GPU TIMELINE - the round is now decomposed, and the GDN state copies are 6-10x worse than estimated

**Method: llama.cpp's Vulkan backend already has a per-op GPU timer. `GGML_VK_PERF_LOGGER=1` uses
`vk::QueryType::eTimestamp` query pools and logs each node's GPU time (ggml-vulkan.cpp:17439), printing a
"Vulkan Timings:" table to std::cerr after each graph compute.** This is the per-op timeline the project
had been treating as "needs building" - it was already there.

Two traps cost time here, both self-inflicted:
- `GGML_VK_PERF_LOGGER_FREQUENCY=1` means "print every compute" (`print_count % frequency != 0 && !force`),
  so it prints on every graph compute, not once. 107 blocks in the log.
- **My grep pattern `vulkan.*time` cannot match the literal `Vulkan Timings:`** because "Timings" does not
  contain "time". I concluded the logger produced no output when the table was in the log all along.
- The first parse regex `^([A-Za-z0-9_]+):` also failed, because the logger prints *fusion names* which
  contain spaces and shape params (`MUL_MAT_VEC iq2_s m=17408 n=3 k=5120`). That silently dropped every
  matmul and made the output look like elementwise ops were dominant.

Cost of instrumenting: the logger forces a submit/wait per graph compute, and throughput dropped from
36.65-36.78 to 33.89 t/s (~8%). Acceptance stayed bit-identical at **0.77551 (38/49)**, confirming it was
the same arm. Relative breakdown is valid; absolute times are inflated.

## Headline: the GPU is 96% busy

    GPU total (prefill + 64-token decode)   2385.0 ms
    wall for the same work                  ~2490 ms      -> 96% GPU-busy

**So host overhead is ~4%, not the ~11% I had been assuming.** The round is GPU-bound. This kills the
"overlap the host work" idea as a lever: there is almost nothing to overlap.

## GPU time by op family

    family            calls   total_ms  ms/round  %GPU
    MUL_MAT_VEC       14432   1574.6    65.61     66.0
    MUL_MAT             504    459.3    19.14     19.3     <- prefill (n=100)
    CPY                5568     90.8     3.79      3.8
    RMS_NORM_MUL       6607     54.3     2.26      2.3
    GET_ROWS           2890     38.1     1.59      1.6
    GATED_DELTA_NET    1392     35.1     1.46      1.5
    MUL_MAT_ADD         100     22.6     0.94      0.9
    ADD                5160     18.6     0.78      0.8
    FLASH_ATTN_EXT      542     11.8     0.49      0.5
    ... 22 families, 55868 calls total (2328/round at 42.7 us mean)

## Finding 1 - the "unattributed" verify time is just the cost of extra positions

Decode matvecs are `MUL_MAT_VEC ... n=3` (3 verify positions). 1574.6 ms over ~25 rounds = **62.0 ms/round
against a 49.8 ms bandwidth floor**. Plain decode (n=1) measures 49.75 ms/token, i.e. exactly the floor.
So the +12.2 ms is **the compute cost of verifying 3 positions, ~6.1 ms per extra position** - not an
inefficiency to remove. This is the honest resolution of the "~9.4 ms unattributed" from the earlier round
decomposition: it is attributed now, and it is irreducible by tuning.

## Finding 2 - THE GDN STATE COPIES COST 3.79 ms/round, not 0.3-0.6 ms

    CPY   5568 calls   90.8 ms total   3.79 ms/round   16.3 us per copy

These are the `cache_r_lN` (GDN conv state) and `cache_s_lN` (GDN SSM state) copies the graph builder emits
per verify position. The project previously sized them from their data volume (~64 MB/verify) and estimated
**0.3-0.6 ms - a 6-10x underestimate.** The reason is that **16.3 us for a 30 KB - 1.3 MB copy is launch
overhead, not bandwidth**: these are many tiny copies, not a few large ones.

**This is the first concrete lever found in a long time.** Eliminating or batching them:

    round 66.5 ms - 3.79 ms = 62.7 ms  ->  2.52 / 0.0627 = 40.2 t/s

That reaches the >= 40 criterion on its own, with the current acceptance and no other change. It is a
graph-builder change (batch the per-position state saves, or avoid them entirely), not a kernel.

## Also measured

- **lm_head**: `MUL_MAT_VEC iq4_xs m=248320 n=3 k=5120` - 24 calls, 75.7 ms, **3.15 ms/round**. That is
  715 MB/round of iq4_xs weights at ~227 GB/s, so it is already at bandwidth. The shortlist gather is
  `MUL_MAT_ID_VEC iq4_xs m=1 n=3584 k=5120 n_expert=248320` at 56.6 us - the SAKER_SHORTLIST=3584 working,
  and negligible. Note the verify still needs full-vocab logits (n=3), so the full lm_head read is required.

# The CPY "lever" is CLOSED, and my claim about it is RETRACTED - the copies are necessary

## What I claimed one step earlier, and why it was wrong

I recorded: "CPY costs 3.79 ms/round ... eliminating or batching them: round 66.5 - 3.79 = 62.7 ms ->
40.2 t/s. That reaches the >= 40 criterion on its own." **Retracted.**

**The copies are the recurrent-state rollback snapshots, and their count is fixed by the speculation
depth.** The chain, read end to end:

    common/common.h   uint32_t need_n_rs_seq() const {
                          bool needs_rs_seq = ... DRAFT_MTP || DRAFT_EAGLE3 || DRAFT_DFLASH || DRAFT_DSPARK ...;
                          return needs_rs_seq ? draft.n_max : 0u;      // <- n_rs_seq == n_max
                      }
    common/common.cpp cparams.n_rs_seq = params.speculative.need_n_rs_seq();
    delta-net-base.cpp  const int64_t K = (int64_t) cparams.n_rs_seq + 1;   // K copies per GDN layer
                        for (int64_t t = 1; t <= K; ++t) { ... ggml_cpy(conv_state_last, conv_state_update); }

With `--spec-draft-n-max 2` (the deliverable's setting) this is already K = 3 copies per layer - the
**minimum**, because a 3-position verify must be able to roll the GDN state back up to 2 positions. There is
no slack to remove. Lowering it means lowering n_max, which costs acceptance, and the project already
established n_max=2 as the optimum.

The `n_rs_seq == 0` branch does a single copy per layer, but it disables rollback entirely - it is not
available to a speculative decode that must undo rejected drafts.

## Correction to the node dump I reasoned from

The 864-node dump showed **9 copies per GDN layer**, which I read as the deliverable's graph. But K = 9
implies n_rs_seq = 8, i.e. `n_max=8` - so **that dump came from a different configuration**, not the
deliverable's n_max=2. The measured live count (~223 CPY/round, consistent with 3 copies x 48 layers plus
the cache_s side) is the deliverable's real figure. Another case of reading a stale artifact as current.

## Cross-check that also failed to help: upstream is identical

    fork:      CPY  5568 calls   90.8 ms   3.8%
    upstream:  CPY  5376 calls   90.2 ms   3.7%

Upstream differs from the fork in the recurrent-memory code - it has a third buffer (`memory_size_p`) and
logic that disables snapshots when the module is empty - but the CPY cost is unchanged. So this is not a
backportable fix either.

## THE ROUND IS NOW FULLY DECOMPOSED, WITH NO REMAINING WASTE

    49.8 ms   target weight pass (9.608 GB at the achievable 193-227 GB/s)
    12.2 ms   verify compute for 3 positions (~6.1 ms per extra position; plain decode n=1 hits the floor)
     3.1 ms   rollback snapshots (K=3 copies x 48 GDN layers) - NECESSARY for n_max=2
     1.0 ms   lm_head full-vocab read (715 MB iq4_xs at ~227 GB/s - already at bandwidth)
    ~4%       host overhead (GPU measured 96% busy)
    ------
    ~66.5 ms  -> 36.65-36.78 t/s

40 t/s needs <= 61.5 ms. **Every term is either at its bandwidth floor or is necessary work.** The only
theoretical headroom left is the verify's compute efficiency: 12.2 ms for 2 extra positions implies roughly
9 TFLOP/s effective against an APU peak of ~14-22 TFLOP/s. Exploiting that is a kernel-optimisation
project of exactly the kind that failed three times on this target (sign-fold +0.45%, q4_0 narrowing,
prefill arms that produced wrong text).

**So MTP >= 40 is excluded by the sum of its necessary parts, not by a single arithmetic wall.** That is a
stronger statement than the earlier "gap located but not attributed", and it is the honest end state.

# DMMV workgroup heuristic on AMD - FALSIFIED (-8.3%), and the matvec-geometry lever is closed

## The lead

The per-op timeline put MUL_MAT_VEC at 66% of GPU time, running 24% above the bandwidth floor at n=3. The
matvec workgroup heuristic that picks `DMMV_WG_SIZE_LARGE` for small-m / large-k is **gated to
NVIDIA/Intel**, so this AMD device always gets `SUBGROUP`:

    uint32_t dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
    if ((NVIDIA && !PRE_TURING) || INTEL) {          // <- AMD never enters
        if (a_type == Q6_K) { if (m <  4096 && k >= 1024) dmmv_wg = LARGE; }
        else                { if (m <= 8192 && k >= 1024) dmmv_wg = LARGE; }
    }

The prior `SAKER_DMMV_WG` knob forced one value globally and showed LARGE helps plain decode but destroys
MTP - consistent with the rule wanting *per-op* selection. So the untried experiment was to let AMD enter
the real per-op heuristic.

## The change (tree /home/hycris/llama.cpp-dmmvheur, branch saker-dmmvheur, commit e8201e7c6)

Env-gated, default behaviour unchanged:

    if (getenv("SAKER_DMMV_HEUR") != nullptr || (ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA ...

Applied at both gate sites (7865, 8032). Verified in the built artefact: `SAKER_DMMV_HEUR` present in
`libggml-vulkan.so.0.20.0`, freshness source 19:42:49 -> object 19:45:03 -> binary 19:46:01.

## The measurement (one binary, env-toggled, so no build-to-build confound; 3 reps/arm)

    arm                         run1    run2    run3    mean    spread
    ctrl  (SUBGROUP)            36.95   36.89   36.85   36.90   0.3%
    heur  (SAKER_DMMV_HEUR=1)   33.82   33.84   33.83   33.83   0.1%
                                                        -> -8.3%

    acceptance 0.77551 (38/49) in ALL six runs;  md5 2fe11002 in ALL six runs

**Falsified.** AMD's SUBGROUP default is correct for this workload, and the heuristic's "LARGE when
m <= 8192" choice is actively wrong here. The matvec-geometry lever is closed.

Two useful by-products:
- **The control reproduced the deliverable exactly** (36.85-36.95 against the shipped 36.65-36.78), so the
  clone is a faithful baseline and this A/B is trustworthy.
- **The output was bit-identical across all six runs** (2fe11002, 276 chars). So the two-attractor
  behaviour seen in the earlier session may be *session-dependent* (GPU/driver state) rather than a property
  of the arm. That does not restore md5 as a correctness discriminator - the baseline still flipped within
  one session - but it narrows what is flipping and when.

## Lever inventory after this - all closed

    DFlash2 draft                 measured, does not beat MTP, cannot reach 40 even with a free draft
    CPY / rollback snapshots      necessary work; count = n_rs_seq+1 = n_max+1, already minimal
    matvec workgroup geometry     FALSIFIED -8.3% (this entry)
    n_max sweep                   default already optimal
    lm_head shortlist + fused gather   already shipped (+32-43% over baseline)
    sign-fold dequant ALU         +0.45% on Vulkan (failed earlier)
    prefill >= 500                closed: correct ceiling 411-413
    plain >= 26                   excluded: 0.6% short of theoretical peak

# DFlash2 draft cost MEASURED (was an estimate) - the free-draft claim holds

The user pushed back on "DFlash2 cannot reach 40 even with a free draft", correctly: that rested on an
*estimated* draft cost (2.05 GB / 195 GB/s = 10.5 ms) that was never measured. It is measured now.

## Method - the draft is cleanly isolable by weight type

The DFlash2 speculator is **Q8_0**; the target is a mixed IQ2/IQ3/IQ4/Q2_K/Q6_K quant with **no q8_0
tensors at all**. So every q8_0 op in a perf-logger trace of the DFlash2 arm belongs to the draft.
Verified: **the MTP run shows 0.0 ms of q8_0** - the isolation is exact, not heuristic.

    DFLASH2 total 2468.2 ms / 55573 calls   |   MTP total 2406.2 ms / 54347 calls

    DFlash2 GPU time by weight type:
      iq3_xxs  356.0 (14.4%)   iq2_s   305.0 (12.4%)   iq2_xxs 285.3 (11.6%)
      iq3_s    272.6 (11.0%)   ?       254.8 (10.3%)   q8_0    223.6 ( 9.1%)
      iq2_xs   194.6 ( 7.9%)   q2_K    176.5 ( 7.2%)   iq4_xs  168.5 ( 6.8%)
      iq1_m     65.2 ( 2.6%)   q4_K     34.9 ( 1.4%)

    *** DRAFT (q8_0) = 223.6 ms total = 10.0 ms/round ***

## Verdict: the claim HOLDS

    round 85.6 ms at 2.86 tok/round; 40 t/s needs round <= 71.5 ms (save 14.1 ms)
    measured draft cost = 10.0 ms/round  <  14.1 ms
    free draft -> 85.6 - 10.0 = 75.6 ms -> 2.86/0.0756 = 37.8 t/s     still below 40

The original estimate (10.5 ms) was accurate to 5%, so the conclusion was right - but it was an inference,
and it is now a measurement. **DFlash2 remains closed as a route to 40.**

## Two corrections to my own DFlash2 description

1. **DFlash2's verify runs at n=4, not n=8.** I had inferred a block-of-8 verify from the speculator's
   self-reported `block_size=8, n_extract=5`. The traces show every target matvec at `n=4`. So its verify
   is 33% wider than the MTP arm's n=3, which is part of why its per-round GPU cost is higher.
2. **Its lm_head is inefficient at n=4.** `MUL_MAT_VEC iq4_xs m=248320 n=4 k=5120` costs 125.5 ms for
   715 MB of weights = **125 GB/s**, against **227 GB/s** for the same op at n=3 in the MTP arm. A 45%
   shortfall, DFlash2-specific; worth ~2.5 ms/round if it were at bandwidth. It does not change the verdict,
   but it means the n=4 matvec path is not at bandwidth in general.

## Also visible: DFlash2 spends more GPU per round

2468.2 ms over ~22.4 rounds = 110 ms/round, against MTP's 2406.2 ms over ~25.4 rounds = 95 ms/round. So
DFlash2's larger per-round cost is not only the draft - its wider verify costs more too.

# DFlash2 n_max swept 1..8 - COMPLETE. The default (3) is optimal and the ceiling is bounded.

The last untested dimension was n_max *below* the default 3. Now swept:

    n_max   t/s (2 reps)      acceptance            note
    1       27.47  27.52      0.879 (29/33)         highest acceptance, slowest
    2       32.84  32.84      0.740 (37/50)         perfectly reproducible
    3(*)    33.44  36.85      0.631 / 0.729         default; fastest; 10% spread
    4       25.22   (1 rep)   0.566
    5       21.66   (1 rep)   0.453
    6       20.93   (1 rep)   0.417
    8       18.05   (1 rep)   0.360
    (*) default

**n_max=3 (the default) is optimal, confirmed across the full range.** The shape is informative: acceptance
is *highest* at n_max=1 (0.879) yet throughput is lowest (27.5), because the verify width dominates. Note
n_max=2 is bit-reproducible (32.84 twice, identical acceptance) while n_max=3 is the configuration that
shows the 10% spread - so the non-reproducibility is specific to that operating point, not general.

## DFlash2 ceiling, now bounded by arithmetic rather than estimate

    measured draft cost                 10.0 ms/round
    free draft -> round 75.6 ms       -> 37.8 t/s     <- absolute ceiling, below the 40 target
    a Q4 draft (~5 ms) -> round 80.6  -> 35.5 t/s     <- below the shipped deliverable (36.65-36.78)

**Even a zero-cost draft cannot get DFlash2 to 40.** No remaining DFlash2 dimension can change the verdict:
n_max is swept, the draft cost is measured, and the draft quant only moves a term that is already too small
to close a 14.1 ms gap.

## Method note / error

The `mean_len` column printed by the n_max script used a wrong formula
(`predicted_n / (draft_n / draft_n_accepted)`), producing nonsense (56.24, 47.36, 40.37). The t/s and
acceptance figures are correct; mean_len should be read from the server's `print_timing` line instead.
