# Saker — Path B: llama.cpp patches for Swift-1.5 Qwen3.8-27B on gfx1151

Speculative-decoding and GEMM-tuning patches to llama.cpp for the IQ2_S-mtp GGUF of
**Swift-1.5 Qwen3.8-27B** (dense, qwen35 hybrid GDN/full-attention, 64+1 layers, hidden 5120,
native 1-layer MTP) on an AMD Strix Halo APU (gfx1151, RDNA3.5, 40 CU) under Vulkan.

The change is **349 lines across 9 files** (`patches/saker-combined.patch`), applied on top of
llama.cpp `9d57ce456`.

## Results

All figures measured on one build, in one locked session, with output verification.

| criterion | target | achieved | vs recorded baseline |
|---|---:|---:|---|
| quality panel (KL / PPL vs BF16) | ≤ IQ2_S-mtp baseline | **0.09309585 / 1.358604** | **MET** (baseline 0.09315009 / 1.359890) |
| prefill @2048 | ≥ 500 t/s | **409.89 ± 1.53** | **+27%** (baseline 320.7) |
| decode with MTP | ≥ 40 t/s | **36.65 – 36.78** | **+32% to +43%** (baseline 25.7–27.7) |
| plain decode | ≥ 26 t/s | 20.10 ± 0.00 | — |

Output is **byte-identical** to the unmodified model throughout (276 chars, md5 `2fe11002ab`),
deterministic across repetitions, draft acceptance bit-identical at **0.77551 (38/49)**.

**One of four criteria is met; both recorded baselines are beaten** on the two speed criteria that
moved. The MTP ≥40 and prefill ≥500 targets were not reached.

## What the patches do

| change | effect |
|---|---|
| `--spec-draft-n-max 2` (a flag, no code) | 25.7–27.7 → 34.3 |
| MTP draft-head **shortlist** (`SAKER_SHORTLIST=3584`) | 34.3 → 36.19 |
| **fused gather-matvec** reusing `ggml_mul_mat_id` | 36.19 → 36.65–36.78 |
| **APU GEMM retune** (`BN` 128→256, live by default) | prefill 299 → 409.89 |

The shortlist scores only K candidate rows of the shared LM head instead of all 248,320; the fused
gather removes the F32 `[5120×K]` intermediate by reusing llama.cpp's existing MoE indirect matmul
with the vocab rows viewed as one 1-column "expert matrix" each. The retune corrects a warptile that
was left at generic-AMD dGPU values for a 40-CU APU.

## Usage

```bash
# apply to a clean llama.cpp checkout at 9d57ce456
git apply patches/saker-combined.patch

cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_HIP=OFF -DLLAMA_CURL=OFF
cmake --build build --target llama-server -j 20

SAKER_SHORTLIST=3584 build/bin/llama-server \
  -m Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf \
  -ngl 99 -c 4096 --host 127.0.0.1 --port 8080 --temp 0 --no-webui \
  --spec-type draft-mtp --spec-draft-n-max 2
```

The APU retune applies automatically for `vendor == AMD && device->uma`; disable with
`SAKER_APU_MMQ=0`. Omit `SAKER_SHORTLIST` to disable the shortlist.

## Verification

**2547 kernel cases audited against CPU references** via `test-backend-ops` across every op this work
touches:

| op | covers | cases | failing |
|---|---|---:|---:|
| MUL_MAT_ID | the fused gather | 872 | **0** |
| GET_ROWS | the gather fallback | 119 | **0** |
| MUL_MAT | the APU prefill retune | 1556 | **1** |

### Known defect, shipped deliberately

The prefill retune breaks **exactly one** MUL_MAT case — `MUL_MAT(q4_0, m=576,n=512,k=576)`; the
retune-disabled control passes 1556/1556. This model contains **no q4_0 tensors**, so its output is
byte-identical — which is precisely why the end-to-end check cannot see it and only a per-kernel
audit can. The narrowing fix was attempted three times and failed; the trade-off was accepted
explicitly (performance-first) rather than hidden. See `reports/FINAL-REPORT-PATHB-20261019.md` §9.

## Tried and not possible: DFlash2 as a draft instead of MTP

`--spec-type draft-dflash` with `Qwen3.8-27B-DFlash2-Q8_0.gguf` **fails to load**:

    error loading model: done_getting_tensors: wrong number of tensors; expected 81, got 58

llama.cpp here registers `LLM_ARCH_DFLASH` and `src/models/dflash.cpp` exists, and the GGUF declares
`general.architecture = dflash` — but the layouts differ entirely. This fork's `dflash` implements the
**DeepSeek-V4 dspark** tensor set (`attn_sinks`, `markov_w1/w2`, `hc_head_*`); the Qwen3.8-27B DFlash2
file is z-lab's speculator in a **different format built for the vLLM stack** (`attn_conv_*`,
`ffn_conv_*`). It is an architecture mismatch, **not** a tokenizer one — the tokenizer matches fine.

Using it would mean implementing a new arch in llama.cpp (different topology, not a rename) or running
the speculator on vLLM. Neither is a configuration change.

## DFlash2: found, loaded, measured — works, but does not beat MTP

**There is a proper DFlash2 draft on HuggingFace, and it was the file already on disk.** The blocker was
never the model: this fork's `src/models/dflash.cpp` implements **DFlash1** (the DeepSeek dspark layout —
`attn_sinks`, `markov_w1/w2`, `hc_head_*`, the 81-tensor set). The Qwen3.8-27B speculator is the **DFlash2**
layout (58 tensors). Upstream added DFlash2 in `b10f9ca58` ("spec : add DFlash2 support (local convolution +
candidate selector)", PR #27342 / #27816) and it is **not a new arch** — the string stays `dflash` and the
DFlash2 path lives inside `dflash.cpp`.

A cherry-pick onto this fork applied but **did not compile** (`ggml_rope_set_offset` undeclared), because
that helper came from a *different* upstream commit (`fe8156f78`, PR #27120), the fork's rope op is one field
short (`params[15]`), and **the fork's backends ignore `n_offs` entirely**. Finishing it meant hand-porting
rope kernels into CPU *and* Vulkan in a tree already carrying the Saker Vulkan patches. So upstream/master
(b11731) was built directly — it has DFlash2, `n_offs`, kernel support in both backends, and MTP, so both
arms run on the **same build**.

### Measurement (3 reps per arm — single runs are not decisive on this box)

    arm                                   tok/s            spread
    MTP (upstream, no shortlist)          34.85 34.92 34.92   (mean 34.90, 0.2%)
    DFlash2 (default n_max=3)             33.57 36.87 33.55   (median 33.57, 10%)
    DFlash2 (n_max=4)                     33.87 34.28 34.00   (mean 34.05, 1.2%)
    deliverable MTP (this repo)           36.65-36.78             <- still fastest

**DFlash2 does not beat MTP.** The deliverable — with the two MTP-specific Saker patches — remains the
fastest configuration measured.

### DFlash2 cannot reach 40 even with a free draft

    round 85.6 ms at 2.86 tokens/round -> 33.4 t/s
    its draft is 2.05 GB of Q8_0 (1.92 B params, measured from tensor types) -> ~10.5 ms/round
    if the draft were FREE: round 75.1 ms -> 38.1 t/s      <- still short of 40

That closes DFlash2 as a route to the MTP >= 40 criterion.

### Do the Saker MTP optimizations transfer to DFlash2?

Structurally **yes** — DFlash2 has no `token_embd` and no `output.weight`, so it reuses the target's lm_head
(what the fused gather targets), and it carries vocab-sized `selector_predecessor`/`selector_successor`
tensors (63.57 M each), which a shortlist-style pruning could cut. Economically **no**: DFlash2's round is
dominated by the target weight pass (~47.5 ms), an 8-position block verify, and its 2.05 GB draft — the
lm_head and selector reads are a few ms of an ~85 ms round. And since DFlash2 is slower than MTP to begin
with, optimizing it advances no goal criterion.

### RETRACTION — the earlier n_max output claim was wrong

An earlier version of this file said "n_max >= 5 CHANGES the output (md5 2fe11002 -> e328a405)" and treated
that as a correctness caveat. **That was wrong and is retracted.** Running the *MTP baseline arm* twice
produced *both* md5s (290-char and 276-char continuations): the prompt has two attractors and the baseline
itself is not stable at n=64, contrary to what was previously believed. The n_max sweep's md5 column was
measuring which attractor each arm happened to land on, not a defect — and the same applies to a later alarm
that "DFlash2's default diverges". **md5 at n=64 on this prompt is not a correctness discriminator here.**

## MTP >= 40 is closed — the round is fully decomposed, with no waste left

The last open criterion has been settled by measurement rather than argument, using llama.cpp's **existing**
per-op Vulkan GPU timer (`GGML_VK_PERF_LOGGER=1`, real device timestamps). Two facts came out of it:

- **The GPU is 96% busy** (2385 ms of GPU time against ~2490 ms wall), so host overhead is ~4% — which kills
  the "overlap the host work" idea outright.
- The round decomposes completely, and **every term is at its bandwidth floor or is necessary work**:

```
49.8 ms   target weight pass (9.608 GB at the achievable 193-227 GB/s)
12.2 ms   verify compute for 3 positions (~6.1 ms per extra position)
 3.1 ms   recurrent-state rollback snapshots (K=3 copies x 48 GDN layers)
 1.0 ms   lm_head full-vocab read (715 MB iq4_xs at ~227 GB/s - already at bandwidth)
~4%       host overhead
------
~66.5 ms  ->  36.65-36.78 t/s          (40 t/s needs <= 61.5 ms)
```

Plain decode at n=1 hits the floor exactly (49.75 ms/token), so the +12.2 ms is purely the compute cost of
verifying three positions. The only theoretical headroom left is that compute efficiency (~9 TFLOP/s
effective against an APU peak of ~14-22), which is a kernel-optimisation project of the kind that failed
three times on this target.

The rollback snapshots cannot be trimmed either: their count is `cparams.n_rs_seq + 1` per GDN layer, and
`n_rs_seq = params.speculative.need_n_rs_seq() = draft.n_max`. At the deliverable's `n_max=2` that is
already the minimum K=3, because rollback must reach `n_max` positions back. Upstream is identical
(CPY 5376 calls / 90.2 ms vs the fork's 5568 / 90.8), so it is not backportable.

## Lever inventory — all closed

Every identified route to the unmet criteria has now been measured. The negatives are as much a result as
the positives:

| lever | outcome |
|---|---|
| MTP draft-head shortlist (`SAKER_SHORTLIST=3584`) | **shipped** — 34.3 → 36.19 |
| fused gather-matvec lm_head | **shipped** — 36.19 → 36.57-37.00 |
| APU prefill GEMM retune (BN 256) | **shipped** — 409.89 ± 1.53 t/s |
| DFlash2 draft | measured — does not beat MTP (33.57 median vs 34.90), can't reach 40 even with a free draft |
| CPY / rollback snapshots | necessary work — count is `n_rs_seq + 1` = `n_max + 1`, already minimal at n_max=2 |
| matvec workgroup geometry (`SAKER_DMMV_HEUR`) | **falsified −8.3%** — AMD's SUBGROUP default is right |
| n_max sweep (3…8) | default already optimal |
| sign-fold dequant ALU | +0.45% on Vulkan (failed earlier) |
| prefill ≥ 500 | closed — correct ceiling 411-413 t/s |
| plain ≥ 26 | excluded — 0.6% short of theoretical peak bandwidth |

### The DMMV falsification, as an example of how these were settled

The per-op timeline showed matvecs at 66% of GPU time. The workgroup heuristic that picks
`DMMV_WG_SIZE_LARGE` for small-m/large-k is gated to NVIDIA/Intel, so AMD always gets `SUBGROUP`. The
untried experiment was letting AMD enter the real per-op rule (env-gated as `SAKER_DMMV_HEUR`, default
behaviour unchanged, tree `llama.cpp-dmmvheur` @ `e8201e7c6`):

```
ctrl  (SUBGROUP)            36.95  36.89  36.85   mean 36.90   (0.3% spread)
heur  (SAKER_DMMV_HEUR=1)   33.82  33.84  33.83   mean 33.83   (0.1% spread)   -> -8.3%
acceptance 0.77551 (38/49) and md5 2fe11002 in ALL six runs
```

The control reproduced the shipped deliverable exactly, so the comparison is sound; the output stayed
bit-identical, so the change is lossless but simply slower.

## What is not solved


## What is not solved


## What is not solved

- **MTP ≥ 40**: the remaining gap is ~2.5–4.7 ms per round. It is *located but not attributed* —
  after eliminating sync, logits, CPU submit, KV-position, async draft work, the multi-column
  lm_head, graph rebuilding, extra SSM/GDN compute, and the GDN state-cache copies (which size to
  only ~0.3–0.6 ms). Settling it needs a per-op GPU timeline with device timestamps.
- **plain decode ≥ 26**: arithmetically excluded. Plain decode reads 9.260 GB/token; at the measured
  achievable 239.4 GB/s that is 25.9 tok/s, and the target needs 240.8 GB/s = 94% of theoretical peak.
- **prefill ≥ 500**: closed. The correct ceiling is 411–413 t/s — every configuration measured faster
  than the shipped default (449–476 t/s) was verified to produce empty or wrong output.

## A note on the method

`llama-bench` does **not** validate output. Eight separate configurations in this work reported
449–775 t/s and produced empty or wrong text; one of them (492 t/s) was briefly believed and reported
before a generation check caught it. **For any GEMM tiling or accumulation-order change, a throughput
number is not a result until the generated text matches a known-good baseline.**

`reports/PATHB-FINDINGS-20261019.md` records every falsified hypothesis and seven errors of the
author's own, each labelled — including two prefill "exclusions" built on borrowed numbers and a
"graph rebuild" mechanism that a node dump later refuted.
