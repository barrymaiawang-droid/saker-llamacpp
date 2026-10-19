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
