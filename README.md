# Saker — tuned MTP inference for Swift-1.5 Qwen3.8-27B on a Strix Halo APU

Patches for `llama.cpp` (Vulkan) that make **native MTP speculative decoding** fast on an AMD
gfx1151 APU, plus a drop-in **loop guard** that stops a model stuck repeating itself and continues
the answer.

Target: **Swift-1.5 Qwen3.8-27B** (dense, qwen35 hybrid GDN + full attention, 64 trunk layers + 1
MTP/nextn layer, hidden 5120), IQ2_S-mtp GGUF, unmodified.

Three things, each measured on the same machine, same GGUF, same prompt, 3 reps:

| | this project | unpatched llama.cpp | |
|---|---:|---:|---|
| **prefill @2048** | **402.82 t/s** (±0.14) | 299.68 | **+34.4%** |
| **decode, native MTP** | **36.8 – 40.7 t/s** | 35.8 – 40.7 | **up to +2.7%**, workload-dependent |
| **quality vs BF16** | KL **0.09309585** / PPL **1.358604** | IQ2_S baseline 0.09315009 / 1.359890 | **equal or better** |

Plus a loop guard with 25 tests that stops repetition, compacts it, and continues the answer instead
of returning nothing.

---

## Merits

### 1. Prefill is 34% faster — from one corrected tile

A warp tile left at generic-AMD dGPU values was starving a 40-CU APU. BN 128→256 with
`wg_denoms {128,256,1}` fixes it:

```
prefill @2048 tokens, -ub 256
  this project   402.82 ± 0.14 t/s
  unpatched      299.68 ± 0.10 t/s     (+34.4%)
```

Reproducible to ±0.14 t/s. It applies automatically on AMD UMA (`SAKER_APU_MMQ=0` disables it).

**Pin `-ub 256`.** The retune is tuned for that micro-batch; llama-server's default is 512, where
the tile is not the right shape. This is a launch requirement, not a suggestion — see
[Install](#install).

### 2. MTP decode, tuned — and it beats DFlash2

Native MTP already helps; three changes make it cheaper. The MTP draft head scores the shared LM
head, which is 248,320 rows wide, for a handful of candidate tokens per step. That is the cost.

- **Shortlist** (`SAKER_SHORTLIST=3584`) — score only the target's top-K candidate rows instead of
  all 248,320. K=3584 is the knee: acceptance matches the full head there, K=3072 loses acceptance,
  K=4096 buys nothing for more traffic.
- **Fused gather** — the shortlist's `ggml_get_rows` emitted an F32 `[5120 × K]` intermediate
  (~146 MB per draft at K=3584) while the gathered weights are only ~9.7 MB. Reusing
  `ggml_mul_mat_id` with the head viewed as one 1-column "expert matrix" per vocab row removes the
  intermediate entirely.
- **Verify width** `--spec-draft-n-max 2` — measured optimum for this target.

**Measured against DFlash2 on the same target, same box:**

| decode | tok/s | second model |
|---|---:|---|
| **this project, native MTP** | **36.8 – 40.7** | none |
| upstream MTP, unpatched | 35.8 – 40.7 | none |
| DFlash2 (Q8_0 drafter, `n_max 3`) | 33.4 – 36.9 | **2.05 GB** |

MTP wins on every axis that matters here: **same or better decode with no second model to download
or load**, one flag instead of a drafter, and a lower memory floor. DFlash2 is a good idea that needs
a 2 GB Q8_0 speculator and is capped below this project's ceiling even if the drafter were free.

### 3. Anti-looping that returns an answer

A model stuck in a repetition loop normally costs you the turn. This project ships a buffered
reverse proxy that catches it, removes the repetition, and continues:

```bash
python3 guard/saker_guard.py --port 18152 --upstream http://127.0.0.1:18151 --show-thinking
```

- **Detects real vomiting, not formatting.** Finds a repeated unit anywhere in the recent window —
  not merely at the end of the text — so `"teras"*40` followed by a run of `/` is caught, while a
  markdown rule or a table separator is not.
- **Removes it instead of truncating.** Exact repeats are deduplicated with no model call; rambling
  gets one bounded summary.
- **Recovery ladder:** continue the same message → restart with the progress as context → return the
  partial flagged `incomplete`. You always get either a usable answer or an explicit signal.
- **The operator nudge.** Both retries inject `--nudge`, defaulting to the short reminder you would
  type by hand ("You are looped. Stop repeating and continue the answer."). Measured: a long
  instruction made the model answer *the instruction*; the short one made it answer *the question*.
- **Buffered by design.** The response is assembled before it is sent, so a loop can be cut and
  rewritten without the caller ever seeing the repetition. The trade is no live token-by-token
  streaming through the guard; port-direct access to the engine still streams normally.
- **A switch for thinking.** `--show-thinking` / `--hide-thinking` controls visibility only —
  reasoning still runs.
- **Safe by construction.** Structured output (JSON, grammar, tool calls, code fences) is never
  rewritten. `/v1/models`, `/health`, `/props` and `/metrics` pass through so clients can discover
  the server; other generation routes are refused rather than run unguarded. `--max-concurrent`
  (default 4) matches the engine's slot count.

25 tests: `python3 -m unittest discover -s tests -v`

---

## Install

Requires: a Vulkan-capable build environment, the IQ2_S-mtp GGUF, and llama.cpp **at commit
`9d57ce456`** (the patch base). Nothing else — no Python packages, the guard is stdlib only.

```bash
# 1. clean llama.cpp at the patch base
git clone https://github.com/ggml-org/llama.cpp llcpp
cd llcpp && git checkout 9d57ce456

# 2. apply the patch (349 lines, 9 files)
git apply /path/to/patches/saker-combined.patch

# 3. build
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
      -DGGML_HIP=OFF -DLLAMA_CURL=OFF
cmake --build build --target llama-server -j "$(nproc)"

# 4. serve. NOTE -ub 256: the prefill retune is tuned for that micro-batch
SAKER_SHORTLIST=3584 build/bin/llama-server \
  -m /path/to/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf \
  -ngl 99 -c 16384 -ub 256 --host 127.0.0.1 --port 18151 \
  --temp 0 --no-webui --spec-type draft-mtp --spec-draft-n-max 2

# 5. optional: guard in front (loop recovery + thinking switch)
python3 guard/saker_guard.py --port 18152 --upstream http://127.0.0.1:18151 --show-thinking
```

Then talk to it as any OpenAI-compatible endpoint:

```bash
curl -s http://127.0.0.1:18151/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":256,"temperature":0}'
```

### Knobs

| env | default | effect |
|---|---|---|
| `SAKER_SHORTLIST` | unset (off) | K candidate rows for the MTP draft head; **3584 recommended** |
| `SAKER_APU_MMQ` | unset (on) | `0` disables the APU prefill retune |

---

## Benchmark

`tools/bench.py` measures a **running** endpoint and emits JSON. No model paths, no build paths.
Point it at a URL.

```bash
# is it healthy? exit 0 = yes, 1 = problem  (CI-friendly)
./tools/bench.py check --base-url http://127.0.0.1:18151

# throughput, 3 reps, rep 1 treated as warm-up
./tools/bench.py bench --base-url http://127.0.0.1:18151 --reps 3
./tools/bench.py bench --base-url http://127.0.0.1:18151 --reps 3 --json
```

`bench` reports prefill (pp), decode (tg) and draft acceptance from the server's own timings, using
a repeated prompt plus a per-rep nonce so every rep prefills from scratch. Observed on gfx1151:

```
mean pp ≈ 300-320 t/s | mean tg ≈ 37-41 t/s | acceptance ≈ 0.78-0.90
```

`pp` **scales with prompt length** — `bench` uses a few hundred tokens, so it reads lower than the
2048-token figure in [Merits](#1-prefill-is-34-faster--from-one-corrected-tile) (402.82), which is
`llama-bench -p 2048`. Compare like-for-like: same prompt length, same tool.

Notes for machine-to-machine comparison:

- **Fix one prompt and one `--max-tokens`**, and change one variable at a time.
- **Decode is workload-dependent.** Throughput tracks draft acceptance, which moves with the prompt
  (≈0.78 and ≈0.90 were both observed on the same server). Report acceptance next to every number.
- **Discard the first request.** It includes slot warm-up and can read ~2.4× slower.
- **Prefill numbers from a cached prompt are meaningless** — a warm run only prefills the new
  tokens. Use `check`, or a fresh prompt per measurement.
- For architecture-level numbers (matvec kernels, APU bandwidth), use `llama-bench` from the same
  build so the comparison shares a tool: `build/bin/llama-bench -m model.gguf -p 2048 -n 0 -ngl 99 -r 3 -ub 256`.

---

## Layout

```
patches/saker-combined.patch   the change: 349 lines, 9 files, over llama.cpp 9d57ce456
patches/saker-lmhead.patch     optional LM-head hoist variant, kept for reference
guard/saker_guard.py           loop guard + thinking switch (stdlib only)
tests/test_saker_guard.py      25 tests
tools/bench.py                 endpoint benchmark + health check (JSON)
docs/LOOP-GUARD.md             guard behaviour in detail
```

The patch touches: `common/speculative.cpp`, `ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
`include/llama.h`, `src/llama-context.cpp`, `src/llama-model.{cpp,h}`, `src/models/qwen35.cpp`,
`tools/llama-bench/llama-bench.cpp`, `tools/server/server-context.cpp`.

### Regenerating the patch

The published patch is a pathspec-limited diff, not a raw branch diff:

```bash
git diff 9d57ce456..HEAD -- \
  common/speculative.cpp ggml/src/ggml-vulkan/ggml-vulkan.cpp include/llama.h \
  src/llama-context.cpp src/llama-model.cpp src/llama-model.h src/models/qwen35.cpp \
  tools/llama-bench/llama-bench.cpp tools/server/server-context.cpp \
  > patches/saker-combined.patch
```
