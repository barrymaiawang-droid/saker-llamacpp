# Handoff: hybrid-model caching + Strix Halo decoding — future directions

Two external write-ups, condensed, mapped onto **our** target, with evidence status marked.
Nothing here is measured on our box unless it says so. Sources:

- **A** — [How Gufo caches a hybrid model](https://gufo.blog/blog/how-gufo-caches-a-hybrid-model) (prefix caching for hybrid LLMs)
- **B** — [How strixite got fast](https://github.com/shawnshekari/strixite/blob/main/docs/how-it-got-fast.md) (single-engine Strix Halo)

Both are **the same hardware class as ours** (Strix Halo, gfx1151, 128 GB unified), which is why they are
worth reading at all. Neither is running our model.

---

## 1. Article A — why hybrid models cache differently

A hybrid model has two kinds of layer, and they behave nothing alike:

| layer kind | state left by prefill | can it be rewound? |
|---|---|---|
| attention | **append-only KV rows**, one per token, never change | yes — truncate the tail |
| recurrent (**Gated DeltaNet**, as in Qwen / Flash-Next) | **fixed-size private state**, updated in place | **no** — no way back to an earlier token |

Consequences A establishes:

1. **Prefix caching can only hit at exact boundaries.** You cannot keep 7,000 of 9,000 tokens and drop the
   rest, because the recurrent state summarises everything processed. So the cache stores **checkpoints**:
   a copy of the private state at chosen boundaries.
2. **Checkpoints borrow, they do not copy.** Rows stay in the slot; a checkpoint references them. If another
   conversation takes that slot, *"the checkpoints that borrowed them would point at garbage"* — so borrowed
   rows must be **preserved** (copied to backing RAM) before any overwrite.
3. **Sharing is by chunk**, not by copy. Rows are split into fixed-size ranges that never change once full;
   a checkpoint = references to full chunks + its own private state + a tail for the last partial chunk. So an
   extra checkpoint costs roughly **its private state, not its depth**.
4. **Checkpoint placement is a policy**: always at end-of-prompt, at the end of reused history, and at the
   "stable boundary"; spaced ~2,048 tokens; bounded list; evicted by reuse likelihood. Capturing is not free —
   the recurrent state exists only *between* prefill passes, so a checkpoint forces a pass to stop there.

Measured (Flash-Next, their box): 133 MB private state per checkpoint, 27,460 bytes per row, a checkpoint at
95,434 tokens held 2.75 GB. Time-to-first-token: from-scratch prefill **57.8 s** vs restoring a checkpoint
**1.38 s** from disk / **0.24 s** from RAM / **0.19 s** still resident — 40–300×.

## 2. Article B — where the milliseconds actually go

Qwen3.8-Flash-Next (~180B MoE) at ~47–50 t/s, flat from empty context to 492k tokens, on one Strix Halo.

Model of the whole game: **tokens/s ≈ bandwidth ÷ bytes read per token**, against a measured ~226 GB/s read
peak. Gains, in the order they mattered:

| step | effect |
|---|---|
| matrix units, BF16 in / FP32 accumulate | 141 → 652 t/s |
| weights as scaled codes, expert tiles of 64, n-gram reads overlapped with GPU work | → ~1,000 |
| bigger prefill chunks 2k → 4k → 8k (weights read once per chunk) | +8%, +5% |
| indexer: one-pass filter top-k instead of full scoring at depth | 99 → 44 ms/layer at 471k, +6.6% on a 490k conversation |

Speculative decoding, and this is the part that matters most to us:

- drafts 0 / 1 / 3 / 4 → **31.5 / 43.6 / 51.8 / 53.4 t/s**; a 5th draft helped again (−0.6 to −0.7% per token)
- code drafts far better than prose (59–71 vs 41–54 t/s; 94% of first drafts right on code)
- **confidence gate** — stop drafting when the draft head's top-2 margin is small, instead of spending a
  check on a coin flip
- **smaller draft vocabulary** — the draft head scores only the 65,536 most frequent token ids, draft
  3.9 → 1.7 ms
- **free rejection** — the check writes the recurrent state to a spare copy, so keeping drafts is a pointer
  swap and rejecting costs nothing

Quantisation, chosen class by class against full precision: mean KL 0.028, top-1 agreement 226/242. An
all-4-bit layout was ~25% faster per token but agreed only 194/242 — **the 8-bit dense path is where accuracy
lives**. Moving 4-bit groups 64 → 128 kept the same accuracy for 5.6% fewer bytes (1.9% less time/token).

Time per token at 64k (32.9 ms total): dense 8-bit projections incl. LM head 10.64 · routed experts 7.09 ·
hyper-connection mixes 4.13 · 4-bit GDN input projections 3.94 · delta rule/routers/attention/rest 3.92 ·
**host turnaround and gaps between ~1,130 launches 3.18**.

---

## 3. Directions for us, ranked

Our target: Swift-1.5 **Qwen3.8-27B IQ2_S-mtp**, qwen35 **hybrid GDN + full attention**, 64+1 layers,
hidden 5120, native MTP head, Vulkan on gfx1151. Current shipped config: `SAKER_SHORTLIST=3584`,
`-ub 256`, `--spec-type draft-mtp --spec-draft-n-max 2`, `-c 262144`.

### D1 — Audit recurrent rollback depth (`n_rs_seq`). **Top suspect for the garbage.** — NOT TESTED
Our code has the exact machinery A and B describe, and it is conditional:

```
src/llama-context.cpp:105  n_rs_seq > 0 && !llm_arch_supports_rs_rollback(arch)
                             -> "clamping to 0"
server-context.cpp:1053    cparams_dft.n_rs_seq = 0      (draft context: no rollback)
server-context.cpp:2987    draft.size() > llama_n_rs_seq(ctx_tgt)
                             -> "forcing full prompt re-processing ... (likely due to SWA or hybrid/recurrent)"
```

With `n_max 2`, rejecting drafts requires rolling back recurrent state. If depth is insufficient or silently
clamped, the GDN state stays mutated and the model continues from a corrupt summary — which is what repeated
nonsense output looks like. **It would only trigger when drafts are actually rejected**, i.e. low acceptance.
Measured acceptance on our prose prompt was 0.48–0.53 versus 0.73 on the arithmetic prompt — consistent.

First step: log `llama_n_rs_seq()` for target and draft at load; confirm it is not clamped to 0 for this arch.
If it is, that is the root cause and the fix is to raise it (or disable MTP as the mitigation).

### D2 — Run one slot (`-np 1`). — NOT TESTED
A names cross-slot row borrowing as a garbage vector. We run `n_slots = 4`, `kv_unified = true`. A single slot
removes slot hand-off entirely. Costs concurrency the model cannot serve anyway (MTP peak ~40 t/s, one user).

### D3 — Check `--swa-full`. — NOT TESTED
The server path itself names SWA alongside hybrid/recurrent as a cause of forced re-processing. If our model
uses a sliding window anywhere, the full-size SWA cache avoids that path.

### D4 — Replace the dynamic logits shortlist with a **static frequent-token draft vocabulary**. — STRONGEST IDEA
B does exactly what our shortlist does, but **statically**: "the draft head scores only the 65,536 most
frequent token ids — 3.9 → 1.7 ms". Ours takes the *dynamic top-K of the target's logits*, and we measured
that it **changes the emitted output on 5 of 6 runs** — so it is not draft-side only, and the README's
"byte-identical" claim is false as written. A static draft vocabulary cannot touch the target's sampling, so
it should keep the speed win and remove the correctness deviation at the same time.

### D5 — Confidence gate on the draft head's top-2 margin. — NOT TESTED
Cheap, draft-side, and directly targets our weak spot: prose acceptance ~0.48–0.53 means many checks are spent
on coin flips. B's version: stop drafting when the margin is small.

### D6 — Tune hybrid checkpointing instead of re-prefilling. — NOT TESTED
`n_ctx_checkpoints` / `checkpoint_min_step` exist in our build. A's numbers (57.8 s → 0.19–1.38 s) are for a
400k prompt on a bigger model, so not directly comparable, but the *mechanism* is the one to get right before
chasing speed. Correctness first: a checkpoint restored at a non-exact boundary is a garbage source.

### D7 — KV precision. — PARTIALLY KNOWN, probably leave alone
B: the 8-bit dense path is where accuracy lives; 4-bit groups 64 → 128 = 5.6% fewer bytes, 1.9% less time.
Our own earlier study found **q8_0 KV flips 2.16% of top tokens** with a 99.9th-percentile KLD of 0.81 against a
mean of 0.0126 — a thin but real tail. At 256k ctx our KV is the dominant memory cost (28.02 GiB GTT), so this
is tempting, but it trades away exactly the tail that agentic work uses.

### D8 — Host launch overhead is worth a look. — RELEVANT TO PRIOR FINDINGS
B spends **3.18 ms of 32.9 ms (10%)** at 64k on "host turnaround and gaps between ~1,130 launches". Our own
per-op trace recorded **2,328 GPU ops per round** with ~2.3 µs of launch cost each — same phenomenon, and B's
number suggests it is not negligible. Any op-count reduction (fusion) attacks this directly.

---

## 4. What does **not** transfer

- **MoE expert tiling / "only open the chapters you need"** — B's model uses ~2% of itself per token. Our 27B
  is dense: the whole weight set is read per token regardless. This is why B's biggest wins do not apply.
- **Bigger prefill chunks (+8%, +5%)** — the win is reading weights once per chunk, which needs MoE sparsity.
  Our own sweep went the *other* way: `-ub 256` beat 512 because of the APU retune. Opposite economics.
- **The indexer / top-512 block attention** — that is B's model architecture, not a tuning knob.
- **180B / 66 GiB weights** — our target is 8.95 GiB. Bandwidth headroom and failure modes differ.

## 5. Standing cautions

- **Never score through the guard.** It rewrites repetitive text and will mask the symptom being measured.
- **The live server holds the `gpu-lock` flock.** Benchmark arms started while it runs block, never bind, and
  surface as `Connection refused` — this silently produced three arms of zero data once.
- **`/completion` bypasses the chat template**, so the model never enters thinking mode. Use
  `/v1/chat/completions` when investigating thinking-path bugs.
- **Generic llama.cpp at `9d57ce456` is not fully deterministic across reps** (observed 23ee/bd8a/23ee at
  temp 0). Any single-run comparison is unsafe; compare rep-for-rep or use ≥3 reps.

## 6. References

- Marconi: *Prefix Caching for the Era of Hybrid LLMs* — https://arxiv.org/abs/2411.19379
- *Gated Delta Networks: Improving Mamba2 with Delta Rule* — https://arxiv.org/abs/2412.06464
- *Hybrid Models Meet SGLang* — https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/
- llama.cpp server slots + context checkpoints — https://github.com/ggml-org/llama.cpp/blob/41abbfd59/tools/server/server-context.cpp
- Unified Radix Cache (SGLang) — https://www.lmsys.org/blog/2026-08-11-unified-radix-cache
