# Garbage reproduction attempt — 2026-10-20 (goal mvfpity1-yeykp2)

Reported symptom: the model emitting repeated-substring vomit, e.g.
`/terasterasterasteras…////`, observed in thinking during real use.

## Status: NOT REPRODUCED (27 generations, 4 conditions)

| condition | prompts | reps | garbage |
|---|---|---:|---|
| shipped (shortlist 3584), ctx 4096, `/completion` | think, long | 3 each | none |
| shipped, ctx 4096, chat template | think, long, code | 3 each | none |
| shipped, ctx **262144**, chat template, ~10k-token context | longctx | 3 | none |
| shipped, ctx 8192, chat template | think, long | 3 each | none |

Harness: `tools/garbage_repro.py` (self-test 7/7 on injected garble, so a zero reading is
trustworthy). Scores the RAW engine, never through the guard — the guard rewrites repetitive text
and would mask the symptom.

Two harness faults worth recording, both of which had produced misleading output earlier:

1. **The live server holds the gpu-lock flock**, so any benchmark arm started while it runs blocks
   on the lock, never binds, and reports `Connection refused`. That invalidated a whole A/B run
   (three arms "completed" with zero data). The harness now stops the live server first.
2. **`/completion` bypasses the chat template**, so the model never enters thinking mode — the exact
   regime where the reported garbage appeared. The harness now uses `/v1/chat/completions`.

## The real finding: the shortlist is NOT output-neutral

Same build, same prompts, temp 0, ctx 8192, chat template. md5 of the generated text:

| prompt | rep | generic | saker, shortlist OFF | saker, shortlist ON |
|---|---|---|---|---|
| think | 1 | 23ee24d0 | 23ee24d0 | **df8775c1** |
| think | 2 | bd8ad03d | bd8ad03d | **df8775c1** |
| think | 3 | 23ee24d0 | 23ee24d0 | **df8775c1** |
| long | 1 | c32b3298 | c32b3298 | **d90e3869** |
| long | 2 | d90e3869 | d90e3869 | **e57dfa21** |
| long | 3 | d90e3869 | d90e3869 | d90e3869 |

- **Without the shortlist the patched build is bit-exact against generic**, matching even rep-for-rep
  (23ee/bd8a/23ee). So the fused gather and the APU retune are output-neutral here, as claimed.
- **With the shortlist the output changes on 5 of 6.** The README claims "output is byte-identical
  to the unmodified model throughout" and the findings claim losslessness "by construction". Both
  are **false as stated**: the shortlist demonstrably changes which tokens are emitted.
- `get_logits_ith: invalid logits id` appears **only** with the shortlist (3 errors / 6 requests
  here; 6 / 9 in an earlier run) — so it is shortlist-gated, not per-request, and my earlier
  "once per request, pre-existing, benign" characterisation was wrong on both counts.
- The differing text is *coherent*, not garbage: same answer, different wording. So this is a
  correctness deviation, not yet the reported symptom.
- Note also that **generic is not deterministic across reps** at temp 0 (23ee vs bd8a), while
  saker-with-shortlist was stable (df87 x3). Another unexplained variance.

## Why this is the right lead

The shortlist narrows the draft head's candidate rows to the target's top-K. Draft tokens are
supposed to be *proposed only*, with verification deciding — so a narrower draft should cost
acceptance, never change the emitted token. It changes the token. That means the shortlist is
reaching the sampling decision, and a mechanism that can reword an answer can also tip the model
into a degenerate loop. That is the plausible path from this deviation to the reported vomiting.

Prime suspects, in order:
1. The narrow `[K]` logits readback + `-INFINITY` scatter in `src/llama-context.cpp` — the
   reconstructed restricted row may not agree with the candidate list the sampler sees.
2. `llama_set_mtp_candidates` timing: `cand_ids` refreshed from the target's logits at `i_end`, with
   `i_batch_end[0] >= 0 ? i_batch_end[0] : (n_tokens - 1)` — a wrong index yields a stale or
   mismatched candidate list for the draft that follows.
3. `get_logits_ith` failing on some requests (a logits read that could not be satisfied) — what the
   caller does with that failure matters.

## Reproduce this run

```bash
scp tools/garbage_repro.py <node A>:~/garbage_repro.py
# on node A (stops the live server first):
REPRO_CTX=8192 REPRO_NPREDICT=900 python3 garbage_repro.py probe saker-on SAKER_SHORTLIST=3584 think,long
REPRO_CTX=8192 REPRO_NPREDICT=900 python3 garbage_repro.py probe generic "" think,long
python3 garbage_repro.py selftest
```

Raw outputs: `~/talon-out/saker/decode-ilp-20261019/pathB-garbage/out-<arm>-<prompt>-rep<N>.txt`
