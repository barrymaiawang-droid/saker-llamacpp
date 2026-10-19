# Saker loop guard (`guard/saker_guard.py`)

A buffered reverse proxy for `llama-server` (Swift-1.5 / MTP) that stops repetitive
generation, compacts the repetitive tail, and continues the same answer. It also
provides the **show/hide-thinking** switch.

Standard library only. No dependencies, no GPU work of its own.

## Why buffered

The earlier Ember guard spliced a retry into an already-streaming response, which
left the looping text visible and could duplicate it. Here the guard reads the
upstream stream to completion first, rewrites it if necessary, and only then emits
one valid stream to the caller. Already-sent repetition cannot be unsent, so the
cost of this choice is that live token-by-token output is not available.

## Loop handling

| Situation | Action |
|---|---|
| Exact repetition (same unit ≥4×) | Delete duplicates, keep one copy. **No model call.** |
| Rambling: 6 near-identical sentences | One bounded summary call, then continue |
| Structured output (JSON/grammar/tool call/code fence) | **Never rewritten** — abort with `unsafe_compaction` |
| Still looping after `--max-recoveries` | Abort with `loop_recovery_exhausted` |
| Empty output with `finish_reason=length` | Abort with `empty_answer_length` (not a silent empty success) |

Recovery preserves the clean prefix byte-for-byte and always runs through
`continue_final_message`, so the model continues the same assistant message rather
than restarting it. `add_generation_prompt=false` and `cache_prompt=false` are set
on continuations, because the prefix is not guaranteed to extend one cached.

The compaction call reuses the **same loaded Swift/MTP model** with
`reasoning_effort:"none"`, a hard `max_tokens` cap and no tools. Keep it bounded or
loading/queueing costs will exceed what it saves. (MTP speeds that call up; it does
not make the MTP head a summarizer, and the DFlash2 draft cannot summarize at all —
it needs the target's hidden states.)

## Show / hide thinking

```bash
--hide-thinking   # default: reasoning omitted from the response
--show-thinking   # reasoning included
```

This controls **visibility only**. Reasoning still runs. Note that
`reasoning_format:"none"` is *not* a hide-thinking setting — it puts raw thoughts
into `content`. The guard forces `reasoning_format:"deepseek"` upstream so thinking
is separable, and refuses to return `content` that still contains
` thinking` markup when hiding (`unparsed_reasoning`).

## Usage

Put the real server on `--upstream` and point clients at the guard:

```bash
# 1. the engine (unchanged form apart from the port)
SAKER_SHORTLIST=3584 build/bin/llama-server \
  -m <target.gguf> -ngl 99 -c 4096 --host 127.0.0.1 --port 8081 \
  --temp 0 --no-webui --spec-type draft-mtp --spec-draft-n-max 2

# 2. the guard
python3 guard/saker_guard.py --port 18160 --upstream http://127.0.0.1:8081 --hide-thinking
```

Clients then call `http://127.0.0.1:18160/v1/chat/completions` with normal OpenAI
payloads. `Authorization` is forwarded upstream and never logged.

Useful flags: `--max-recoveries` (0–2), `--tail-chars` (512–16384),
`--summary-tokens` (16–512), `--timeout`, `--idle-timeout`, `--log-level`,
`--pidfile`.

## Safety and limits

- One guarded request at a time; extras get `503 guard_busy`. `continue_final_message`
  and slot reuse make concurrent guard traffic hard to reason about, so it is refused
  rather than silently interleaved.
- Aborting a loop **closes the upstream socket before** the summary/continuation
  call, so a single-slot engine is not left generating the abandoned task while the
  next one queues.
- Caller disconnect and total deadline are detected by a watchdog that closes the
  socket from another thread — a peer dribbling bytes cannot evade a per-read timeout.
- Upstream error bodies are not reflected, since they can contain prompt text.
- Thinking is not logged; log lines carry codes and counters only.
- `503`/error responses never include partial output from a failed attempt.

## Test status

`python3 -m unittest discover -s tests -v` — **17 tests, all passing**, covering detection,
exact vs rambling recovery, continuation request shape, both visibility settings, streaming
shape, tool-call assembly, cancellation, deadline, recovery exhaustion, structured-output
refusal, echoed-prefix collapsing, the budget-exhausted trap, and the no-leak paths.

### Live verification against the real engine (2026-10-19)

Run on node A under `gpu-lock.sh` with the shipped build, the IQ2_S-mtp target and
`--spec-type draft-mtp --spec-draft-n-max 2`. Guard bytes matched local before the run
(`md5 b3205c1233`). **All live checks passed.**

What live testing settled, including two things unit tests could not:

1. **The engine echoes the whole assistant message on `continue_final_message`.** Measured:
   sent prefix 21 chars, returned 57 chars, `completion_tokens=14` — so the returned content is
   prefix + new tokens, *not* just the new tokens. This was a real bug in the first implementation:
   the guard would have emitted the preserved prefix twice. `reconcile()` collapses it, and loop
   detection is re-derived against the reconciled text so the guard's own prefix can never
   masquerade as a loop. Unit tests only caught this after the fake upstream was taught the real
   behaviour.
2. **The template honours both continuation channels** — `content` and `reasoning_content`.
3. **Thinking really does arrive as `reasoning_content`**, so hiding is a display filter and not a
   behaviour change. Verified both ways on the same engine: hidden → no `reasoning_content`;
   shown → 190 chars of it.
4. **A forced loop is detected, bounded and stopped.** The guard's log recorded
   `loop recovered channel=content kind=exact attempt=1/2` and then `loop_recovery_exhausted` —
   exact repetition, so no summary call was needed.
5. **The budget trap is now reported.** Empty content with `finish_reason=length` used to reach the
   caller as a silent empty success — much more likely to be hit *because* thinking is hidden.
   An A/B against the engine directly proved this is pre-existing, not caused by the guard:
   ```
   engine direct  max_tokens=2000  finish=length  content=0   reasoning=4977
   engine direct  effort=none      finish=stop    content=964 reasoning=0
   guard          effort=none      finish=stop    content=964 reasoning=0   <- byte-identical to direct
   ```
   The guard now returns `empty_answer_length` with an actionable message instead.
6. **No false positives** on genuine long output (964 chars, zero recoveries).

### A limitation live testing exposed

Tail compaction **bounds** repetition; it does not guarantee a good continuation. With a prompt
that orders the model to repeat a sentence twelve times, the guard compacted 12 copies to 1 and
continued — after which the model wrote three more copies and stopped, returning no answer. The
repetition was bounded (12 → 3, one recovery logged), which is the guard's contract, but the
answer never arrived. Breaking a loop the model is *committed* to is a harder problem than
detecting it; the continuation nudge helps and is not sufficient on its own.
