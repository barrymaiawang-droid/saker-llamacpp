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

`python3 -m unittest discover -s tests -v` — 14 tests, all passing, covering
detection, exact vs rambling recovery, continuation request shape, both visibility
settings, streaming shape, tool-call assembly, cancellation, deadline, recovery
exhaustion, structured-output refusal, and the no-leak paths.

**Not yet verified against a live `llama-server`.** The proxy logic, request shape
and failure paths are covered by tests against a fake upstream; what remains is the
real template interaction — that this model's chat template honours
`continue_final_message` and that its thinking arrives as `reasoning_content`. Until
that run happens, treat those two as assumptions.
