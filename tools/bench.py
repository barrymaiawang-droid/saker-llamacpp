#!/usr/bin/env python3
"""Benchmark and sanity-check a running Saker (llama.cpp) endpoint.

Deliberately decoupled from any local layout: point it at a URL, get JSON back. No model paths,
no build paths, no assumptions about how the server was started. That makes it safe for an agent
to run and easy for a human to read.

    ./tools/bench.py check  --base-url http://127.0.0.1:8080
    ./tools/bench.py bench  --base-url http://127.0.0.1:8080 --reps 3
    ./tools/bench.py bench  --base-url http://127.0.0.1:8080 --max-tokens 400 --json

`check`   verifies the endpoint answers, lists a model, generates sane text, and that /v1/models
          is reachable. Exit code 0 = healthy, 1 = problem. Suitable for CI.
`bench`   reports prefill (pp) and decode (tg) throughput from the server's own timings, plus draft
          acceptance when MTP speculation is enabled.

Measured on GPU. Treat numbers as machine-specific; compare like-for-like or not at all.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

PROMPT = ("Explain in a few sentences why 2/3 is greater than 3/5, then give one more example "
          "of comparing two fractions with different denominators.")


def call(base, path, payload=None, timeout=1800):
    url = base.rstrip("/") + path
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(url, json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else {}


def longest_repeat_run(text, max_period=64):
    """Longest run of a repeating unit, over non-overlapping blocks on every alignment.

    Comparing overlapping windows would count positions rather than blocks and turn a markdown
    separator into a false positive.
    """
    best, n = (0, 0), len(text)
    for period in range(1, max_period + 1):
        for offset in range(period):
            run, i = 1, offset
            while i + 2 * period <= n:
                if text[i:i + period] == text[i + period:i + 2 * period]:
                    run += 1
                    best = max(best, (run, period))
                else:
                    run = 1
                i += period
    return best


def generate(base, max_tokens, prompt, nonce=None):
    """One chat completion. Returns (content, reasoning, timings).

    `nonce` is prepended when measuring prefill: without it the server's prefix cache matches the
    previous rep and only a handful of tokens are actually prefilled, which makes pp meaningless
    (observed: 31.7 t/s "prefill" that was really a 4-token cache hit).
    """
    if nonce:
        prompt = "[run %s]\n%s" % (nonce, prompt)
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
    out = call(base, "/v1/chat/completions", body)
    choice = (out.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return (message.get("content") or "", message.get("reasoning_content") or "",
            out.get("timings") or {}, choice.get("finish_reason"))


def cmd_check(args):
    problems = []
    try:
        models = call(args.base_url, "/v1/models", timeout=60)
        count = len(models.get("data") or models.get("models") or [])
        print("models: %d listed" % count)
        if count == 0:
            problems.append("/v1/models returned no models")
    except Exception as exc:
        problems.append("/v1/models unreachable: %s" % exc)
        print("models: FAILED (%s)" % exc)

    started = time.time()
    try:
        content, reasoning, timings, finish = generate(args.base_url, args.max_tokens, args.prompt)
    except Exception as exc:
        problems.append("generation failed: %s" % exc)
        print("generate: FAILED (%s)" % exc)
        content = reasoning = ""
        timings, finish = {}, None
    if content or reasoning:
        blocks, period = longest_repeat_run(content + "\n" + reasoning)
        print("generate: %.1fs  content=%d chars  reasoning=%d chars  finish=%s"
              % (time.time() - started, len(content), len(reasoning), finish))
        print("output:   longest repeat run = %d x %d chars" % (blocks, period))
        if blocks >= 8 and blocks * period >= 240:
            problems.append("output looks degenerate (%d repeats of %d chars)" % (blocks, period))
        if not content.strip() and not reasoning.strip():
            problems.append("generation produced nothing")
    else:
        problems.append("generation produced no text")

    for problem in problems:
        print("PROBLEM: %s" % problem)
    print("RESULT: %s" % ("OK" if not problems else "%d problem(s)" % len(problems)))
    return 0 if not problems else 1


def bench_prompt(text, repeats=6):
    """Longer prompt so prefill is measured over enough tokens to be meaningful."""
    return "\n".join("%d: %s" % (i + 1, text) for i in range(repeats))


def cmd_bench(args):
    pp, tg, acc = [], [], []
    prompt = bench_prompt(args.prompt)
    for rep in range(1, args.reps + 1):
        content, reasoning, timings, finish = generate(
            args.base_url, args.max_tokens, prompt, nonce="%d-%d" % (rep, int(time.time() * 1000)))
        pps = timings.get("prompt_per_second") or 0
        tps = timings.get("predicted_per_second") or 0
        dn = timings.get("draft_n") or 0
        da = timings.get("draft_n_accepted") or 0
        # rep 1 includes slot warm-up; report it but exclude it from the summary when reps > 1
        warm = " (warm-up)" if rep == 1 and args.reps > 1 else ""
        print("rep%d  pp=%8.1f t/s  tg=%6.2f t/s  accept=%s%s"
              % (rep, pps, tps, ("%.5f" % (da / dn)) if dn else "n/a", warm))
        if not (rep == 1 and args.reps > 1 and args.drop_warmup):
            pp.append(pps)
            tg.append(tps)
            if dn:
                acc.append(da / dn)
    result = {
        "pp_t_s": round(statistics.mean(pp), 2) if pp else None,
        "tg_t_s": round(statistics.mean(tg), 2) if tg else None,
        "acceptance": round(statistics.mean(acc), 5) if acc else None,
        "reps_used": len(tg),
        "max_tokens": args.max_tokens,
        "base_url": args.base_url,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\nmean pp = %s t/s | mean tg = %s t/s | acceptance = %s (n=%d)"
              % (result["pp_t_s"], result["tg_t_s"], result["acceptance"], result["reps_used"]))
        print("note: pp is measured with a cache-defeating nonce and a repeated prompt, so each rep\n"
              "      prefills from scratch. pp scales with prompt length: a short prompt reads lower\n"
              "      than a 2048-token one. For the architecture-level figure use\n"
              "      llama-bench -p 2048 -n 0, and compare only like-for-like.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check", "bench"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8080",
                        help="OpenAI-compatible endpoint (default: %(default)s)")
    parser.add_argument("--reps", type=int, default=3, help="repetitions for bench (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=400, help="decode length (default: 400)")
    parser.add_argument("--prompt", default=PROMPT, help="prompt to use")
    parser.add_argument("--drop-warmup", action="store_true", default=True,
                        help="exclude rep 1 from the summary (default: on)")
    parser.add_argument("--json", action="store_true", help="machine-readable summary")
    args = parser.parse_args()
    return cmd_check(args) if args.command == "check" else cmd_bench(args)


if __name__ == "__main__":
    sys.exit(main())
