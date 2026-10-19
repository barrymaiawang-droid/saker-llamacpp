#!/usr/bin/env python3
"""Garbage metric + reproduction harness for the Saker Swift server.

Two hard-won design points:

1. Score the RAW ENGINE (port direct to llama-server), never through guard/saker_guard.py.
   The guard rewrites repetitive text, which would MASK the vomiting this is meant to measure.

2. Stop the live server first. The canonical launcher runs llama-server under gpu-lock.sh, which
   `flock`s for the process lifetime. An arm started while it runs blocks on the lock and never
   binds, which shows up as "Connection refused" and silently produces zero data. (That exact
   failure invalidated an earlier A/B run.)

Usage (on node A):
    python3 garbage_repro.py selftest
    python3 garbage_repro.py probe <tag> <env-assignments> [--prompts N]

`env-assignments` is a shell fragment such as "SAKER_SHORTLIST=3584" or "" (empty).
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

MODEL = "/home/hycris/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf"
BIN = "/home/hycris/llama.cpp-qwen38-merge/build/bin/llama-server"
GENERIC = "/home/hycris/llama.cpp-qwen38/build/bin/llama-server"
LOCK = "/home/hycris/talon-out/saker/decode-ilp-20261019/gpu-lock.sh"
LAUNCHER = "/home/hycris/launch-saker.sh"
OUT = "/home/hycris/talon-out/saker/decode-ilp-20261019/pathB-garbage"
PORT = 18240
ALIAS = "gbtest"
# The shipped config runs ctx 262144. Long context is a regime worth reproducing in: the reported
# garbage happened during real use, not on these short probes.
CTX = int(os.environ.get("REPRO_CTX", "4096"))
NPREDICT = int(os.environ.get("REPRO_NPREDICT", "1500"))

PROMPTS = {
    # long free-form thinking: where vomiting actually shows up
    "think": ("Think carefully and at length, then answer. A train leaves at 3pm going 60mph; another "
              "leaves at 4pm going 80mph on the same route. When does the second catch the first? "
              "Show your reasoning in detail."),
    "long": ("Explain in detail, over several paragraphs, how speculative decoding works, what draft "
             "acceptance means, and why a shortlist of the draft head's candidate rows changes cost."),
    "code": ("Write a complete Python implementation of a red-black tree with insert, delete, search "
             "and in-order traversal, with docstrings and unit tests."),
}


def build_prompt(key, approx_tokens=10000):
    """Static prompts plus a long-context one. The long-context case matters because the shipped
    server runs ctx 262144 and the reported garbage came from real use, not a short probe."""
    if key != "longctx":
        return PROMPTS[key]
    body = ("The measurement harness records prompt evaluation time separately from token generation, "
            "which lets one separate the bandwidth-bound weight pass from the compute-bound prefill. ")
    chunks = []
    i = 0
    while sum(len(c) for c in chunks) < approx_tokens * 4:
        i += 1
        chunks.append("Section %d records sample %d with measured value %d and a distinct note. %s"
                      % (i, i, (i * 7919) % 100003, body))
    chunks.append("\n\nUsing the sections above, explain in detail how the harness separates prompt "
                  "evaluation from generation, and why that distinction matters.")
    return "".join(chunks)


def worst_repeat_run(text, max_period=64):
    """Longest run of a repeating unit. Scans NON-OVERLAPPING blocks over every alignment, because
    comparing overlapping windows counts positions rather than blocks and turns a separator into a
    'loop'. Returns (blocks, period, start, span_chars)."""
    best = (0, 0, 0, 0)
    n = len(text)
    for period in range(1, max_period + 1):
        for offset in range(period):
            run_start, run = offset, 1
            i = offset
            while i + 2 * period <= n:
                if text[i:i + period] == text[i + period:i + 2 * period]:
                    if run == 1:
                        run_start = i
                    run += 1
                else:
                    run = 1
                if run > best[0]:
                    best = (run, period, run_start, run * period)
                i += period
    return best


def garbage(text, run_floor=4, span_floor=120):
    """A text is garbage if some unit repeats >= run_floor times covering >= span_floor characters.
    Also reports what fraction of the text sits inside such runs."""
    blocks, period, start, span = worst_repeat_run(text)
    covered = 0
    if span >= span_floor and blocks >= run_floor:
        covered = span
    ratio = covered / max(1, len(text))
    return {"garbage": covered > 0, "blocks": blocks, "period": period,
            "start": start, "span": span, "len": len(text), "ratio": round(ratio, 3),
            "longest_run": text[start:start + min(span, 70)]}


def selftest():
    print("=== metric self-test: injected garble MUST score, clean text MUST NOT ===")
    cases = [
        ('injected "teras"*40', "teras" * 40, True),
        ('injected "/"*1400', "/" * 1400, True),
        ('injected vomit then prose', "teras" * 50 + " and that is the answer.", True),
        ('injected "abc"*60 + "xyz"', "abc" * 60 + "xyz", True),
        ("clean prose", "The train catches the other at 7pm. " + "We can verify by equating distances. " * 3, False),
        ("numbered progress", "\n".join("Sample %d has a distinct verified result to preserve." % i
                                        for i in range(1, 12)), False),
        ("markdown rule separator", "-" * 80, False),
    ]
    bad = 0
    for name, text, expect in cases:
        g = garbage(text)
        ok = g["garbage"] == expect
        bad += 0 if ok else 1
        print("  [%s] %-30s garbage=%-5s span=%-5d period=%-2d" % (
            "PASS" if ok else "FAIL", name, g["garbage"], g["span"], g["period"]))
    print("\n  %s" % ("SELF-TEST PASSED - a zero reading is now trustworthy" if not bad
                      else "SELF-TEST FAILED (%d) - fix the metric before trusting any run" % bad))
    return 0 if not bad else 1


def stop_live():
    """Free the GPU lock and the port. Without this every arm silently blocks on flock."""
    subprocess.run(["bash", LAUNCHER, "stop"], capture_output=True, timeout=120)
    for _ in range(40):
        if subprocess.run(["pgrep", "-f", "llama-serve[r].*--port 18151"],
                          capture_output=True).returncode != 0:
            break
        time.sleep(1)
    subprocess.run(["pkill", "-f", "llama-serve[r].*--alias " + ALIAS], capture_output=True)
    for _ in range(30):
        out = subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout
        if ":%d " % PORT not in out:
            return True
        time.sleep(1)
    return False


def start_arm(binary, env_fragment):
    os.makedirs(OUT, exist_ok=True)
    log = os.path.join(OUT, "srv-arm.log")
    cmd = ["env"] + (env_fragment.split() if env_fragment.strip() else []) + [
        LOCK, binary, "-m", MODEL, "-ngl", "99", "-c", str(CTX), "-ub", "256",
        "--host", "127.0.0.1", "--port", str(PORT), "--alias", ALIAS,
        "--temp", "0", "--no-webui", "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    with open(log, "wb") as fh:
        subprocess.Popen(["setsid"], stdout=fh, stderr=fh,
                         stdin=subprocess.DEVNULL).wait() if False else None
        proc = subprocess.Popen(cmd, stdout=fh, stderr=fh, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    for _ in range(180):
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=3):
                return proc, log
        except Exception:
            time.sleep(2)
    return None, log


def stop_arm(proc):
    for sig in ("terminate", "kill"):
        try:
            getattr(proc, sig)()
        except Exception:
            pass
        for _ in range(20):
            if proc.poll() is not None:
                break
            time.sleep(1)
        if proc.poll() is not None:
            break
    subprocess.run(["pkill", "-f", "llama-serve[r].*--alias " + ALIAS], capture_output=True)
    time.sleep(3)


def probe(tag, binary, env_fragment, prompt_keys, reps=3):
    print("=== arm %s  binary=%s  env=%r ===" % (tag, os.path.basename(os.path.dirname(
        os.path.dirname(os.path.dirname(binary)))), env_fragment))
    if not stop_live():
        print("  FAIL: could not free port %d" % PORT)
        return
    proc, log = start_arm(binary, env_fragment)
    if proc is None:
        print("  FAIL: arm never became healthy; log tail:")
        print("\n".join(open(log, errors="replace").read().splitlines()[-8:]))
        return
    try:
        for key in prompt_keys:
            for rep in range(1, reps + 1):
                # MUST go through the chat template: /completion bypasses it, so the model never
                # enters thinking mode - and thinking is exactly where the reported garbage appeared.
                body = {"messages": [{"role": "user", "content": build_prompt(key)}],
                        "max_tokens": NPREDICT, "temperature": 0.0, "stream": False}
                req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                                             json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=1800) as resp:
                        r = json.loads(resp.read())
                except Exception as exc:
                    print("  %-5s rep%d ERROR %s" % (key, rep, exc))
                    continue
                t = r.get("timings", {})
                msg = (r.get("choices") or [{}])[0].get("message") or {}
                text = (msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")
                g = garbage(text)
                dn = t.get("draft_n") or 0
                da = t.get("draft_n_accepted") or 0
                print("  %-5s rep%d t/s=%-6.2f accept=%-8s GARBAGE=%-5s span=%-5d period=%-2d %s" % (
                    key, rep, t.get("predicted_per_second") or 0,
                    ("%.5f" % (da / dn)) if dn else "n/a",
                    g["garbage"], g["span"], g["period"],
                    ("run=%r" % g["longest_run"][:40]) if g["garbage"] else ""))
                # Always save the text: the decisive test is whether generic and saker produce
                # BYTE-IDENTICAL output at temp 0. The patch claims to be lossless, so any difference
                # is corruption whether or not it looks like garbage.
                with open(os.path.join(OUT, "out-%s-%s-rep%d.txt" % (tag, key, rep)), "w") as fh:
                    fh.write(text)
                    fh.write("\n@@MD5 " + hashlib.md5(text.encode()).hexdigest() + "\n")
        errs = open(log, errors="replace").read().count("get_logits_ith")
        print("  get_logits_ith errors in log: %d" % errs)
    finally:
        stop_arm(proc)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    if sys.argv[1] == "selftest":
        return selftest()
    if sys.argv[1] == "probe":
        tag = sys.argv[2]
        envfrag = sys.argv[3] if len(sys.argv) > 3 else ""
        keys = sys.argv[4].split(",") if len(sys.argv) > 4 else ["think", "long"]
        print("  ctx=%d n_predict=%d" % (CTX, NPREDICT))
        binary = GENERIC if tag.startswith("generic") else BIN
        probe(tag, binary, envfrag, keys)
        return 0
    print("unknown command %r" % sys.argv[1])
    return 2


if __name__ == "__main__":
    sys.exit(main())
