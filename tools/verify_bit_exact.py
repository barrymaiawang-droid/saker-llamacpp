#!/usr/bin/env python3
"""Verify the Saker patch is OUTPUT-NEUTRAL: same prompt, same seeds, temp 0, byte-identical text.

Why this exists
---------------
The patch's central safety claim was "output is byte-identical to the unmodified model" and
"lossless by construction". A shortlist run showed that is FALSE: with SAKER_SHORTLIST set, the
generated text changed on 5 of 6 runs while the same build without it was bit-exact against
unpatched llama.cpp. That is exactly the kind of silent behavioural drift that can also surface as
degenerate output, so it needs to be a one-command check rather than a claim in a README.

What it does
------------
Runs the same prompts on N arms (generic / saker+noshortlist / saker+shortlist by default), saves
every output, and reports per prompt+rep whether each arm matches the reference byte for byte.

IMPORTANT measured caveat baked in: generic at temp 0 is NOT deterministic across reps
(observed 23ee/bd8a/23ee on one prompt). So a single-rep comparison is unsafe. This tool therefore
records every rep and reports both "matches the reference arm rep-for-rep" and "is the reference
arm even stable across its own reps" - if the reference is unstable, a mismatch proves nothing.

Usage (on node A, from the tree that holds the patched build):
    python3 verify_bit_exact.py                    # generic vs saker-noshortlist vs saker-shortlist
    python3 verify_bit_exact.py --arms generic,saker-shortlist --reps 5
    python3 verify_bit_exact.py --prompts think,code --ctx 8192

Stops the live dsh server first: it holds the gpu-lock flock, so arms started while it runs block
and never bind (which surfaces as Connection refused and silently yields zero data).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

MODEL = "/home/hycris/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf"
PATCHED = "/home/hycris/llama.cpp-qwen38-merge/build/bin/llama-server"
GENERIC = "/home/hycris/llama.cpp-qwen38/build/bin/llama-server"
LOCK = "/home/hycris/talon-out/saker/decode-ilp-20261019/gpu-lock.sh"
LAUNCHER = "/home/hycris/launch-saker.sh"
OUT = "/home/hycris/talon-out/saker/decode-ilp-20261019/pathB-bitexact"
PORT = 18260
ALIAS = "bx"

ARMS = {
    "generic":         (GENERIC, ""),
    "saker-noshortlist": (PATCHED, ""),
    "saker-shortlist": (PATCHED, "SAKER_SHORTLIST=3584"),
}

PROMPTS = {
    "think": ("Think carefully and at length, then answer. A train leaves at 3pm going 60mph; "
              "another leaves at 4pm going 80mph on the same route. When does the second catch the "
              "first? Show your reasoning in detail."),
    "long": ("Explain in detail, over several paragraphs, how speculative decoding works, what draft "
             "acceptance means, and why a shortlist of the draft head's candidate rows changes cost."),
    "code": ("Write a complete Python implementation of a red-black tree with insert, delete, search "
             "and in-order traversal, with docstrings and unit tests."),
}


def free_port():
    """The canonical launcher runs llama-server under gpu-lock.sh, which flocks for its lifetime."""
    subprocess.run(["bash", LAUNCHER, "stop"], capture_output=True, timeout=120)
    for _ in range(40):
        if subprocess.run(["pgrep", "-f", "llama-serve[r].*--port 18151"],
                          capture_output=True).returncode != 0:
            break
        time.sleep(1)
    subprocess.run(["pkill", "-f", "llama-serve[r].*--alias " + ALIAS], capture_output=True)
    for _ in range(30):
        if ":%d " % PORT not in subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout:
            return True
        time.sleep(1)
    return False


def run_arm(tag, binary, envfrag, prompt_keys, reps, ctx, npredict):
    os.makedirs(OUT, exist_ok=True)
    log = os.path.join(OUT, "srv-%s.log" % tag)
    cmd = ["env"] + (envfrag.split() if envfrag.strip() else []) + [
        LOCK, binary, "-m", MODEL, "-ngl", "99", "-c", str(ctx), "-ub", "256",
        "--host", "127.0.0.1", "--port", str(PORT), "--alias", ALIAS,
        "--temp", "0", "--no-webui", "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    with open(log, "wb") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=fh, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    for _ in range(180):
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=3):
                break
        except Exception:
            time.sleep(2)
    else:
        print("  FAIL: %s never became healthy" % tag)
        proc.kill()
        return {}

    results = {}
    try:
        for key in prompt_keys:
            for rep in range(1, reps + 1):
                body = {"messages": [{"role": "user", "content": PROMPTS[key]}],
                        "max_tokens": npredict, "temperature": 0.0, "stream": False}
                req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                                             json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=1800) as resp:
                        r = json.loads(resp.read())
                except Exception as exc:
                    print("  %-18s %-5s rep%d ERROR %s" % (tag, key, rep, exc))
                    continue
                msg = (r.get("choices") or [{}])[0].get("message") or {}
                text = (msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")
                digest = hashlib.md5(text.encode()).hexdigest()
                results[(key, rep)] = digest
                with open(os.path.join(OUT, "out-%s-%s-rep%d.txt" % (tag, key, rep)), "w") as fh:
                    fh.write(text)
                print("  %-18s %-5s rep%d %s  %d chars" % (tag, key, rep, digest[:8], len(text)))
    finally:
        proc.terminate()
        for _ in range(20):
            if proc.poll() is not None:
                break
            time.sleep(1)
        else:
            proc.kill()
        subprocess.run(["pkill", "-f", "llama-serve[r].*--alias " + ALIAS], capture_output=True)
        time.sleep(3)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="generic,saker-noshortlist,saker-shortlist")
    ap.add_argument("--prompts", default="think,long")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--npredict", type=int, default=900)
    ap.add_argument("--reference", default="generic")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not free_port():
        print("FAIL: could not free port %d" % PORT)
        return 2

    data = {}
    for tag in arms:
        if tag not in ARMS:
            print("unknown arm %r (have: %s)" % (tag, ", ".join(ARMS)))
            return 2
        binary, envfrag = ARMS[tag]
        print("\n=== arm %s ===" % tag)
        data[tag] = run_arm(tag, binary, envfrag, prompts, args.reps, args.ctx, args.npredict)

    ref = args.reference
    print("\n=== verdict, against reference arm %r ===" % ref)
    print("  a mismatch is only meaningful if the reference is stable across its own reps")
    problems = 0
    for key in prompts:
        ref_hashes = [data.get(ref, {}).get((key, r)) for r in range(1, args.reps + 1)]
        stable = len(set(h for h in ref_hashes if h)) <= 1
        print("  %-5s reference stable across reps: %s %s" % (
            key, "yes" if stable else "NO - comparison is unreliable for this prompt",
            "" if stable else "(%s)" % ", ".join((h or "-")[:8] for h in ref_hashes)))
        for tag in arms:
            if tag == ref:
                continue
            for r in range(1, args.reps + 1):
                a, b = data.get(ref, {}).get((key, r)), data.get(tag, {}).get((key, r))
                if a is None or b is None:
                    continue
                if a != b:
                    problems += 1
                    print("    %-18s %s rep%d MISMATCH  ref=%s got=%s" % (
                        tag, key, r, a[:8], b[:8]))
    print("\n  total mismatches: %d" % problems)
    print("  %s" % ("ALL ARMS BIT-IDENTICAL to the reference" if problems == 0
                    else "NOT output-neutral - do not describe this build as byte-identical"))
    print("  outputs: %s" % OUT)
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
