#!/usr/bin/env python3
"""Exercise real server graph construction and cache reuse with random tiny Qwen35 weights."""
import argparse
import os
from pathlib import Path
import runpy
import socket
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--server", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--log-dir", type=Path, required=True)
parser.add_argument("--expect-mtp-acceptance", choices=["any", "some", "zero"], default="any")
args = parser.parse_args()
args.log_dir.mkdir(parents=True, exist_ok=True)
helpers = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/benchmark-sparse.py"))
request, wait_ready = helpers["request"], helpers["wait_ready"]

for mode in ["dense", "dense-mtp", "representative", "representative-mtp", "quest", "quest-mtp", "sparq", "sparq-mtp"]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["SAKER_SPARSE_MODE"] = mode.removesuffix("-mtp") if not mode.startswith("dense") else "representative"
    env["SAKER_SPARSE_DENSE_LAYERS"] = "0"
    env["SAKER_SPARQ_CHANNELS"] = "8"
    env["SAKER_SPARQ_RECENT_TOKENS"] = "256"
    env["SAKER_SPARQ_SINK_TOKENS"] = "16"
    env["SAKER_SPARSE_TOKENS"] = "0" if mode.startswith("dense") else "512"
    env["SAKER_SPARSE_RECENT_BLOCKS"] = "16"
    env["SAKER_SPARSE_SINK_BLOCKS"] = "1"
    env["SAKER_SPARSE_TRACE"] = "1"
    env["SAKER_SHORTLIST"] = "64"
    command = [str(args.server.resolve()), "-m", str(args.model.resolve()),
               "--host", "127.0.0.1", "--port", str(port), "-ngl", "0", "-c", "2048",
               "-np", "1", "-fa", "on", "-ctk", "f16", "-ctv", "f16", "-t", "2",
               "--no-webui", "--no-warmup", "--chat-template", "chatml"]
    if mode.endswith("mtp"):
        command += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    log_path = args.log_dir / f"{mode}.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_ready(proc, base, 30)
            for count, cache in [(800, False), (800, False), (820, True), (300, False)]:
                result = request(base, "/completion", {
                    "prompt": [65 + i % 20 for i in range(count)], "n_predict": 12,
                    "temperature": 0, "seed": 41, "ignore_eos": True, "cache_prompt": cache,
                    "logit_bias": [[i, -10000] for i in range(258) if not 65 <= i <= 90],
                }, timeout=120)
                assert "error" not in result, result
                assert result["timings"]["predicted_n"] >= 11, result
                if mode.endswith("mtp"):
                    assert result["timings"].get("draft_n", 0) > 0
                    accepted = result["timings"].get("draft_n_accepted", 0)
                    if args.expect_mtp_acceptance == "some":
                        assert accepted > 0, result["timings"]
                    elif args.expect_mtp_acceptance == "zero":
                        assert accepted == 0, result["timings"]
                print(mode, count, cache, result["timings"], flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    text = log_path.read_text()
    if mode.startswith("dense"):
        assert "saker-sparse: layer=" not in text
    else:
        assert "saker-sparse: active queries=1" in text, text[-4000:]
        if not mode.endswith("mtp"):
            assert "saker-sparse: layer=3 queries=1" in text, text[-4000:]
    if mode.endswith("mtp") and not mode.startswith("dense"):
        assert "saker-sparse: active queries=3" in text, text[-4000:]
        assert "saker-sparse: layer=3 queries=3" in text, text[-4000:]
        assert "saker-sparse: layer=4" in text, text[-4000:]
print("PASS: dense/representative/Quest/SparQ AR and MTP, cache resets, prefix reuse and short-context fallback")
