#!/usr/bin/env python3
"""Compare dense/sparse AR and MTP on the same model and real prompt tokens."""

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


def request(base, path, payload=None, timeout=3600):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def wait_ready(proc, base, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited; inspect its log")
        try:
            if request(base, "/health", timeout=2).get("status") == "ok":
                return
        except (OSError, urllib.error.HTTPError):
            pass
        time.sleep(0.5)
    raise TimeoutError("llama-server did not become ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True,
                        help="real corpus long enough for the largest depth; never repeated automatically")
    parser.add_argument("--output", type=Path, required=True, help="new JSONL file; existing files are refused")
    parser.add_argument("--depths", type=int, nargs="+", default=[4096, 16384, 32768])
    parser.add_argument("--predict", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--shortlist", type=int, default=3584)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--port", type=int, default=18089)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--reverse-arms", action="store_true", help="reverse run order to check thermal drift")
    parser.add_argument("--modes", choices=["ar", "mtp"], nargs="+", default=["ar", "mtp"])
    args = parser.parse_args()
    if min(args.depths) < 1 or args.predict < 2 or args.repetitions < 1:
        parser.error("depths/repetitions must be positive and predict must be at least 2")
    if args.budget < 288 or args.budget > 8192 or args.budget % 16:
        parser.error("budget must be a multiple of 16 between 288 and 8192")
    if min(args.depths) <= args.budget:
        parser.error("all depths must exceed the sparse budget")
    for path in [args.server, args.model, args.prompt_file]:
        if not path.is_file():
            parser.error(f"file not found: {path}")

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    base = f"http://127.0.0.1:{args.port}"
    corpus = args.prompt_file.read_text(encoding="utf-8")
    ctx = max(args.depths) + args.predict + 256
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the output before starting any servers.
    with args.output.open("x", encoding="utf-8") as output:
        tokens = None
        for mode in (list(reversed(args.modes)) if args.reverse_arms else args.modes):
            for sparse in ([True, False] if args.reverse_arms else [False, True]):
                arm = f"{mode}-{'sparse' if sparse else 'dense'}"
                log_path = args.output.with_name(args.output.name + f".{arm}.log")
                env = os.environ.copy()
                env["SAKER_SPARSE_TOKENS"] = str(args.budget if sparse else 0)
                env["SAKER_SPARSE_RECENT_BLOCKS"] = "16"
                env["SAKER_SPARSE_SINK_BLOCKS"] = "1"
                env["SAKER_SHORTLIST"] = str(args.shortlist)
                # Capture proof that the sparse graph was used, including verification batches.
                env["SAKER_SPARSE_TRACE"] = "1"
                command = [str(args.server.resolve()), "-m", str(args.model.resolve()),
                           "--host", "127.0.0.1", "--port", str(args.port),
                           "-ngl", str(args.gpu_layers), "-t", str(args.threads), "-c", str(ctx), "-np", "1", "-fa", "on",
                           "-ctk", "f16", "-ctv", "f16", "--no-webui", "--temp", "0"]
                if mode == "mtp":
                    command += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
                proc = None
                try:
                    with log_path.open("x", encoding="utf-8") as log:
                        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                        wait_ready(proc, base, args.startup_timeout)
                        arm_tokens = request(base, "/tokenize", {"content": corpus, "add_special": True})["tokens"]
                        if tokens is None:
                            tokens = arm_tokens
                        if arm_tokens != tokens:
                            raise RuntimeError("tokenizer outputs differ between benchmark arms")
                        if len(tokens) < max(args.depths):
                            raise ValueError(f"corpus has {len(tokens)} tokens; need {max(args.depths)}")
                        for depth in args.depths:
                            payload = {"prompt": tokens[:depth], "n_predict": args.predict,
                                       "temperature": 0, "seed": 1234, "cache_prompt": False,
                                       "ignore_eos": True, "stream": False}
                            request(base, "/completion", dict(payload, n_predict=32))
                            if sparse and "saker-sparse: active" not in log_path.read_text(encoding="utf-8", errors="replace"):
                                raise RuntimeError("sparse graph was never used; inspect architecture and build")
                            for repeat in range(args.repetitions):
                                start = time.monotonic()
                                result = request(base, "/completion", payload)
                                timings = result["timings"]
                                actual = timings["prompt_n"] + timings.get("cache_n", 0)
                                if actual != depth or result.get("truncated", False):
                                    raise RuntimeError(f"requested {depth} context tokens, measured {actual}; refusing misleading result")
                                if timings["predicted_n"] < args.predict - 1:
                                    raise RuntimeError("generation ended before the requested length")
                                content = result.get("content", "")
                                row = {"arm": arm, "depth": depth, "repeat": repeat,
                                       "budget": args.budget if sparse else 0,
                                       "wall_seconds": time.monotonic() - start,
                                       "timings": timings, "content": content,
                                       "output_sha256": hashlib.sha256(content.encode()).hexdigest(),
                                       "prompt_sha256": hashlib.sha256(json.dumps(tokens[:depth]).encode()).hexdigest(),
                                       "model": str(args.model.resolve()), "command": command,
                                       "generation_settings": result.get("generation_settings"),
                                       "tokens_evaluated": result.get("tokens_evaluated"),
                                       "tokens_predicted": result.get("tokens_predicted")}
                                output.write(json.dumps(row) + "\n")
                                output.flush()
                                print(arm, depth, repeat, timings["predicted_per_second"], flush=True)
                    if sparse and "saker-sparse: active" not in log_path.read_text(encoding="utf-8", errors="replace"):
                        raise RuntimeError("sparse graph was never used; inspect model architecture and build")
                finally:
                    if proc is not None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
    print(f"Wrote {args.output}. Output agreement alone is not a quality evaluation.")


if __name__ == "__main__":
    main()
