"""Owned-process engine probe; persists real outputs and fails on missing data."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(os.environ.get("ITEM23_WORK", str(Path.home() / "Documents/Codex/2026-11-02/the-item-2-has-similar-issues/work")))
MODEL = ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
PROMPTS = {
    "arithmetic": "What is 17 multiplied by 23? Explain how to check it.",
    "geometry": "Write a Three.js function that creates a bicycle with both wheel axles along X, the bicycle travelling along Z, and wheels spinning without wobbling. Explain the coordinate conventions, then give code.",
    "code": "Write a correct Python function merge_intervals(intervals) with tests for overlapping, nested, adjacent, empty and unsorted intervals.",
    "pelican": "make a peligan rides a bike 3d webpage, dont search or refre to any other webpage in the folder, make one independantly on your own",
    "prefill": "Review the following code for coordinate-system mistakes.\n" + Path(os.environ.get("ITEM23_ORIGINAL", str(Path(__file__).resolve().parents[1] / "fixtures/original-pelican-bike.html"))).read_text(),
}


def call(base, path, body=None, timeout=180):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + path, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{e.code}: {e.read().decode()[:1000]}") from e


@contextlib.contextmanager
def server(args):
    binary = getattr(args, "binary", None) or ROOT / "llama.cpp" / ("build-hip" if args.backend == "hip" else "build") / "bin/llama-server"
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("SAKER_"):
            env.pop(key)
    if args.profile == "shortlist":
        env["SAKER_SHORTLIST"] = "3584"
    env["GGML_VK_VISIBLE_DEVICES"] = "0"
    if args.no_retune:
        env["SAKER_APU_MMQ"] = "0"
    if getattr(args, "force_qy", None) is not None:
        env["SAKER_FORCE_QY"] = str(args.force_qy)
    if getattr(args, "trace", False):
        env["SAKER_PF_TRACE"] = "1"
    if getattr(args, "wide_warp", False):
        env["SAKER_LMMQ_WT"] = "256,128,256,32,64,128,2,16,16,16,64"
    env["LD_LIBRARY_PATH"] = ":".join(p for p in [str(Path(binary).resolve().parent), env.get("LD_LIBRARY_PATH", "")] if p)
    cmd = [str(binary), "-m", str(MODEL), "-ngl", "99", "-c", str(args.ctx), "-ub", str(args.ubatch),
           "-b", "2048", "-np", "1", "-fa", "on", "--host", "127.0.0.1", "--port", str(args.port),
           "--alias", "q38gsq-review", "--temp", "1", "--top-k", "20", "--top-p", ".95", "--min-p", "0",
           "--repeat-penalty", "1", "--no-webui", "--cache-ram", "2048"]
    if args.threads:
        cmd += ["-t", str(args.threads), "-tb", str(args.threads)]
    if args.profile != "ar":
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    args.out.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    start = time.perf_counter()
    with (args.out / "server.log").open("wb") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode}; see {args.out / 'server.log'}")
            try:
                call(base, "/health", timeout=2)
                break
            except Exception:
                if time.perf_counter() - start > 180:
                    raise RuntimeError("server startup exceeded 180 seconds")
                time.sleep(.5)
        metadata = {"cmd": cmd, "backend": args.backend, "profile": args.profile, "load_seconds": time.perf_counter()-start,
                    "shortlist": env.get("SAKER_SHORTLIST"), "engine_env":{k:v for k,v in env.items() if k.startswith("SAKER_")}, "props": call(base, "/props"), "runs": []}
        yield base, metadata
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["vulkan", "hip"], default="vulkan")
    p.add_argument("--profile", choices=["ar", "mtp", "shortlist"], default="ar")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--ubatch", type=int, default=256)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--port", type=int, default=18153)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--probs", type=int, default=5)
    p.add_argument("--cases", default="arithmetic,geometry,code,pelican")
    p.add_argument("--no-retune", action="store_true")
    p.add_argument("--force-qy", choices=["0", "1"])
    p.add_argument("--trace", action="store_true")
    p.add_argument("--binary", type=Path)
    p.add_argument("--wide-warp", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if MODEL.with_suffix(".gguf.aria2").exists():
        raise RuntimeError("checkpoint download is incomplete")
    with server(args) as (base, metadata):
        # Warm the slot, outside measured runs.
        call(base, "/completion", {"prompt": "Warm up.", "n_predict": 16, "temperature": 0, "cache_prompt": False})
        for name in args.cases.split(","):
            prompt = PROMPTS[name]
            templated = call(base, "/apply-template", {"messages": [{"role": "user", "content": prompt}]})
            rendered = templated["prompt"]
            for rep in range(args.reps):
                body = {"prompt": rendered, "temperature": 0, "seed": 42, "n_predict": args.tokens,
                        "return_tokens": True, "n_probs": args.probs, "cache_prompt": False}
                start = time.perf_counter()
                result = call(base, "/completion", body, timeout=900)
                elapsed = time.perf_counter() - start
                if not result.get("tokens") or not result.get("content") or result.get("error"):
                    raise RuntimeError(f"missing output in {name} rep {rep}")
                digest = hashlib.sha256(json.dumps(result["tokens"]).encode()).hexdigest()
                (args.out / f"{name}-{rep}.json").write_text(json.dumps(result, indent=2)+"\n")
                item = {"case": name, "rep": rep, "hash": digest, "tokens": result["tokens"],
                        "timings": result.get("timings"), "elapsed": elapsed}
                metadata["runs"].append(item)
                (args.out / "metrics.json").write_text(json.dumps(metadata, indent=2)+"\n")
                print(name, rep, digest[:10], result.get("timings"), flush=True)
        print("DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
