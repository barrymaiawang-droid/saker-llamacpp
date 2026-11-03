"""Compare inference paths on identical inputs, without changing generated artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

from probe_engine import MODEL, PROMPTS, ROOT, call, server


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["vk-ar", "vk-stock-ar", "hip-ar", "vk-mtp"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    out = args.root / args.arm
    out.mkdir(parents=True, exist_ok=False)
    launch = SimpleNamespace(
        backend="hip" if args.arm == "hip-ar" else "vulkan",
        profile="mtp" if args.arm == "vk-mtp" else "ar",
        ctx=16384, ubatch=256, threads=4, port=18153,
        no_retune=args.arm == "vk-stock-ar", out=out,
    )
    with server(launch) as (base, metadata):
        binary_dir = Path(metadata["cmd"][0]).resolve().parent
        metadata["runtime_hashes"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in binary_dir.glob("*.so.*") if not p.is_symlink()
        }
        metadata["model_path"] = str(MODEL.resolve())
        save(out / "metadata.json", metadata)
        fixture_path = args.root / "fixtures.json"
        if fixture_path.exists():
            fixtures = json.loads(fixture_path.read_text())
        else:
            fixtures = {}
            for case in ["geometry", "pelican", "prefill"]:
                prompt = call(base, "/apply-template", {"messages": [
                    {"role": "user", "content": PROMPTS[case]}]})["prompt"]
                ids = call(base, "/tokenize", {
                    "content": prompt, "add_special": False, "parse_special": True})["tokens"]
                fixtures[case] = {"prompt": prompt, "ids": ids}
            for case, offset in [("geometry", 99), ("pelican", 27)]:
                prior = json.loads((ROOT / "metrics-vulkan-ar" / f"{case}-0.json").read_text())
                fixtures[case + "-prefix"] = {
                    "ids": fixtures[case]["ids"] + prior["tokens"][:offset],
                    "source_tokens": str(ROOT / "metrics-vulkan-ar" / f"{case}-0.json"),
                    "offset": offset,
                }
            save(fixture_path, fixtures)
        fixture_hash = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
        call(base, "/completion", {"prompt": "Warm up.", "n_predict": 8,
             "temperature": 0, "cache_prompt": False})
        runs = []
        cases = [("geometry", 128, 3), ("pelican", 128, 3),
                 ("geometry-prefix", 1, 3), ("pelican-prefix", 1, 3),
                 ("prefill", 1, 2)]
        for case, budget, repetitions in cases:
            for rep in range(repetitions):
                request = {"prompt": fixtures[case]["ids"], "temperature": 0,
                    "seed": 42, "n_predict": budget, "return_tokens": True,
                    "n_probs": 8, "post_sampling_probs": False, "cache_prompt": False,
                    "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
                    "samplers": ["temperature"]}
                stem = f"{case}-{rep}"
                save(out / (stem + "-request.json"), request)
                started = time.perf_counter()
                result = call(base, "/completion", request, timeout=900)
                elapsed = time.perf_counter() - started
                if not result.get("tokens") or not result.get("completion_probabilities"):
                    raise RuntimeError("missing inference output: " + stem)
                save(out / (stem + ".json"), result)
                item = {"case": case, "rep": rep, "elapsed": elapsed,
                    "tokens": len(result["tokens"]),
                    "token_hash": hashlib.sha256(json.dumps(result["tokens"]).encode()).hexdigest(),
                    "timings": result.get("timings"), "fixture_sha256": fixture_hash}
                runs.append(item)
                save(out / "runs.json", runs)
                print(json.dumps({"arm": args.arm, **item}), flush=True)
        save(out / "complete.json", {"runs": len(runs), "fixture_sha256": fixture_hash})


if __name__ == "__main__":
    main()
