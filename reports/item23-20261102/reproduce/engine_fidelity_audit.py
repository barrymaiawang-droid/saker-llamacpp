"""Compare runtime distributions on the unedited failing code, not visual quality."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(os.environ.get("ITEM23_WORK", str(Path.home() / "Documents/Codex/2026-11-02/the-item-2-has-similar-issues/work")))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["hip-reference", "hip-control", "vk-fixed", "vk-stock", "vk-before",
                                       "vk-no-fa", "hip-no-fa", "hip-no-graphs", "vk-no-fusion", "vk-no-opt", "vk-no-async", "vk-f32", "cpu-control", "cpu-pure", "cpu-reference"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--chunks", type=int, default=2)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / (args.tag or args.arm)
    out.mkdir(parents=True, exist_ok=False)
    original = Path(os.environ.get("ITEM23_ORIGINAL", str(Path(__file__).resolve().parents[1] / "fixtures/original-pelican-bike.html")))
    fixture = root / "original-pelican-code.txt"
    if fixture.exists():
        assert fixture.read_bytes() == original.read_bytes()
    else:
        fixture.write_bytes(original.read_bytes())
    binary_dir = ROOT / "llama.cpp" / ("build-hip" if args.arm.startswith("hip-") else "build") / "bin"
    library_dir = ROOT / "runtime-before-fix/vulkan" if args.arm == "vk-before" else binary_dir
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SAKER_", "LLAMA_ARG_"))}
    env["LD_LIBRARY_PATH"] = str(library_dir) + ":" + str(binary_dir)
    env["GGML_VK_VISIBLE_DEVICES"] = "0"
    if args.arm == "vk-stock":
        env["SAKER_APU_MMQ"] = "0"
    if args.arm == "vk-f32":
        env["GGML_VK_DISABLE_F16"] = "1"
        env["GGML_VK_DISABLE_INTEGER_DOT_PRODUCT"] = "1"
    for arm, key in {"vk-no-fusion": "GGML_VK_DISABLE_FUSION", "vk-no-opt": "GGML_VK_DISABLE_GRAPH_OPTIMIZE",
                     "vk-no-async": "GGML_VK_DISABLE_ASYNC", "hip-no-graphs": "GGML_CUDA_DISABLE_GRAPHS"}.items():
        if args.arm == arm:
            env[key] = "1"
    reference = args.reference.resolve() if args.reference else root / ("cpu-reference.kld" if args.arm == "cpu-reference" else "hip-reference.kld")
    command = [str(binary_dir / "llama-perplexity"), "-m", str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"),
               "-ngl", "0" if args.arm.startswith("cpu-") else "99", "-c", "2560", "-b", "512", "-ub", "256",
               "-t", "16" if args.arm.startswith("cpu-") else "4", "-tb", "16" if args.arm.startswith("cpu-") else "4",
               "-fa", "off" if args.arm.endswith("no-fa") else "on", "--chunks", str(args.chunks)]
    if args.arm in ("cpu-pure", "cpu-reference"):
        command += ["--device", "none", "--no-op-offload", "--no-kv-offload"]
    if args.arm.endswith("reference"):
        assert not reference.exists()
        command += ["-f", str(fixture), "--save-all-logits", str(reference)]
    else:
        assert reference.exists()
        if not args.reference:
            assert (root / "hip-reference/complete.json").exists()
        command += ["--kl-divergence", "--kl-divergence-base", str(reference)]
    linkage = subprocess.check_output(["ldd", command[0]], env=env, text=True)
    if args.arm == "vk-before":
        assert str(library_dir / "libggml-vulkan.so") in linkage, linkage
    save(out / "metadata.json", {
        "command": command, "library_dir": str(library_dir), "ldd": linkage,
        "original_path": str(original), "fixture_sha256": sha(fixture),
        "runtime_hashes": {p.name: sha(p) for p in library_dir.glob("*.so.*") if not p.is_symlink()},
        "engine_environment": {k: v for k, v in env.items() if k.startswith(("SAKER_", "GGML_"))},
        "purpose": "Numerical fidelity on fixed text; not a visual quality score or full-model unquantized reference.",
    })
    start = time.perf_counter()
    with (out / "run.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    log = (out / "run.log").read_text()
    metrics = {"elapsed_seconds": time.perf_counter() - start}
    if args.arm.endswith("reference"):
        assert "Final estimate" in log and reference.stat().st_size > 100_000_000, log[-4000:]
        metrics["reference_bytes"] = reference.stat().st_size
    else:
        for key, pattern in {
            "mean_kld": r"Mean\s+KLD:\s+([0-9.eE+-]+)",
            "max_kld": r"Maximum KLD:\s+([0-9.eE+-]+)",
            "same_top_percent": r"Same top p:\s+([0-9.eE+-]+)",
            "ppl_ratio": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([0-9.eE+-]+)",
        }.items():
            match = re.search(pattern, log)
            if not match:
                raise RuntimeError("missing " + key + " in " + str(out / "run.log"))
            metrics[key] = float(match.group(1))
    save(out / "complete.json", metrics)
    print(json.dumps({"arm": args.arm, **metrics}), flush=True)


if __name__ == "__main__":
    main()
