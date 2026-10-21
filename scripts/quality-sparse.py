#!/usr/bin/env python3
"""Compare teacher-forced CPU decode against dense attention on identical text tokens."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", type=Path, required=True, help="build/bin/saker-quality")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory")
    parser.add_argument("--prefix", type=int, default=4096)
    parser.add_argument("--score", type=int, default=128)
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--budgets", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--dense-layers", type=int, default=2)
    parser.add_argument("--selectors", choices=["representative", "quest", "sparq"], nargs="+", default=["representative", "quest", "sparq"])
    parser.add_argument("--state-check", action="store_true", help="also verify saved-cache restoration")
    args = parser.parse_args()
    if min(args.score, args.windows, args.threads) < 1 or args.prefix < 2:
        parser.error("prefix must be >= 2; score/windows/threads must be positive")
    if any(b < 288 or b > 8192 or b % 16 or b >= args.prefix for b in args.budgets):
        parser.error("budgets must be multiples of 16 in 288..8192 and below the prefix")
    if not 1 <= args.channels <= 4096 or not 0 <= args.dense_layers <= 256:
        parser.error("channels must be 1..4096; dense-layers must be 0..256")
    for path in [args.evaluator, args.model, args.corpus]:
        if not path.is_file():
            parser.error(f"file not found: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results, dense = [], None
    arms = [("dense", 0)] + [(mode, budget) for budget in args.budgets for mode in args.selectors]
    for mode, budget in arms:
        name = "dense" if mode == "dense" else f"{mode}-{budget}"
        output = args.output_dir / f"{name}.csv"
        log_path = args.output_dir / f"{name}.log"
        env = {key: value for key, value in os.environ.items() if not key.startswith("SAKER_SPAR")}
        env.update(SAKER_SPARSE_MODE=mode if budget else "representative", SAKER_SPARSE_TOKENS=str(budget),
                   SAKER_SPARSE_DENSE_LAYERS=str(args.dense_layers), SAKER_SPARQ_CHANNELS=str(args.channels), SAKER_SPARSE_TRACE="1")
        command = [str(args.evaluator.resolve()), str(args.model.resolve()), str(args.corpus.resolve()), str(output.resolve()),
                   str(args.prefix), str(args.score), str(args.windows), str(args.threads), str(int(args.state_check))]
        with log_path.open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        with output.open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != args.score * args.windows:
            raise RuntimeError(f"incomplete scores: {name}")
        if dense is None:
            dense = rows
        if any((a["window"], a["step"], a["target"]) != (b["window"], b["step"], b["target"]) for a, b in zip(rows, dense)):
            raise RuntimeError("evaluation tokens differ between arms")
        if budget:
            log = log_path.read_text()
            if "saker-sparse: active queries=1" not in log or f"mode={mode}" not in log:
                raise RuntimeError(f"no proof of sparse activation: {name}")
        result = {"mode": mode, "budget": budget, "tokens": len(rows),
                  "perplexity": math.exp(-sum(float(row["logprob"]) for row in rows) / len(rows)),
                  "dense_top1_agreement": sum(a["top1"] == b["top1"] for a, b in zip(rows, dense)) / len(rows),
                  "command": command}
        results.append(result)
        print(json.dumps(result), flush=True)
        report = {"model": str(args.model.resolve()), "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
                  "prefix": args.prefix, "score": args.score, "windows": args.windows,
                  "channels": args.channels, "dense_layers": args.dense_layers, "results": results}
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
