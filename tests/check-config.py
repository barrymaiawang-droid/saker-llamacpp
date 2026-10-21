#!/usr/bin/env python3
"""Check sparse configuration parsing in separate processes (options are read once)."""
import os
from pathlib import Path
import subprocess
import sys

binary = str(Path(sys.argv[1]).resolve())
base = {key: value for key, value in os.environ.items() if not key.startswith("SAKER_SPAR")}
cases = [({}, True), ({"TOKENS": "0", "MODE": "ignored-while-disabled"}, True)]
for mode in ["representative", "quest", "sparq"]:
    cases.append(({"TOKENS": "512", "MODE": mode}, True))
for tokens in ["-1", "abc", "512x", "999999999999999999999999999999", "8193", "513", "256"]:
    cases.append(({"TOKENS": tokens}, False))
cases.extend([
    ({"TOKENS": "512", "MODE": "unknown"}, False),
    ({"TOKENS": "512", "DENSE_LAYERS": "-1"}, False),
    ({"TOKENS": "512", "MODE": "sparq", "SAKER_SPARQ_CHANNELS": "0"}, False),
    ({"TOKENS": "512", "MODE": "sparq", "SAKER_SPARQ_CHANNELS": "4097"}, False),
    ({"TOKENS": "512", "MODE": "sparq", "SAKER_SPARQ_RECENT_TOKENS": "496"}, False),
    ({"TOKENS": "512", "MODE": "quest", "RECENT_BLOCKS": "31"}, False),
    ({"TOKENS": "32", "MODE": "sparq", "SAKER_SPARQ_RECENT_TOKENS": "0", "SAKER_SPARQ_SINK_TOKENS": "0"}, True),
])
for settings, valid in cases:
    env = dict(base, **{key if key.startswith("SAKER_") else "SAKER_SPARSE_" + key: value for key, value in settings.items()})
    result = subprocess.run([binary, "--config"], env=env, text=True, capture_output=True)
    assert (result.returncode == 0) == valid, (settings, result.stdout, result.stderr)
    if valid and settings.get("MODE") in ["quest", "sparq"]:
        assert f"mode={settings['MODE']}" in result.stdout
        assert f"dense_layers={2 if settings['MODE']=='quest' else 0}" in result.stdout
print(f"PASS: {len(cases)} configuration cases")
