# Review evidence subset

This directory contains selected, unchanged evidence from the completed node A investigation: numerical comparison JSON, exact token fixtures, model-loading and conversion metadata, lineage checks, serializer source/tests, performance measurements, and unedited replay pages/screenshots.

`MANIFEST.json` records the source location, size and SHA-256 of every copied file. It also records hashes of 26 raw full-vocabulary NPZ archives that are deliberately not copied into the repository. Reports copied here describe the original full investigation; references to raw logits or other files outside this subset refer to those original locations.

The complete local investigation is:

`<NODE_A_HOME>/Documents/Codex/2026-11-02/the-item-2-has-similar-issues/work/node-a-independent-reference-20261102`

The node A investigation is:

`<NODE_A_HOME>/item23-independent-reference-20261102`

The original reference outputs and commands are identified in each comparison and command JSON. The complete local folders `short-f32/`, `long-f32/` and `diagnostic-upstream-cpu/` retain the raw NPZ arrays; model files remain on node A. The same retained comparator can recompute a summary with:

```bash
python3 compare_logits.py /absolute/path/reference.npz /absolute/path/candidate.npz --output /absolute/path/comparison.json
```

The standalone HTML context is teacher-forced, not the original sampled conversation. The evidence supports the normalization repair and bounded numerical agreement; it does not establish a visual-quality fix or clear every engine/backend/sampler path.
