# Handoff to Claude: item 23 engine quality and speed

**Date:** 2026-11-02. **Status:** confirmed Vulkan arithmetic defect repaired; remaining pelican generation-quality problem unresolved. **User's next action:** let Claude continue fixing the engine.

**Update, 2026-11-02 (Claude, offline review without GPU or model access):** no second engine
defect found; see [the offline follow-up](#offline-follow-up-2026-11-02-claude) at the end and the
[upstream delta report](item23-20261102/upstream-delta-20261102.md). One upstream-confirmed graph
fidelity fix is published as `patches/saker-gdn-l2norm-eps.patch` (not built or installed on node
B). The reproduction harness gained a CPU reference arm; running it is the next decisive step.

## Read this first

The user wants the **generation engine** diagnosed and fixed, using the bad pelican as evidence. They explicitly rejected editing the generated HTML, adding art-specific instructions, or polishing a test result and presenting it as an engine repair. Keep the original prompt and untouched outputs as test fixtures. A prettier manually repaired page is not success.

A real numerical defect was found, fixed, built and installed before the latest follow-up. Subsequent numerical tests support that repair. The latest fresh generation still fails visual/geometry checks, and the last follow-up added **no new production engine patch**. Do not tell the user the full quality problem is solved. Also do not claim it is impossible to solve or definitely caused by quantization: that has not been established.

Work in this branch is a handoff, not an upstream PR. The user explicitly authorized this commit and push. No PR was opened.

## Entry points and evidence

Paths below are relative to this repository unless marked as node-local.

| Start here | Purpose |
| --- | --- |
| [Latest follow-up](item23-20261102/q38gsq-followup.txt) and [JSON](item23-20261102/q38gsq-followup.json) | State/rollback checks, speed sweep, fresh failed generation and limits |
| [Independent CPU audit](item23-20261102/q38gsq-engine-diagnosis.md) and [JSON](item23-20261102/q38gsq-engine-diagnosis.json) | Evidence that the earlier kernel patch repairs numerical corruption |
| [Correctness delta](../patches/saker-quality-fix.patch) | Apply to the pre-repair Saker tree |
| [Combined patch](../patches/saker-combined.patch) | Apply to a fresh pinned llama.cpp checkout; includes the earlier Saker work |
| [Reproduction sources](item23-20261102/reproduce/) | Owned-process server probes, CPU fidelity audit, C++ state probes and browser QA |
| [Frozen inputs and raw outputs](item23-20261102/fixtures/) | Original failing page, fresh unedited page, request, screenshot, QA and fixed token IDs |
| [Full numerical audit archive](item23-20261102/q38gsq-engine-audit.zip) | 217 captured files: requests, responses, logs, manifests and scripts |
| [State/generation follow-up archive](item23-20261102/q38gsq-followup-evidence.zip) | State logs, benchmarks, complete streamed response, raw HTML and browser QA |
| [Handoff validation](item23-20261102/handoff-validation.json) | 37 passing Python tests and both patch application checks |
| [Artifact hashes](item23-20261102/SHA256SUMS.json) | Hashes of the published evidence copies |

The archives preserve failed and excluded experiments as well as successful ones. Read their status files and the reports; the presence of a log is not a pass. Published copies redact private home/workspace roots and the LAN address as `<NODE_B_HOME>`, `<TASK_ROOT>` and `<NODE_B_IP>`. Some recorded original artifact hashes therefore predate path redaction. Model, runtime and HTML hashes are not changed by redaction. The reproduction directory has environment-configurable paths; archived scripts are historical captures.

## Repository and working state

- Project origin: `https://github.com/barrymaiawang-droid/saker-llamacpp.git`.
- Project base before these changes: `f50ec4b2007fba55eb499a85be0f5d4e0ae3443a`.
- Published branch: `codex/item23-engine-handoff-20261102`.
- llama.cpp base: **`9d57ce456c94d241dde672b2db9cf18879766568`**.
- The root Codex workspace is not a Git repository. This repository stores patches for the separate llama.cpp checkout; the compiled source checkout itself is not pushed to ggml-org.
- The combined patch includes pre-existing experimental Saker shortlist code. The installed service explicitly disables the shortlist. Do not mistake every line in the combined patch for a new correctness fix.
- Node-local fleet/registry changes are already installed. Their unrelated entries were preserved. This handoff publishes the service launcher and records the runtime configuration; it does not publish private whole-fleet configuration files.

On node B, define:

```bash
export TASK_ROOT="$HOME/Documents/Codex/2026-11-02/the-item-2-has-similar-issues"
export ITEM23_WORK="$TASK_ROOT/work"
export LLAMA="$ITEM23_WORK/llama.cpp"
export MODEL="$HOME/models/Qwen3.8-27B-GSQ-RCO/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
export SERVICE="$HOME/llmfleet/engines/q38gsq/q38gsq_service.py"
# Set REPO to this repository's actual checkout location.
export REPO="$ITEM23_WORK/saker-llamacpp"
export REPRO="$REPO/reports/item23-20261102/reproduce"
export ITEM23_ORIGINAL="$REPO/reports/item23-20261102/fixtures/original-pelican-bike.html"
```

Node-local preserved material:

- `$LLAMA/build/bin`: corrected Vulkan build; `$LLAMA/build-hip/bin`: HIP comparison build.
- `$ITEM23_WORK/runtime-before-fix/vulkan`: preserved bad Vulkan runtime for controlled comparisons.
- `$ITEM23_WORK/engine-audit-20261102`: original, unredacted numerical evidence.
- `$ITEM23_WORK/state-repair-20261102`: original, unredacted latest state/benchmark/generation evidence.
- `$TASK_ROOT/outputs`: earlier review artifacts, including some manually edited pages. **Those edited pages are excluded from engine validation.**
- Original user artifact: `$HOME/dsh/dsh2/test3/pelican-bike.html`; never modified in these follow-ups.
- Large CPU reference: `$ITEM23_WORK/engine-audit-20261102/fidelity/cpu-first-chunk.kld`; deliberately excluded from Git.

The actual source checkout remains dirty with the combined Saker changes. Do not reset it or apply the combined patch on top of it. For a fresh checkout, use the pinned base and apply only the combined patch. For an old Saker checkout, use only the quality delta. Both application paths were verified during handoff preparation.

## Hardware, model and live service

All relevant execution was on **node B**, locally: Ryzen AI Max+ 395, Radeon 8060S / gfx1151, about 123 GiB RAM. Node A was not contacted or modified. The user explicitly authorized node B. No item-22 server was running during the latest follow-up; this agent did not stop it in that follow-up.

Model: `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, architecture `qwen35`, 64 trunk layers plus one MTP layer, mixed quantization, 12,120,016,960 bytes.

- GGUF SHA256: `58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f`.
- Model source: [pinned ISTA-DASLab GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/blob/c7b6817b8affebf3b38c58313394b061345a4924/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf).
- Embedded native template was verified unchanged against the official template; SHA256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.
- The model was not requantized or replaced.

**At handoff, item 23 is RUNNING and healthy on port 18151.** Use the launcher, which checks its owned PID and process start time:

```bash
python "$SERVICE" status
python "$SERVICE" dry-run
# Stop only item 23 when an exclusive GPU experiment requires it:
python "$SERVICE" stop
# Restore after the experiment:
python "$SERVICE" start
```

The service config is `$HOME/llmfleet/engines/q38gsq/q38gsq-runtime.json`. The runtime is in its `runtime-v1` sibling. The installed runtime files matched the audited build; see the archived process/library verification.

Current settings: full-vocabulary native MTP2, max draft 2; ctx 262144; batch 2048; ubatch 256; threads 4 / batch threads 4; flash attention on; one slot; RAM prompt cache 2048 MiB; reasoning enabled, medium effort; output budget 32768. Native thinking sampling is temperature 1, top-p .95, top-k 20, min-p 0, repeat penalty 1. `SAKER_SHORTLIST` is removed by the launcher. Low/medium/xhigh remain selectable. Explicit request sampling overrides are respected.

Avoid broad process kills. The older `tools/verify_bit_exact.py` still has historical node-A paths and a launcher/port cleanup routine. Its verdict tests are mocked; **do not run its real GPU workflow blindly on node B**. Prefer the owned-process `reproduce/probe_engine.py` harness or the item-23 service launcher.

## Confirmed engine defect and implemented repair

The Saker Vulkan APU retune widened a quantized block from BN128 to BN256 while leaving per-warp WN64, so half the columns could remain unwritten. It also reused a 256-column dispatch divisor for floating kernels still using 128-wide tiles, and reused grids for integer K/IQ kernels with different tile shapes.

The quality patch changes `ggml/src/ggml-vulkan/ggml-vulkan.cpp` to grow BN/WN together, retain the correct floating grid, and derive integer-dot dispatch dimensions from the actual kernel tile. `tests/test-backend-ops.cpp` adds boundary and Qwen-shaped matrix/GDN regressions. The patch files show the exact changes; do not reconstruct them from prose.

Earlier validation: 80/80 matrix cases, 50/50 GDN cases, native sampler/grammar checks, 71 captured tool schemas and 16,098 streaming parser prefixes passed. Launcher/verifier tests now reject missing outputs and unstable references. The 37 Python tests were rerun successfully for this handoff.

The independent CPU audit is the strongest full-model evidence: on 1,279 scored positions in a fixed 2,560-token block of the original pelican code, mean KL fell from **6.271537** in the preserved bad engine to **0.002961** in corrected Vulkan. Top-token agreement with CPU increased from **42.377%** to **98.436%**. The approximately 99.953% reduction is numerical discrepancy, **not a visual-quality improvement percentage**. Read the audit report for the controls and limitations.

## Latest reproduction: still a failure

Exact user prompt:

> make a peligan rides a bike 3d webpage, dont search or refre to any other webpage in the folder, make one independantly on your own

The request uses generic file-write instructions and a Three.js 0.160.0 import map, with no pelican anatomy, wheel-axis or scenery-direction hints. The complete request is [fresh-request.json](item23-20261102/fixtures/fresh-request.json).

On the installed service, ctx262144, medium effort, temperature1, seed42, the fresh request completed one native `write_file` call: 5539 completion tokens, 190.84 seconds total, 29.33 generated tokens/s, 3462 of 4156 proposed drafts accepted. The output is **byte-identical** to the earlier medium run at ctx32768. It is saved without edits as [fresh-pelican.html](item23-20261102/fixtures/fresh-pelican.html).

Browser QA fails despite no JavaScript exceptions. Spokes extend 0.87 units from a hub whose wheel radius is 0.48; scenery moves perpendicular to the bicycle; frame-dependent motion travels 3.36 units at 60fps versus 1.68 at 30fps in equal time. Visual review still finds awkward bird proportions and segmented neck geometry. The [screenshot](item23-20261102/fixtures/fresh-pelican.png) and [QA measurements](item23-20261102/fixtures/fresh-qa.json) are evidence of failure, not a repaired deliverable.

- Original failing HTML SHA256: `7cb912dead4b640bc4554b9384659d858552de92a6b2d8efb8b1bddb4d5b7733`.
- Fresh raw HTML SHA256: `07be2bdea466290c920a2311dbeca981fd6199e42952ca6055b37db64d21aef4`.

## Findings that prevent repeating dead ends

1. **HIP is not a trustworthy reference here.** It diverged from pure CPU with mean KL .432554 and top agreement 81.392% on the scored block. Its cause remains unresolved. Corrected Vulkan was much closer. Do not call Vulkan wrong merely because it differs from this HIP build.
2. **`-ngl 0` alone was not CPU-only.** A valid CPU reference also used `--device none --no-op-offload --no-kv-offload`. The earlier `cpu-control` run is excluded as an independent CPU baseline.
3. **The CPU reference contains one complete block.** The original run requested two; it was stopped after its first complete block. Extraction changed the header's block count from two to one and retained that block's exact tokens/probabilities. The incomplete second block is excluded. The valid file is 635,223,052 bytes; SHA256 `f80864ee9b059a824a4b9e2f61ebd9466094de92c8e5ce012dca4fbe2f9c0873`.
4. **The alleged max_tokens cap was false.** A newer interrupted stream had been compared with an older response file. The pinned server accepts `max_tokens`, `max_completion_tokens` and `n_predict`. Always use fresh output directories and require completion markers.
5. **Recurrent dirty restore was not reproduced as the cause.** The upstream regression first tripped a strict 1e-5 Vulkan logit threshold. Its later dirty comparison overwrote the correct expected logits with a replay that rolled back across separate single-token decodes, outside the last decode's saved snapshots. Preserving the original reference gives exact CPU logits for fresh and dirty restores; Vulkan KL is below .00005 at the three tested positions. Clearing dirty memory did not fix the original mismatched comparison. Do not patch restore code on the basis of that false attribution. The scratch diagnostic reports distributions; its exit0 and inherited success text alone are not a numerical pass.
6. **Actual MTP target rollback tested clean on the selected case.** `spec_state_probe.cpp` forces 192 code tokens after a 4096-token prefix and compares ordinary one-token decoding, batch3, MTP target state with two rollback slots, and alternating partial rejection with deliberately wrong drafts. All 192 top tokens agree. Mixed rejection versus ordinary decode has mean KL .000006839. This is target-state validation, not a proof of every stochastic sampler path.
7. **No obvious causality error in corrected Vulkan.** Earlier 256/1536-token probes left earlier logits unchanged when future tokens changed. Repeat results matched; prefix-only differences were small. Full page generation also matched byte-for-byte between 32768 and 262144 contexts for this seed. Three fixed next-token probes matched top token/top-eight IDs at 16384 versus 262144, but their probabilities were not all identical (largest top-eight logprob delta .14933).
8. **Diagnostic toggles did not isolate another Vulkan defect.** Disabling FA, fusion, graph optimization or async did not explain the HIP/Vulkan difference; disabling HIP graph capture did not fix HIP. Vulkan float32 was slightly closer to CPU (mean KL .001687) but was not installed as a speed profile. Do not repeat broad flag sweeps without a new falsifiable hypothesis.
9. **Some artifacts are explicitly invalid.** `layers-vk` started before compilation finished; `layers-vk2` is the replacement. Interrupted/missing runs and edited HTML are excluded. Short greedy matches with HIP are not a quality proof.

Relevant upstream reports that prompted the state investigation: [#29493](https://github.com/ggml-org/llama.cpp/issues/29493) and [#26695](https://github.com/ggml-org/llama.cpp/issues/26695). They are leads, not proof of this installation's failure. The native MTP verifier preserves an anchor and does not use the full-last-decode rollback case in #26695.

## Speed: keep the valid baseline

MTP2 is retained. The prior controlled short probes measured about 26-28 tokens/s versus 15 for ordinary decoding. The fresh complete request measured 29.33 tokens/s. Do not compare these as a new improvement; workloads and measurement conditions differ.

The latest 2048-token prompt-only sweep, three repeats per setting, found the current corrected retune/ub256 best among tested alternatives: 293.12 tokens/s versus 268.30 at ub512, 237.48 at ub1024, 186.90 at ub128, and standard kernels 263.16 at ub256 / 230.18 at ub128. These were sequential runs, not a randomized performance study. No new tuning was deployed. Historical high throughput from unwritten columns or the shortlist is not a valid speed win.

## Reproduction commands

Stop only item23 if exclusive GPU use is needed, then restore it when done. The scripts expect the existing `ITEM23_WORK` layout and model symlink; change that environment path if using another checkout.

Python regressions (do not load the model):

```bash
cd "$REPO"
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Fixed-token four-arm comparison, using the published frozen fixture instead of regenerating it from older local runs:

```bash
AUDIT=$(mktemp -d /tmp/item23-recheck.XXXXXX)
cp "$REPO/reports/item23-20261102/fixtures/frozen-inference-fixtures.json" "$AUDIT/fixtures.json"
python "$REPRO/engine_diagnosis.py" vk-ar --root "$AUDIT"
python "$REPRO/engine_diagnosis.py" vk-mtp --root "$AUDIT"
# Optional controls: vk-stock-ar or hip-ar. HIP is not ground truth.
```

CPU-reference comparison using the already validated local one-block file:

```bash
FIDELITY=$(mktemp -d /tmp/item23-fidelity.XXXXXX)
python "$REPRO/engine_fidelity_audit.py" vk-fixed --root "$FIDELITY" \
  --reference "$ITEM23_WORK/engine-audit-20261102/fidelity/cpu-first-chunk.kld" --chunks 1
```

Build and run the targeted MTP state probe without modifying production source:

```bash
PROBE_BIN=$(mktemp -d /tmp/item23-probes.XXXXXX)
g++ -std=c++17 -O2 "$REPRO/spec_state_probe.cpp" \
  -I"$LLAMA/include" -I"$LLAMA/src" -I"$LLAMA/ggml/include" \
  -L"$LLAMA/build/bin" -Wl,-rpath,"$LLAMA/build/bin" \
  -lllama -lggml -lggml-base -o "$PROBE_BIN/spec_state_probe"
"$PROBE_BIN/spec_state_probe" "$MODEL" "$ITEM23_ORIGINAL" 4096 192
```

The C++ probe's token perturbation is intentionally specific to this model's 248320-token vocabulary. Do not reuse unchanged on a different model.

Run the unchanged task in a fresh directory; this harness starts/cleans its own server on port18153, using ctx32768:

```bash
FRESH=$(mktemp -d /tmp/item23-generation.XXXXXX)
python "$REPRO/probe_project.py" --profile mtp --effort medium --seed 42 \
  --tokens 16384 --out "$FRESH"
```

For an exact installed-service replay at ctx262144, POST the published `fresh-request.json` to `http://127.0.0.1:18151/v1/chat/completions`, collect the entire stream, and require a completed `tool_calls` finish. `probe_project.generate()` is the reusable stream collector. Do not compare a partial stream with an old response file.

Browser QA requires Python Playwright, Chrome at `/opt/google/chrome/chrome`, and a local Three.js 0.160.0 vendor tree. Set `ITEM23_THREE_VENDOR` if it is not at `$HOME/dsh/dsh2/test3/vendor/three`. It instruments the library for inspection, never rewrites the generated HTML. Its torus-wheel/spoke identification is a heuristic; inspect source and screenshot before interpreting a failure on a different page.

```bash
python "$REPRO/check_scene.py" "$FRESH/pelican.html" --out "$FRESH/qa"
```

## What Claude should do next

1. Read the two reports and inspect the raw failed output before changing code. Separate the confirmed repaired numerical defect from the unresolved generation problem.
2. Choose a falsifiable hypothesis for a remaining engine error. A meaningful gap is equivalence to a higher-precision/original model implementation; all CPU/Vulkan fidelity results so far use the same quantized GGUF. That comparison requires a trustworthy model/reference, not the currently discrepant HIP build. Do not silently swap models and call it an engine patch.
3. If investigating a serving/state/sampling path, use identical frozen tokens and full distributions at the suspected boundary. The short tests do not prove arbitrary long contexts, multi-turn cache histories or every stochastic sampling path. The last follow-up did not exercise every one of those combinations.
4. Make an engine change only after a minimal reproduction identifies its failing seam. Add a general regression, rebuild, verify loaded runtime hashes, and replay the original request without custom art hints. Keep the raw output even if it fails.
5. If arithmetic and serving remain faithful but semantic scene construction stays wrong, report that distinction plainly. Do not manufacture another code defect or repair the specimen to close the task.
6. Restore the owned item23 service after exclusive tests. Do not contact/change node A or other fleet entries without task need and authorization.

## Suggested skills and discovery rules

- **diagnosing-bugs**: use the fast reproduction / falsifiable hypothesis / minimal fix loop. The previous session read this skill.
- **tdd**, if available and useful once a concrete defect is isolated: put the regression at the engine seam, not a pelican-specific output matcher.
- **handoff**: update this document when transferring work again, with exact status and failed attempts.
- Do not use image generation or page-polishing workflows to substitute for the engine task.

The user requires codebase-memory-mcp for code discovery: index if needed, then prefer search_graph, trace_path, get_code_snippet, query_graph/search_code. It returned `Transport closed` in this session, so `rg` and source reads were used as fallback. Retry the graph in a new session; do not pretend it was available.

Read the applicable AGENTS.md before editing. The llama.cpp checkout exempts private forks from upstream contribution requirements but prohibits committing/pushing without explicit user authorization and has strict upstream-PR restrictions. This handoff commit is in the Saker project, not ggml-org/llama.cpp. No upstream PR, issue or comment was sent.

## Offline follow-up, 2026-11-02 (Claude)

This review ran in a container with no GPU, no model file and no access to node B. It could read
the evidence archives, the patches and upstream llama.cpp history, compile-check C++ against the
pinned headers, and run the Python tests. It could not generate, measure fidelity or benchmark.
Nothing on node B was changed. The codebase-memory graph tool was not available either; Git history,
`grep` and source reads were used.

### What was checked

1. **The combined patch, hunk by hunk.** Outside the repaired Vulkan tile every Saker change is
   gated on an environment variable the launcher does not set or removes. The installed engine is
   upstream `9d57ce456` plus the tile repair. Details: upstream delta report, section 1.
2. **Per-position fidelity, not just the mean.** The `run.log` files in `q38gsq-engine-audit.zip`
   hold the full distribution. Corrected Vulkan vs CPU: median KL 0.000118, 99.9% 0.110, max
   0.1428; the retune-disabled (pure upstream) arm has max 0.1485. No rare gross error hides under the
   mean. Vulkan is faithful to the CPU implementation of the same graph on this block.
3. **Upstream history from the pin (2026-08-14) to master (2026-11-01, 1,595 commits)** for every
   change touching the qwen35 graph, GDN, MTP/speculative decoding, recurrent rollback, Vulkan and
   the Qwen tool-call parser. Each relevant commit and its verdict for the installed profile is
   tabulated in the delta report. Two independent confirmations of earlier handoff findings:
   upstream fixed the rollback **test** to retain the anchor (`fa3c2fab3`, finding 5), and the
   `n_rs_seq`/checkpoint fixes (`7df8b876b`, `ab3768cdc`) are no-ops for `--spec-draft-n-max 2`
   with single-slot batches of at most three tokens (finding 6).
4. **Patch hygiene on a real checkout of the pin.** Fresh tree + combined patch, and old tree +
   quality fix, produce identical files. Recorded in
   [handoff-validation-20261102b.json](item23-20261102/handoff-validation-20261102b.json).
5. **Python tests:** 41 pass (37 existing, 4 new).

### What was added

- `patches/saker-gdn-l2norm-eps.patch`: backport of upstream #28068 (`5fdfa6282`). The pinned
  qwen35 graph normalizes GDN q/k with `ggml_l2_norm`, which is `x / max(sqrt(sum x²), eps)`; the
  reference implementation uses `x * rsqrt(sum x² + eps)`. This is the only upstream-confirmed
  deviation of the pinned graph from the reference for this architecture, and it is invisible to
  every CPU-versus-Vulkan comparison because both backends run the same graph. It applies on top of
  the combined patch, on top of the quality fix, or on the pristine pin, and `qwen35.cpp` compiles
  with it against the pinned headers. **Expected effect: far below the measured f16 noise floor**
  (relative change `eps / (2·Σx²)` per vector). It is a faithfulness fix. It is not claimed to fix
  the pelican, and it has not been built, measured or installed on node B.
- `reports/item23-20261102/reproduce/probe_engine.py` and `probe_project.py`: a `--backend cpu`
  arm that launches the same server with `-ngl 0 --device none --no-op-offload --no-kv-offload`
  (the flags finding 2 established as necessary on this UMA system), 16 threads by default, a
  longer startup allowance, plus `--ctx`, `--threads`, `--port` on the project replay. Covered by
  `tests/test_probe_harness.py`.
- `reports/item23-20261102/upstream-delta-20261102.md`: the commit survey.

### Assessment

The arithmetic and the serving/state paths that the installed profile exercises are faithful as far
as every available comparison can see. The remaining evidence of an "engine quality problem" is one
temperature-1 sample at the default effort, with about 250 tokens of reasoning before a 14 KB page,
failing a strict geometric QA on a prompt that is a known hard generative benchmark. That is weak
evidence of an engine defect and consistent with model/quantization capability (3.44 bits per weight).
This has **not** been proven either way; the experiments below separate the two.

### Next steps on node B, in order

Stop item 23 only while a step needs the GPU exclusively (`python "$SERVICE" stop` / `start`).

1. **CPU reference generation of the unchanged request** (decisive for the Vulkan path; expect well
   under 5 tokens/s, so roughly an hour):

   ```bash
   CPU=$(mktemp -d /tmp/item23-cpu-generation.XXXXXX)
   python "$REPRO/probe_project.py" --backend cpu --profile ar --effort medium --seed 42 \
     --tokens 16384 --out "$CPU"
   python "$REPRO/check_scene.py" "$CPU/pelican.html" --out "$CPU/qa"
   ```

   With temperature 1 the CPU and Vulkan samples will diverge at the first near-tie, so the
   comparison is "does the reference implementation also fail QA", not byte identity. If it fails
   the same checks, the engine is exonerated for this task.
2. **Variance on the installed profile:** repeat the Vulkan MTP2 replay for seeds 1-5 and at
   `--effort xhigh`, QA each page, keep every raw output. One failing draw is not a defect rate.
3. **Build the GDN patch in a separate build directory** (`build-gdn`), run `test-backend-ops` and
   the `vk-fixed` fidelity arm against the existing CPU reference, and compare its KL to the current
   0.002961. Expect no measurable change; only install if the figures are no worse and the
   generation replay still completes.
4. **Higher-precision reference for the quantization question** (handoff item 2): a Q8_0 or BF16
   GGUF of the same checkpoint on the CPU arm, `--kl-divergence-base` against IQ3_S. This is the
   only comparison that can attribute the residual to IQ3_S rather than to the engine.
5. **Rebase planning:** the pin is 1,595 commits behind. Upstream now has int8 coopmat1 matmul for
   AMD RDNA3/RDNA4 (#27952), IQ3_S MMQ kernels (#28822) and several Vulkan correctness fixes. A
   rebase supersedes maintaining the retune on an old base; re-measure the retune against #27952
   before carrying it forward.
