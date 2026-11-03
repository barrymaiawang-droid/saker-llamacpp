# Item 23: independent reference results on node A

The GDN normalization correction is verified and included in the combined and quality patches. The patched engine closely matches an independent Transformers implementation on the tested contexts. **The pelican generation-quality problem remains unresolved.** The unedited patched replay still has scene and animation errors.

Three GPT-6.1 Terra subagents performed the investigation and independent result review. All model computation ran on node A. Production on node B and the user's original page were unchanged. This isolated worktree has not been committed, pushed or deployed.

## Engine findings

The pinned graph used `x / max(sqrt(sum(x*x)), eps)` for GDN Q/K normalization. The model reference uses `x * rsqrt(sum(x*x) + eps)`. The repair expresses the reference formula through RMS normalization and scaling.

On 7,680 captured Q/K vectors, the original native operation differed from PyTorch by up to 0.0016449; the corrected operation's maximum difference was 1.79e-7. The test exercised native GGML operations on actual model inputs.

For the independent model comparison, all 18 official `Qwen/Qwen3.8-27B` shards at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` passed their pinned checksums. The same weights were run in Transformers float32 and converted to F32 GGUF for llama.cpp. All 866 GGUF tensors are float32; both full embedding/output matrices and 13 selected transformed tensors matched their source. Token ordering, including 243 padded output rows, was verified.

Both implementations used identical explicit token IDs, full-vocabulary logits, CPU execution and float32 arithmetic. Native KV storage was float32, flash attention and offload were disabled, and no sampling or MTP was used in these comparisons. The original engine and epsilon-patched engine differ only by the normalization repair.

| Context / mode | Measured positions | Original mean KL | Patched mean KL |
| --- | ---: | ---: | ---: |
| 5-token calibration, prefill | 2 | 1.243e-9 | 4.235e-12 |
| Calibration, cached | 2 | 1.237e-9 | 3.962e-12 |
| 367-token original request, prefill | 8 | 1.686e-9 | 2.694e-12 |
| Request, cached | 8 | 1.514e-9 | 4.706e-12 |
| 4,642-token HTML context, prefill | 13 | 5.211e-8 | 7.224e-9 |
| HTML context, cached | 13 | 5.191e-8 | 7.057e-9 |

KL is computed from Transformers to native over all 248,320 vocabulary entries. Every measured top-token choice matched in every arm. The long patched outputs remain farther from Transformers than its own prefill/cached comparison (mean KL 6.188e-12), but their absolute discrepancies are small; maximum per-row KL is 7.460e-8. No second shared-graph defect was established by these tests. An independent agent recomputed the results and checked IDs, vocabulary, positions and chunk boundaries.

These are selected positions in three contexts, not proof of every runtime path, sampling behavior or production context length. The HTML was an unchanged standalone teacher-forced context, not the original sampled conversation. The joint probability of the two tokens forming `Wheel` was about 0.000511 in both Transformers and native F32; that observation does not establish the original error's cause.

## Quality and model effects

The paired Vulkan replay used the original request, IQ3 weights, seed, sampling, MTP2 and matching settings. The baseline reproduced the previous page byte-for-byte. The patched page rendered, and its spokes fit the wheel, but it still had awkward bird geometry and broken road-mark motion. Neither generated HTML was edited. See the [replay report](item23-reference-20261102/vulkan-replay/REPORT.md) and the [patched screenshot](item23-reference-20261102/vulkan-replay/candidate/qa.png).

With identical request chunk boundaries, IQ3 differed from the official reference by mean KL 0.0094–0.0175, while retaining all eight measured top tokens. This is much larger than the same-weight F32 discrepancy. It combines GSQ optimization, mixed-precision weights and quantized arithmetic. Twenty-seven retained F32 tensors match the official checkpoint, supporting common lineage, but the release does not establish exact whole-model source identity. These results do not prove bit width caused the pelican failure.

The next useful quality experiment would be a controlled higher-precision model replay with the same request and engine. The current evidence does not justify another speculative engine change or a claim that the scene is fixed.

## Speed candidate

An isolated upstream build at `785322d17f2d51e5185cf996a1f4e46d87640841` was compared with corrected Saker plus the normalization patch.

| Fixed MTP2 workload | Corrected Saker | Upstream candidate |
| --- | ---: | ---: |
| Prompt processing | 249.13 tokens/s | 318.37 tokens/s |
| Generation | 32.04 tokens/s | 32.89 tokens/s |

This used one exact 2,048-token prefix, 128 greedy output tokens and three warm runs per build. All six output token sequences were identical. The measured gains are approximately 27.8% for prompt processing and 2.6% for generation on this workload. Runs were grouped by build; this is not a general production speed claim. A separate interleaved combined autoregressive benchmark showed about 7.17% higher throughput.

The upstream CPU F32 candidate also matched all selected short/request reference top tokens in both modes, with mean KL 3.28e-12–8.76e-12. Its Vulkan server remains available on node A at `<NODE_A_HOME>/item23-independent-reference-20261102/benchmark-upstream/build/bin/llama-server`. It is a tested candidate, not a deployed upgrade. See the [MTP report](item23-reference-20261102/diagnostic-upstream-gpu/REPORT.md) and [independent follow-up](item23-reference-20261102/diagnostic-upstream-cpu/FOLLOWUP-REPORT.md).

## Patch and evidence

The [README](../README.md) gives mutually exclusive application paths. Fresh pin plus the updated combined patch and pre-repair Saker plus the updated quality patch produce the same source. The model files and repaired Vulkan source match the tested candidate exactly. All 41 project tests pass; see [patch validation](item23-20261102/patch-package-validation-20261102.md).

The [review evidence bundle](item23-reference-20261102/README.md) includes comparison JSON, fixtures, metadata, source checks and hashes of the raw logit archives. Large model files and raw logit archives remain in the original investigation locations. All owned node A inference/conversion jobs have exited; the verified checkpoint and F32 GGUF are retained, with 43 GiB free disk reported at completion.
