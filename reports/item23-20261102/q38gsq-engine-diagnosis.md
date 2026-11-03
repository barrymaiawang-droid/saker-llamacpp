Item 23 engine diagnosis from the pelican failure — 2026-11-02, node B

The engine had a reproducible numerical bug, and the existing Vulkan patch fixes that bug. This follow-up validates the engine against an independent CPU implementation using the original, unedited pelican code. It does not count a manually repaired page or a more specific art prompt as an engine fix.

The defective Saker prefill optimization widened the matrix block without widening its per-warp coverage, leaving columns unwritten. It also used dispatch dimensions inconsistent with the floating-point and integer-dot kernels. The installed patch corrects coverage and derives dispatch sizes from the actual kernel tiles. The earlier CPU-reference operator regressions passed 80/80 matrix cases and 50/50 recurrent-attention cases after the repair.

The new comparison scores 1,279 next-token predictions within the first 2,560 tokens of the original pelican source. All candidates use the same GGUF weights and exact input token IDs. The reference explicitly disables GPU devices, operation offload and KV offload. Lower KL divergence means closer probability distributions; top-token agreement measures how often the highest-scored token matches the CPU. These are inference-fidelity measurements, not visual-quality scores.

| Runtime | Mean KL divergence from CPU | Top-token agreement |
|---|---:|---:|
| Preserved engine before the patch | 6.271537 | 42.377% |
| Corrected Vulkan, installed configuration | 0.002961 | 98.436% |
| Corrected Vulkan, retune disabled | 0.002832 | 98.593% |
| Vulkan, float32 diagnostic configuration | 0.001687 | 99.062% |
| HIP comparison build | 0.432554 | 81.392% |

The corrected configuration reduces this measured numerical discrepancy by 99.953% relative to the preserved old runtime. Its remaining difference is comparable to Vulkan with the retune disabled. It is not bit-identical to CPU. The float32 diagnostic has a slightly closer distribution; it is not the installed speed profile.

HIP initially appeared to reveal a large remaining Vulkan error. The CPU comparison shows that HIP itself is a poor reference for this checkpoint and workload in this build. Its separate discrepancy is unresolved and no HIP fix is claimed. Item 23 uses Vulkan. Disabling Vulkan flash attention, graph optimization, fusion and asynchronous execution did not explain the HIP/Vulkan disagreement; disabling HIP graph capture did not resolve it either.

Additional engine checks used identical, frozen token inputs across four execution paths: corrected Vulkan, retune-disabled Vulkan, HIP and full-vocabulary Vulkan MTP2. All 56 requests completed. On both 128-token generation probes, MTP2 matched the HIP outputs in all three repetitions. Ordinary Vulkan diverged at close-scoring alternatives. These short matches are not proof of completed-page quality. Causality checks at 256 and 1,536 input tokens found that changing future tokens left earlier Vulkan logits unchanged; identical repeats were also unchanged in those checks.

On node B without the former item-22 service running, median decode rates over three 128-token repetitions were:

| Probe | Corrected ordinary decoding | Full-vocabulary MTP2 |
|---|---:|---:|
| Bicycle geometry | 14.975 tokens/s | 26.209 tokens/s |
| Original pelican request | 14.978 tokens/s | 28.284 tokens/s |

That is approximately 75% and 89% faster decoding on these short probes. The requests returned token probabilities and used cold request caches; these are observed rates, not an unlimited-context throughput guarantee. The current long-prefill comparison did not reproduce the earlier retune speed advantage: corrected ordinary decoding measured about 280 tokens/s versus 288 with the retune disabled. No new prefill speedup is claimed. Numerical diagnostic runs are excluded from speed comparisons.

The original visual failure contains incorrect wheel orientation/spin axes and poor bird geometry. Numerical corruption could contribute to bad generation, but this experiment cannot assign every visual mistake to that corruption. The corrected engine still cannot guarantee a well-proportioned pelican. The remaining visual failure has not been isolated to another defect in the installed Vulkan engine. Changing the HTML would not establish that.

The earlier claim that the server capped `max_tokens: 12288` at 6,144 was wrong. A newer interrupted stream was compared with an older saved response. The pinned server accepts `max_tokens`, `max_completion_tokens` and `n_predict` aliases. That supposed cap is excluded from the diagnosis.

All seven installed runtime libraries match the hashes of the corrected Vulkan libraries used in this audit. The installed profile retains full-vocabulary MTP2, the native chat template and medium reasoning effort; the experimental shortlist remains disabled. Item 23 has been started on node B at port 18151 and its health/API response is recorded in the evidence bundle. The configured context is 262,144 tokens. The earlier numerical source repair was already installed; this follow-up adds the controlled engine validation, not another unproven engine patch.

The original pelican file was not modified in this follow-up. Its SHA256 remains `7cb912dead4b640bc4554b9384659d858552de92a6b2d8efb8b1bddb4d5b7733`. The model SHA256 remains the previously verified `58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f`.

Evidence handling: the CPU run requested two blocks, but was intentionally stopped after its first complete block. The extracted reference retains that block's exact token IDs and probability data, with the block-count header changed from two to one. Only the completed block is counted. The second block is excluded. An earlier `-ngl 0` control still allowed GPU operation offload and is also excluded as a CPU reference. A tensor-dump attempt started before compilation finished is excluded; its replacement is identified in the machine-readable summary. No missing or interrupted run is treated as a pass.

Review files: `q38gsq-engine-diagnosis.json` contains measurements and limitations; `q38gsq-engine-audit.zip` contains requests, responses, logs, manifests and reproduction scripts, excluding large binary tensor/logit dumps. `q38gsq-engine-fix.patch` contains the engine correctness delta. `q38gsq-combined.patch` is the alternative full patch against the pinned llama.cpp base; apply one patch or the other, not both. Previous edited pelican pages are not part of the engine validation.
