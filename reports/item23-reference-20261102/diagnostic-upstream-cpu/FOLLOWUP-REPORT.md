# Upstream F32 and IQ3 request controls

All runs used explicit fixed token IDs, full-vocabulary logits, CPU-only builds, 16 threads, F32 K/V caches, disabled flash attention and disabled operation/KV offload. The upstream arm used commit `785322d17f2d51e5185cf996a1f4e46d87640841` and the same F32 GGUF converted from the official BF16 checkpoint used by the HF F32 reference. CPU backend allocation is recorded in each log. Raw rows, run commands, metadata, full comparison JSON and hashes are retained alongside this report.

| Upstream F32 vs HF F32 | Positions | Mean KL | Maximum absolute logit delta | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| Short prefill | 2 | 7.60e-12 | 0.0000215 | 2/2 |
| Short cached | 2 | 4.75e-12 | 0.0000238 | 2/2 |
| Request, chunked prefill | 8 | 8.76e-12 | 0.0002356 | 8/8 |
| Request, cached | 8 | 3.28e-12 | 0.0001087 | 8/8 |

Request chunked prefill uses 256-token chunks; cached mode uses the same chunking through the first selected position, followed by individual tokens. These are short selected positions, not a full-context, stochastic, Vulkan or MTP validation. No upstream long run was performed because these bounded comparisons showed no discrepancy.

IQ3 CPU controls were also rerun on the request using those same 256-token boundaries. Against official HF F32, all eight selected top tokens agree. Mean KL was 0.00942 and 0.00940 for unpatched prefill and cached; 0.01051 and 0.01751 for patched prefill and cached. The IQ3 model and official checkpoint are different artifacts whose base-weight identity has not been established, so these differences cannot be assigned solely to quantization. The metrics do not establish a cause for the generated-page issue.
