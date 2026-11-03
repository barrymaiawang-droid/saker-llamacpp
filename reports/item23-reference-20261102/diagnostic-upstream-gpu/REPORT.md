# Upstream Vulkan fixed-token and MTP checks

The existing corrected Saker plus GDN-epsilon build and isolated upstream commit `785322d17f2d51e5185cf996a1f4e46d87640841` used the same verified IQ3 model. Production runtimes were not modified.

Full-vocabulary Vulkan prefill logits were compared on explicit fixed tokens, with F32 K/V cache, flash attention disabled, GPU and operation offload enabled, and identical batch settings. Top tokens agreed at all two short-fixture and eight 367-token-request positions. Upstream versus patched mean KL was 0.000195 for the short fixture and 0.001175 for the request positions; the largest request-position KL was 0.004736. These are relative differences on selected inputs, not independent evidence of correctness or full long-context fidelity.

The MTP2 test used the same 2,048-token prefix in both servers, `n_predict=128`, temperature zero, ignore EOS, prompt cache disabled, context 32768, full GPU offload, flash attention on, batch 2048, microbatch 256 and four threads. Each server performed a warmup followed by three measured requests. Runs were grouped by build, not interleaved. All six requests processed 2,048 uncached prompt tokens and generated 128 identical greedy token IDs (token-array SHA256 `cf3caf905daa8463f3bd14cab1f273e7ce9973a9f5b6daf91c7e61170a2e8e0d`). Draft proposal and acceptance counts were 93–95 and 79–80 respectively.

| Build | Prompt samples (tokens/s) | Generation samples (tokens/s) |
| --- | --- | --- |
| Corrected Saker + epsilon | 246.38, 250.95, 250.07 | 32.00, 32.50, 31.63 |
| Pinned upstream | 320.06, 316.02, 319.03 | 32.47, 33.07, 33.13 |

Means were 249.13 vs 318.37 prompt tokens/s and 32.04 vs 32.89 generated tokens/s. This is one prefix, six grouped runs, and shared desktop GPU conditions; it is not a general production speed claim or a deployment recommendation. Identical greedy outputs do not establish complete distributional, stochastic, long-context or official-model fidelity. The official-weight float32 reference remains pending.

Raw logits, full responses, command lines, prefix, logs, library hashes, model identity and server build logs are retained here. Both owned servers exited and the pre-existing desktop compositor remained active.
