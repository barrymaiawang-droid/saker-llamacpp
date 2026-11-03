# Pinned upstream Vulkan benchmark

An isolated build of upstream llama.cpp commit `785322d17f2d51e5185cf996a1f4e46d87640841` was compared with the existing corrected Saker build plus the GDN epsilon patch. The upstream revision also contains epsilon-style Q/K normalization. No production runtime was changed.

Six independent `llama-bench` invocations ran in order A-B-B-A-A-B, with one warmup and one measured run each. All used the same verified IQ3 model, full GPU offload, flash attention, four threads, batch 2048, microbatch 256, and a combined 2048-token prompt plus 128-token generation. This is ordinary autoregressive decoding at a workload-sized context, not MTP serving throughput.

| Build | Combined throughput samples (tokens/s) | Mean |
| --- | --- | --- |
| Corrected Saker + epsilon | 149.703, 148.291, 146.850 | 148.281 |
| Pinned upstream | 159.344, 158.563, 158.820 | 158.909 |

The upstream mean was 7.17% higher for this one workload. Three samples per variant do not establish a general production speed gain. Temperature rose from 49 to 70 C and reported GPU clocks varied; the existing desktop compositor was active. At this series' completion, numerical fidelity, MTP behavior and generation quality remained untested. Subsequent bounded MTP and GPU checks are in `../diagnostic-upstream-gpu/REPORT.md`, and independent short/request F32 comparisons are in `../diagnostic-upstream-cpu/FOLLOWUP-REPORT.md`. Those follow-ups do not establish a general production speed or generation-quality result; the build remains a candidate, not a deployed change.

The result JSON, per-run commands and telemetry, source revision and hashes, build logs and compiler flags are retained here. The full isolated source and build remain on node A under `~/item23-independent-reference-20261102/benchmark-upstream`.
