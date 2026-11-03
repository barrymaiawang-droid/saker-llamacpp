# Item 23 patch package validation

Status: application paths verified and the selected long-context F32 reference checks completed. This package is not deployed. The original user-facing generation-quality issue remains unresolved.

The GDN epsilon delta is now part of both `saker-combined.patch` and `saker-quality-fix.patch`. The unchanged standalone `saker-gdn-l2norm-eps.patch` is for a tree that already has the earlier Vulkan tile repair but lacks the GDN change. The README specifies the three mutually exclusive starting states; do not apply the standalone delta after either updated default patch.

Validation used isolated trees derived from pinned llama.cpp `9d57ce456c94d241dde672b2db9cf18879766568`:

1. Applied the prior combined patch and the standalone epsilon delta to make the expected corrected tree.
2. Applied the updated combined patch to a separate fresh pin. `git apply --check` and application passed; recursive comparison with the expected tree found no differences.
3. Reconstructed the pre-repair Saker tree by applying the prior combined patch and reversing the prior quality delta. The updated quality patch passed `git apply --check` and application; recursive comparison with the expected tree found no differences.
4. Compared the expected tree with the earlier corrected Saker tree. Only `src/models/models.h` and `src/models/qwen35.cpp` differed. Those files and `ggml/src/ggml-vulkan/ggml-vulkan.cpp` are byte-identical to the isolated patched candidate used in the prior tests. This retains the prior Vulkan fix and all other combined-patch changes.
5. The standalone delta passed `git apply --check` on a tile-repaired tree without the epsilon fix. Reverse checks passed on both updated trees, and a second forward application was rejected as expected.
6. `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v` passed all 41 tests. These project tests do not substitute for model validation.

The independent same-weight F32 comparison has now completed for the selected 4,642-token HTML prefix, at 13 positions (4629 through 4641). Against the HF F32 reference, patched CPU chunked prefill and cached decoding both agreed on all 13 top tokens; their mean KL divergences were `7.22405e-9` and `7.05738e-9`, respectively, and maximum absolute logit deltas were `0.001581` and `0.001567`. This is evidence for these selected positions and paths, not proof for every long context or a claim that the page quality issue is solved.

SHA256 of the patch files:

```text
e2b81c338b2242696cc620c008fd26248269e3cff753474198d60a85fae3ff75  saker-combined.patch
82e99d57e5ae9aff57b0124f8cf843edb747f266f7ccaade163e6780431d2b5f  saker-quality-fix.patch
9289f34be1cf0a766a5c19e2e3f743f9ebfacecdfc80cb0594966cc664a98bf3  saker-gdn-l2norm-eps.patch
```

`git diff --check` flags added single-space blank context lines inside the patch files. These are unified-diff context markers, not trailing whitespace in the patched source; `git diff --no-index --check` on the source trees emitted no whitespace diagnostics, and both application checks and comparisons above pass. The README and this report pass whitespace checking separately.
