# ArcAD smoke: deterministic CUDA gather backward

The failure occurred at the synthetic normal loss backward pass. PyTorch 1.12's installed `torch.use_deterministic_algorithms` documentation identifies CUDA `gather` with differentiable inputs as unsupported in strict mode. ArcAD's original DGC `augment_features` chooses one of 50 candidates per anchor using such a gather. Its backward invokes the reported scatter-add kernel.

The comparison adapter now subclasses the original prototype learner and replaces only this selection implementation. Original model files are unchanged. Strict `torch.use_deterministic_algorithms(True)` remains enabled in both smoke and training; neither `warn_only` nor a global determinism relaxation is used.

1. Draw the same candidate-noise tensor with the same dimensions and RNG consumption.
2. Use the same normalization, prototype similarities, and argmin selection, without recording gradients for this discrete search.
3. Gather the selected **noise**, which never requires gradients.
4. Recompute `normalize(anchor + noise_std * selected_noise)` with gradients enabled. Only this chosen branch contributed to the original gather derivative; gradients still flow to the anchor/bottleneck. No synthetic anomaly loss is detached or omitted.

This is the same candidate-selection algorithm and mathematical derivative, expressed without a differentiable gather. It also avoids storing candidate-search autograd state. Candidate count 50, noise 0.015, prototypes, losses, weights, seeds, sampling streams, batch sizes, and training steps remain unchanged.

`scripts/check_deterministic_selection.py` compares the implementation against the exact original method extracted from the pinned ArcAD source. It checks selected outputs, anchor gradients, RNG state, and exact repeated strict-mode results in FP32/FP64. On CUDA only, the old reference is permitted to run without strict checks so its unsupported gather backward can be evaluated; the replacement always runs with strict checks. That reference-only setting is never used for fitting.

CPU regression checks pass. GPU regression and full synthetic smoke validation are recorded under `logs/` when run. The `xFormers is disabled/not available` warnings are expected because the experiment explicitly disables xFormers; they are unrelated to this error.

The complete SSD preparation is unchanged and does not need to be rebuilt. Pre-fix code/config/docs/lock and the prepared completion fingerprint are preserved under `provenance/revisions/before_arcad_deterministic_selection_fix/`. No real training/evaluation is started by this fix.

Retry from the comparison folder:

```bash
bash run.sh smoke arcad --device cuda:0
```
