# Preflight dependency and adapter fix

The reported failure was caused by this experiment unnecessarily requiring `natsort`. That package is used by Dinomaly2's original video dataset loader; this comparison reads its own PNG cache and never imports that loader. Removed the unused natsort/tifffile preflight requirements. No package installation, environment change, or original-repository edit is needed.

Also corrected the adapter to respect the different encoder loader APIs: ArcAD `load(name)` and Dinomaly2 `load(name, WEIGHTS_DIR=...)`. Both still resolve the same pinned local pretrained weights through the download-blocking resolver. Preflight now imports the actual runtime losses/classes/optimizer and checks the loader argument binding, without constructing a model or loading weights.

Actual repository imports passed in the existing ArcAD environment, where natsort remains absent. Thirteen synthetic tests passed, including two new API/dependency regression tests. The pre-fix code, documentation, and lock are archived under `provenance/revisions/before_preflight_dependency_fix/`; the current lock supersedes that setup. Protocol B labels, split rules, budgets, source payload, and original model sources are unchanged.

Retry from this folder:

```bash
bash run.sh preflight
```

Then continue one command at a time from `RUN_COMMANDS.txt`. No preparation, model smoke check, prototype fitting, training, or evaluation is launched by this fix.
