"""Strictly load the published MVTec checkpoints and check a GPU forward pass."""
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.dont_write_bytecode = True
SETUP = Path(__file__).resolve().parent
# Reuse the already verified architecture and forward pass without running
# that script's random-weight benchmarks or duplicating model definitions.
source = SETUP / "verify_inference.py"
tree = ast.parse(source.read_text())
assert isinstance(tree.body[-1], ast.For)
tree.body.pop()
namespace = {"__file__": str(source), "__name__": "checkpoint_verification"}
exec(compile(tree, str(source), "exec"), namespace)
model, filter_model = namespace["model"], namespace["filter_model"]
model_root = namespace["repo"]

manifest_path = SETUP / "weights_manifest.json"
manifest = json.loads(manifest_path.read_text())
assert not manifest["failures"] and len(manifest["files"]) == 3
specs = {spec["role"]: spec for spec in manifest["files"]}

for role, target, state_key in [
    ("dinov2_backbone", model.encoder, None),
    ("dinomaly_mvtec", model, None),
    ("costfilter_mvtec", filter_model, "model_state_dict"),
]:
    spec = specs[role]
    path = model_root / spec["relative_path"]
    assert path.stat().st_size == spec["expected_bytes"]
    # Match the upstream loader for these official checkpoints. Torch 1.13's
    # restricted weights-only loader cannot rebuild the backbone Parameters.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state_key:
        state = state[state_key]
    result = target.load_state_dict(state, strict=True)
    print("%s: %d state entries; %s" % (role, len(state), result), flush=True)
    spec["state_entries"] = len(state)
    spec["strict_load_verified"] = True
    del state

model.eval()
filter_model.eval()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
shape = namespace["forward"](1)
torch.cuda.synchronize()
verification = {
    "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(),
    "input": [1, 3, 392, 392],
    "output": list(shape),
    "finite_output": True,
    "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    "note": "Random input with published weights; no dataset accuracy evaluation.",
}
manifest["verification"] = verification
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(verification, indent=2), flush=True)
