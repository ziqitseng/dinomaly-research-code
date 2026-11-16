"""Download the three published checkpoints used by the MVTec Dinomaly script."""
import hashlib
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gdown
import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "costfilter-ad/Costfilter_Dinomaly"
BASELINE_DIR = (
    "saved_results_mvtec/"
    "vitill_mvtec_uni_dinov2br_c392_en29_bn4dp2_de8_laelu_md2_i1_it10k_"
    "sams2e3_wd1e4_w1hcosa2e4_ghmp09f01w01_b16_s1"
)
FILES = [
    {
        "role": "dinov2_backbone",
        "relative_path": "backbones/weights/dinov2_vitb14_reg4_pretrain.pth",
        "expected_bytes": 346393545,
        "drive_id": "1zmVB6fTCsT1A-OoAwn-YThxjXBUEovWF",
        "direct_url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
    },
    {
        "role": "dinomaly_mvtec",
        "relative_path": BASELINE_DIR + "/model_10000.pth",
        "expected_bytes": 592137369,
        "drive_id": "1F0IH9iOqQWQguX6qqmZ8osXAWJZgXysF",
    },
    {
        "role": "costfilter_mvtec",
        "relative_path": "checkpoint_paths/mvtecad0.0001/costfilter_dinaomaly_mvtecad.pth",
        "expected_bytes": 174467764,
        "drive_id": "1G7glo7sb8l7-OsSW0dZRfwwVpWynYD-G",
    },
]


def download(spec):
    destination = MODEL_ROOT / spec["relative_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = spec.get("direct_url", "https://drive.google.com/file/d/" + spec["drive_id"] + "/view")
    if not destination.exists():
        partial = destination.with_name(destination.name + ".part")
        print("Downloading " + spec["role"], flush=True)
        if "direct_url" in spec:
            start = time.monotonic()
            received = 0
            progress = 0
            with requests.get(source, stream=True, timeout=(30, 60)) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            output.write(chunk)
                            received += len(chunk)
                            if received - progress >= 64 * 1024 * 1024:
                                print("%s: %.0f MB in %.1f s" % (spec["role"], received / 1e6, time.monotonic() - start), flush=True)
                                progress = received
        else:
            result = gdown.download(id=spec["drive_id"], output=str(partial), quiet=True, use_cookies=False, resume=True)
            if result is None:
                raise RuntimeError("Download did not complete: " + spec["role"])
        if partial.stat().st_size != spec["expected_bytes"]:
            raise RuntimeError("Unexpected file size for " + str(partial))
        if not zipfile.is_zipfile(partial):
            raise RuntimeError("Expected a PyTorch ZIP checkpoint: " + str(partial))
        partial.rename(destination)
    if destination.stat().st_size != spec["expected_bytes"]:
        raise RuntimeError("Existing file has unexpected size: " + str(destination))
    digest = hashlib.sha256()
    with destination.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    record = dict(spec, downloaded_from=source, sha256=digest.hexdigest())
    print("Verified size and recorded SHA-256: " + spec["role"], flush=True)
    return record


if __name__ == "__main__":
    records, failures = [], []
    with ThreadPoolExecutor(max_workers=3) as executor:
        pending = {executor.submit(download, spec): spec["role"] for spec in FILES}
        for future in as_completed(pending):
            try:
                records.append(future.result())
            except Exception as error:
                failures.append({"role": pending[future], "error": str(error)})
                print("Failed: %s: %s" % (pending[future], error), flush=True)
    manifest = {"repository_commit": "1900a2f", "files": records, "failures": failures}
    (ROOT / "costfilter_setup/weights_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sys.exit(bool(failures))
