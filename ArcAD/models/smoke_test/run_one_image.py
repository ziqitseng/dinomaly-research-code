#!/usr/bin/env python3
"""Run one Real-IAD image through ArcAD's reconstruction inference path.

This is a wiring smoke test, not a quality evaluation: the decoder and
bottleneck are randomly initialized unless a trained checkpoint is added.
"""

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
# The encoder weight cache in models/vit_encoder.py is relative to the cwd.
os.chdir(REPO_ROOT)

import numpy as np
import torch

from dataset import RealIADDataset, get_data_transforms
from gen_protos import get_feature_extractor
from utils import cal_anomaly_maps


def main():
    default_data = REPO_ROOT.parent / "dataset" / "Real-IAD"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=default_data)
    parser.add_argument("--category", default="audiojack")
    parser.add_argument("--split", choices=("labeled", "test"), default="test")
    parser.add_argument("--index", type=int, default=None,
                        help="Dataset index; by default, use the first anomalous image (or index 0).")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable to this process.")

    image_transform, mask_transform = get_data_transforms(size=448, isize=392)
    dataset = RealIADDataset(
        root=str(args.data_path),
        category=args.category,
        transform=image_transform,
        gt_transform=mask_transform,
        phase=args.split,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found for {args.category}/{args.split}.")

    if args.index is None:
        anomalies = np.flatnonzero(dataset.labels)
        index = int(anomalies[0]) if len(anomalies) else 0
    else:
        index = args.index
    if index < 0 or index >= len(dataset):
        raise IndexError(f"Index {index} is outside {args.category}/{args.split} (size {len(dataset)}).")

    image, mask, label, image_path = dataset[index]
    model = get_feature_extractor(args.device).eval()

    with torch.no_grad():
        encoder_features, decoder_features, bottleneck = model(image.unsqueeze(0).to(args.device))
        anomaly_map, _ = cal_anomaly_maps(
            encoder_features, decoder_features, out_size=image.shape[-1]
        )

    if anomaly_map.shape != (1, 1, image.shape[-1], image.shape[-1]):
        raise RuntimeError(f"Unexpected anomaly-map shape: {tuple(anomaly_map.shape)}")
    if not torch.isfinite(anomaly_map).all():
        raise RuntimeError("The anomaly map contains NaN or infinite values.")

    flattened = anomaly_map.flatten()
    top_count = max(1, int(flattened.numel() * 0.01))
    score = flattened.topk(top_count).values.mean().item()

    print(f"Device: {args.device}")
    if args.device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(torch.device(args.device))}")
    print(f"Sample: {image_path}")
    print(f"Label: {int(label)} | mask shape: {tuple(mask.shape)}")
    print(f"Encoder maps: {[tuple(x.shape) for x in encoder_features]}")
    print(f"Decoder maps: {[tuple(x.shape) for x in decoder_features]}")
    print(f"Bottleneck: {tuple(bottleneck.shape)}")
    print(f"Anomaly map: {tuple(anomaly_map.shape)} | finite: yes | top-1% score: {score:.6f}")
    print("Smoke test passed. The score is not meaningful until the ArcAD model is trained.")


if __name__ == "__main__":
    main()
