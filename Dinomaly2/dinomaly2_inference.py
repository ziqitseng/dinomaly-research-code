#!/usr/bin/env python3
"""Evaluate a trained Dinomaly2 Real-IAD checkpoint on the official test split."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import RealIADDatasetv2, get_data_transforms
from dinov2.models import vision_transformer as dinov2_vit
from models.uad import Dinomaly
from models.vision_transformer import Block as VitBlock, LinearAttention2
from utils import evaluation_batch_multiview


REPO_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = REPO_DIR.parent
DEFAULT_DATASET = WORKSPACE_DIR / "dataset" / "Real-IAD"
DEFAULT_CHECKPOINT = REPO_DIR / "out" / "real_iad out" / "train" / "model.pth"
DEFAULT_OUTPUT = REPO_DIR / "out" / "real_iad out" / "inference_metrics"
METRIC_NAMES = (
    "I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO",
    "O-AUROC", "O-AP", "O-F1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Dinomaly2 checkpoint using Real-IAD's test split and metrics."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATASET,
                        help="Dataset root containing category images and split JSON files.")
    parser.add_argument("--dataset-type", choices=("realiad", "realiad-variety"), default="realiad",
                        help="Select the regular Real-IAD or Real-IAD Variety directory/split layout.")
    parser.add_argument("--category", action="append", dest="selected_categories",
                        help="Evaluate this category; repeat the option to evaluate several. Default: all JSON categories.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=392)
    parser.add_argument("--map-size", type=int, default=256,
                        help="Evaluation map and mask size (training log uses 256).")
    parser.add_argument("--score-top-percent", type=float, default=1.0,
                        help="Top map-pixel percentage for image/object scores; training log uses 1.0.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def make_model(device: torch.device) -> Dinomaly:
    """Recreate the trained ViT-B/14-register, LA, LC=2 Dinomaly2 architecture."""
    encoder = dinov2_vit.vit_base(
        patch_size=14,
        img_size=518,
        block_chunks=0,
        init_values=1e-8,
        num_register_tokens=4,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )
    embed_dim, dropout = 768, 0.4
    bottleneck = nn.ModuleList([
        nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=dropout)),
        nn.Sequential(
            nn.Linear(256, embed_dim * 4), nn.GELU(), nn.Dropout(p=dropout),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(p=dropout),
        ),
    ])
    decoder = nn.ModuleList([
        VitBlock(
            dim=embed_dim,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=partial(LinearAttention2, eps=1e-8),
        )
        for _ in range(8)
    ])
    model = Dinomaly(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
        remove_class_token=False,
        fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        context_aware_recenter=True,
    )
    model.init_weights()
    return model.to(device)


def main() -> None:
    args = parse_args()
    if min(args.image_size, args.crop_size, args.map_size, args.batch_size) <= 0:
        raise ValueError("Image size, crop size, map size, and batch size must be positive.")
    if args.crop_size % 14:
        raise ValueError("crop-size must be divisible by 14 for the ViT-B/14 backbone.")
    if not 0 < args.score_top_percent <= 100:
        raise ValueError("score-top-percent must be in the range (0, 100].")
    if args.workers < 0:
        raise ValueError("workers cannot be negative.")

    data_root = args.data_path.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if args.dataset_type == "realiad-variety":
        split = "realiadvariety_jsons/Real-IAD_Variety_jsons"
        version = "."
    else:
        split = "realiad_jsons/realiad_jsons"
        version = "realiad_1024"
    json_dir = data_root / split
    if not json_dir.is_dir():
        raise FileNotFoundError(f"Real-IAD split JSON directory not found: {json_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    categories = sorted(path.stem for path in json_dir.glob("*.json"))
    if not categories:
        raise FileNotFoundError(f"No category split files found in: {json_dir}")
    if args.selected_categories:
        unknown = sorted(set(args.selected_categories) - set(categories))
        if unknown:
            raise ValueError(f"Unknown category/categories for {args.dataset_type}: {', '.join(unknown)}")
        categories = [category for category in categories if category in set(args.selected_categories)]

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to this process.")
    device = torch.device(
        "cuda:0" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    # Dinomaly2's fast evaluator uses CUDA tensors internally (including .cuda()).
    if device.type != "cuda":
        raise RuntimeError(
            "The Dinomaly2 log-compatible evaluator requires CUDA. Run with a visible GPU, "
            "or use the training script's evaluation_batch_multiview with adeval=False "
            "for CPU evaluation."
        )

    data_transform, gt_transform = get_data_transforms(args.image_size, args.crop_size)
    model = make_model(device)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    category_results: dict[str, dict[str, float]] = {}
    print(f"Checkpoint: {checkpoint}")
    print(f"Real-IAD root: {data_root}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Device: {device}; categories: {len(categories)}")
    print(
        f"Evaluation: test split; maps {args.map_size}x{args.map_size}; "
        f"top {args.score_top_percent:g}% for image/object scores"
    )

    for category in categories:
        test_data = RealIADDatasetv2(
            root=str(data_root),
            category=category,
            transform=data_transform,
            gt_transform=gt_transform,
            phase="test",
            split=split,
            version=version,
        )
        loader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        metrics = evaluation_batch_multiview(
            model,
            loader,
            device,
            max_ratio_image=args.score_top_percent / 100.0,
            max_ratio_object=args.score_top_percent / 100.0,
            resize_mask=args.map_size,
            adeval=True,
        )
        category_results[category] = {
            name: float(value) for name, value in zip(METRIC_NAMES, metrics)
        }
        values = category_results[category]
        print(category + ": " + ", ".join(f"{name}:{value:.4f}" for name, value in values.items()))

    mean_results = {
        name: float(np.mean([row[name] for row in category_results.values()]))
        for name in METRIC_NAMES
    }
    print("Mean: " + ", ".join(f"{name}:{value:.4f}" for name, value in mean_results.items()))

    report = {
        "checkpoint": str(checkpoint),
        "dataset": str(data_root),
        "dataset_type": args.dataset_type,
        "split": "test",
        "categories": category_results,
        "mean": mean_results,
        "settings": {
            "image_size": args.image_size,
            "crop_size": args.crop_size,
            "map_size": args.map_size,
            "score_top_percent": args.score_top_percent,
            "gaussian_smoothing": {"kernel_size": 5, "sigma": 4},
            "evaluator": "Dinomaly2 evaluation_batch_multiview (adeval=True)",
        },
    }
    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics report: {report_path}")


if __name__ == "__main__":
    main()
