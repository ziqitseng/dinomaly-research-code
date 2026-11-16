#!/usr/bin/env python3
"""Plot the lowest-scoring defective test objects for selected Real-IAD Variety categories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop

from dataset import RealIADDatasetv2, get_data_transforms
from dinomaly2_inference import make_model
from utils import cal_anomaly_maps, get_gaussian_kernel


DEFAULT_DATA = Path("/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024")
DEFAULT_CHECKPOINT = Path("out/real_iad_variety_1024 out/full run/variety_160cat/model.pth")
DEFAULT_OUTPUT = Path("out/real_iad_variety_1024 out/full run/variety_160cat/review_low3/selected_examples_10_per_category")
DEFAULT_CATEGORIES = ("lattice_block_plug", "laser_diode", "power_jack")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--category", action="append", dest="categories",
                        help="Category to include; repeat as needed. Defaults to the three lowest I-AUROC categories.")
    parser.add_argument("--count-per-category", type=int, default=10,
                        help="Number of lowest-scoring defective objects to plot from each selected category.")
    parser.add_argument("--image-size", type=int, default=280)
    parser.add_argument("--crop-size", type=int, default=280)
    parser.add_argument("--map-size", type=int, default=256)
    parser.add_argument("--score-top-percent", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count_per_category < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("count-per-category and batch-size must be positive; workers cannot be negative.")
    if min(args.image_size, args.crop_size, args.map_size) <= 0 or args.crop_size % 14:
        raise ValueError("Image/map sizes must be positive and crop-size divisible by 14.")
    if not 0 < args.score_top_percent <= 100:
        raise ValueError("score-top-percent must be in (0, 100].")

    root = args.data_path.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    split = "realiadvariety_jsons/Real-IAD_Variety_jsons"
    json_dir = root / split
    if not json_dir.is_dir():
        raise FileNotFoundError(f"Variety split JSON directory not found: {json_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    categories = args.categories or list(DEFAULT_CATEGORIES)
    unknown = [name for name in categories if not (json_dir / f"{name}.json").is_file()]
    if unknown:
        raise ValueError(f"No Variety split JSON for categories: {', '.join(unknown)}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type != "cuda":
        raise RuntimeError("Dinomaly2's map scoring path requires CUDA in this repository.")

    data_transform, gt_transform = get_data_transforms(args.image_size, args.crop_size)
    display_transform = Compose([Resize((args.image_size, args.image_size)), CenterCrop(args.crop_size)])
    model = make_model(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    candidates = []
    test_sets = {}

    # First pass: score every held-out object using the same pooled top-pixel
    # object score used by evaluation_batch_multiview in the training log.
    for category in categories:
        test_data = RealIADDatasetv2(
            root=str(root), category=category, transform=data_transform,
            gt_transform=gt_transform, phase="test", split=split, version=".",
        )
        test_sets[category] = test_data
        loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        cursor = 0
        with torch.inference_mode():
            for images, _masks, labels, _image_paths in loader:
                images = images.to(device, non_blocking=True)
                batch, views, channels, height, width = images.shape
                en, de = model(images.reshape(batch * views, channels, height, width))
                anomaly_map, _ = cal_anomaly_maps(en, de, width)
                anomaly_map = F.interpolate(anomaly_map, size=(args.map_size, args.map_size),
                                            mode="bilinear", align_corners=False)
                anomaly_map = gaussian(anomaly_map).view(batch, views, 1, args.map_size, args.map_size)
                flat = anomaly_map.flatten(1)
                top_count = max(1, int(flat.shape[1] * args.score_top_percent / 100.0))
                object_scores = flat.topk(top_count, dim=1, sorted=False).values.mean(dim=1)
                for local_index in range(batch):
                    object_index = cursor + local_index
                    is_defect = bool(labels[local_index].max().item())
                    if is_defect:
                        candidates.append({
                            "category": category,
                            "object_index": object_index,
                            "object_score": float(object_scores[local_index].item()),
                            "paths": [str(path) for path in test_data.img_paths[object_index]],
                        })
                cursor += batch

    # Keep the sample balanced across categories: the raw anomaly-score scale
    # can shift by category, so cross-category raw-score ordering is misleading.
    by_category = {
        category: sorted((item for item in candidates if item["category"] == category),
                         key=lambda item: item["object_score"])
        for category in categories
    }
    selected = [item for category in categories
                for item in by_category[category][:args.count_per_category]]
    if not selected:
        raise RuntimeError("No defective test objects found in the selected categories.")
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    # Second pass: generate one triptych per selected object, using its lowest
    # image-level anomaly score among the five views as the displayed example.
    for rank, item in enumerate(selected, start=1):
        data = test_sets[item["category"]]
        images, masks, labels, paths = data[item["object_index"]]
        batch = images.to(device).unsqueeze(0)
        with torch.inference_mode():
            en, de = model(batch.flatten(0, 1))
            anomaly_map, _ = cal_anomaly_maps(en, de, args.crop_size)
            anomaly_map = F.interpolate(anomaly_map, size=(args.map_size, args.map_size),
                                        mode="bilinear", align_corners=False)
            anomaly_map = gaussian(anomaly_map)
            flattened = anomaly_map.flatten(1)
            per_view_count = max(1, int(flattened.shape[1] * args.score_top_percent / 100.0))
            view_scores = flattened.topk(per_view_count, dim=1, sorted=False).values.mean(dim=1)
            selected_view = int(view_scores.argmin().item())
            heat = F.interpolate(anomaly_map[selected_view:selected_view + 1],
                                 size=(args.crop_size, args.crop_size), mode="bilinear",
                                 align_corners=False)[0, 0].cpu().numpy()
            displayed_score = float(view_scores[selected_view].item())

        image_path = Path(paths[selected_view])
        with Image.open(image_path) as image:
            original = display_transform(image.convert("RGB"))
        mask = masks[selected_view, 0].cpu().numpy()
        norm_heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-12)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        axes[0].imshow(original)
        axes[0].set_title("Test image")
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground-truth mask")
        axes[2].imshow(original)
        color = axes[2].imshow(norm_heat, cmap="jet", alpha=0.55, vmin=0, vmax=1)
        axes[2].set_title(f"Dinomaly2 heatmap\nview score {displayed_score:.4f}")
        fig.colorbar(color, ax=axes[2], fraction=0.046, pad=0.04)
        for axis in axes:
            axis.axis("off")
        fig.suptitle(f"Rank {rank}: {item['category']} | test object {item['object_index']}")
        filename = f"{rank:02d}_{item['category']}_object_{item['object_index']}_view_{selected_view}.png"
        fig.savefig(output / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)
        rows.append({
            "rank": rank,
            "category": item["category"],
            "object_index": item["object_index"],
            "object_score": item["object_score"],
            "displayed_view_index": selected_view,
            "displayed_view_score": displayed_score,
            "image_path": str(image_path),
            "figure": filename,
        })
        print(f"{rank:02d}/{len(selected)} {item['category']} object={item['object_index']} "
              f"object_score={item['object_score']:.6f} view={selected_view} "
              f"view_score={displayed_score:.6f} -> {output / filename}")

    report = {
        "checkpoint": str(checkpoint),
        "dataset": str(root),
        "categories": categories,
        "selection": (f"lowest {args.count_per_category} scoring defective objects within each category; "
                      f"top {args.score_top_percent:g}% of pooled smoothed anomaly map"),
        "settings": {"image_size": args.image_size, "crop_size": args.crop_size,
                     "map_size": args.map_size, "score_top_percent": args.score_top_percent},
        "objects": rows,
    }
    (output / "selected_objects.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Selection report: {output / 'selected_objects.json'}")


if __name__ == "__main__":
    main()
