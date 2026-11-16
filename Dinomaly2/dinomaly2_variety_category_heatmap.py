#!/usr/bin/env python3
"""Create an ArcAD-style 160-category heatmap for the Dinomaly2 Variety run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader

from dataset import RealIADDatasetv2, get_data_transforms
from dinomaly2_inference import make_model
from utils import cal_anomaly_maps, get_gaussian_kernel


DEFAULT_DATA = Path("/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024")
DEFAULT_CHECKPOINT = Path("out/real_iad_variety_1024 out/full run/variety_160cat/model.pth")
DEFAULT_OUTPUT = Path("out/real_iad_variety_1024 out/full run/variety_160cat/category_heatmap")
SPLIT = "realiadvariety_jsons/Real-IAD_Variety_jsons"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=280)
    parser.add_argument("--crop-size", type=int, default=280)
    parser.add_argument("--map-size", type=int, default=256)
    parser.add_argument("--score-top-percent", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def max_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        raise ValueError("Cannot select an F1 threshold without score variation.")
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-7)
    index = int(np.nanargmax(f1_values))
    return float(thresholds[index]), float(f1_values[index])


def calculate_rates(labels: np.ndarray, scores: np.ndarray) -> dict:
    threshold, f1_max = max_f1_threshold(labels, scores)
    predicted = scores >= threshold
    normal = labels == 0
    defect = labels == 1
    tn = int(np.count_nonzero(normal & ~predicted))
    fp = int(np.count_nonzero(normal & predicted))
    fn = int(np.count_nonzero(defect & ~predicted))
    tp = int(np.count_nonzero(defect & predicted))
    return {
        "I-AUROC": float(roc_auc_score(labels, scores)),
        "normal_specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "defect_recall": tp / (tp + fn) if tp + fn else float("nan"),
        "F1_max": f1_max,
        "test_images": int(len(labels)),
        "normal_images": tn + fp,
        "defect_images": fn + tp,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "f1_max_threshold": threshold,
    }


def write_heatmap(rows: list[dict], output: Path) -> None:
    columns = [
        ("I-AUROC", "I-AUROC\n(threshold-free)", 0.5, 1.0, "Blues"),
        ("normal_specificity", "Normal specificity\n(TN / actual normal)", 0.0, 1.0, "YlGnBu"),
        ("defect_recall", "Defect recall\n(TP / actual defect)", 0.0, 1.0, "YlOrRd"),
        ("F1_max", "Image F1\n(category max-F1 cutoff)", 0.0, 1.0, "Purples"),
    ]
    fig, axes = plt.subplots(1, len(columns), figsize=(22, 42), sharey=True,
                             gridspec_kw={"width_ratios": [1.0] * 4, "wspace": 0.12})
    for ax, (field, title, low, high, cmap) in zip(axes, columns):
        values = np.asarray([row[field] for row in rows], dtype=float).reshape(-1, 1)
        image = ax.imshow(values, cmap=cmap, norm=Normalize(vmin=low, vmax=high),
                          aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=16, pad=16, fontweight="bold")
        ax.set_xticks([])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        if ax is axes[0]:
            ax.set_yticks(np.arange(len(rows)))
            ax.set_yticklabels([row["category"] for row in rows], fontsize=7.2)
            ax.tick_params(axis="y", length=0, pad=8)
            ax.set_ylabel("Categories sorted from lowest to highest I-AUROC", fontsize=13, labelpad=16)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        for row_index, value in enumerate(values[:, 0]):
            label = "n/a" if np.isnan(value) else f"{100 * value:.1f}%"
            text_color = "white" if value >= (low + high) / 2 else "#142d4e"
            ax.text(0, row_index, label, ha="center", va="center", fontsize=6.5,
                    color=text_color, fontweight="medium")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.055, pad=0.025)
        colorbar.ax.tick_params(labelsize=9)
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        colorbar.set_label("Percent", fontsize=10, labelpad=9)

    fig.suptitle(
        "Dinomaly2 — Real-IAD Variety 1024\n160-category image-level performance and confusion-derived rates\n"
        "Rows sorted by ascending I-AUROC (weakest image-level category first)",
        fontsize=22, fontweight="bold", y=0.998,
    )
    fig.text(
        0.5, 0.006,
        "Specificity, recall, and F1 use category-wise thresholds selected to maximize F1 on each category's test set. "
        "These are descriptive test-set operating points, not deployment thresholds.",
        ha="center", fontsize=11,
    )
    fig.savefig(output / "dinomaly2_variety_1024_category_heatmap.svg", format="svg", bbox_inches="tight")
    fig.savefig(output / "dinomaly2_variety_1024_category_heatmap.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if min(args.image_size, args.crop_size, args.map_size, args.batch_size) <= 0:
        raise ValueError("Image/map sizes and batch-size must be positive.")
    if args.crop_size % 14:
        raise ValueError("crop-size must be divisible by 14 for ViT-B/14.")
    if not 0 < args.score_top_percent <= 100 or args.workers < 0:
        raise ValueError("score-top-percent must be in (0, 100] and workers cannot be negative.")

    root = args.data_path.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    json_dir = root / SPLIT
    if not json_dir.is_dir():
        raise FileNotFoundError(f"Variety split JSON directory not found: {json_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    categories = sorted(path.stem for path in json_dir.glob("*.json"))
    if len(categories) != 160:
        raise ValueError(f"Expected 160 Variety manifests, found {len(categories)} in {json_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type != "cuda":
        raise RuntimeError("This full-run evaluation is configured for the trained CUDA model.")

    transform, gt_transform = get_data_transforms(args.image_size, args.crop_size)
    model = make_model(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    rows = []

    for index, category in enumerate(categories, start=1):
        dataset = RealIADDatasetv2(
            root=str(root), category=category, transform=transform, gt_transform=gt_transform,
            phase="test", split=SPLIT, version=".",
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        label_batches, score_batches = [], []
        with torch.inference_mode():
            for images, _masks, labels, _paths in loader:
                images = images.to(device, non_blocking=True)
                batch, views, channels, height, width = images.shape
                en, de = model(images.reshape(batch * views, channels, height, width))
                maps, _ = cal_anomaly_maps(en, de, width)
                maps = F.interpolate(maps, size=(args.map_size, args.map_size),
                                     mode="bilinear", align_corners=False)
                maps = gaussian(maps).view(batch, views, 1, args.map_size, args.map_size)
                flattened = maps.flatten(2)
                top_count = max(1, int(flattened.shape[-1] * args.score_top_percent / 100.0))
                scores = flattened.topk(top_count, dim=2, sorted=False).values.mean(dim=2)
                label_batches.append(labels.cpu().numpy().reshape(-1).astype(np.uint8))
                score_batches.append(scores.cpu().numpy().reshape(-1).astype(np.float64))
        labels = np.concatenate(label_batches)
        scores = np.concatenate(score_batches)
        rows.append({"category": category, **calculate_rates(labels, scores)})
        if index % 10 == 0 or index == len(categories):
            print(f"[{index}/160] {category}: {len(labels)} test views", flush=True)

    rows.sort(key=lambda row: (row["I-AUROC"], row["category"]))
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with (output / "category_confusion_rates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "task": "Dinomaly2 Real-IAD Variety 1024 category-level image metrics and max-F1 confusion rates",
        "checkpoint": str(checkpoint), "dataset_root": str(root), "split": str(json_dir),
        "category_count": len(rows), "test_views": int(sum(row["test_images"] for row in rows)),
        "score": "mean of the top score_top_percent pixels of each 256x256 smoothed anomaly map",
        "preprocessing": {"image_size": args.image_size, "crop_size": args.crop_size,
                          "map_size": args.map_size, "score_top_percent": args.score_top_percent,
                          "gaussian": {"kernel_size": 5, "sigma": 4}},
        "threshold_selection": "For each category independently, select its labeled test-score threshold that maximizes image F1.",
        "categories_sorted_by_ascending_I-AUROC": rows,
    }
    (output / "category_heatmap_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_heatmap(rows, output)
    print("Wrote", output / "dinomaly2_variety_1024_category_heatmap.svg")
    print("Wrote", output / "dinomaly2_variety_1024_category_heatmap.png")
    print("Wrote", output / "category_confusion_rates.csv")


if __name__ == "__main__":
    main()
