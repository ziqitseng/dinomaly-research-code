#!/usr/bin/env python3
"""Build full-test image- and object-level confusion matrices for Variety 1024."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
DEFAULT_OUTPUT = Path("out/real_iad_variety_1024 out/full run/variety_160cat/confusion_matrix")
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
    parser.add_argument("--report-json", type=Path,
                        help="Render an existing confusion_matrix.json without rerunning inference.")
    parser.add_argument("--image-only", action="store_true",
                        help="Render a single image-level matrix with only N shown in its title (requires --report-json).")
    return parser.parse_args()


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best = int(np.nanargmax(f1_values))
    return float(thresholds[best]), float(f1_values[best])


def confusion_summary(labels: np.ndarray, scores: np.ndarray) -> dict:
    threshold, f1 = best_f1_threshold(labels, scores)
    predicted = scores >= threshold
    tp = int(np.count_nonzero(predicted & (labels == 1)))
    fp = int(np.count_nonzero(predicted & (labels == 0)))
    tn = int(np.count_nonzero(~predicted & (labels == 0)))
    fn = int(np.count_nonzero(~predicted & (labels == 1)))
    return {
        "threshold": threshold,
        "f1_at_threshold": f1,
        "auroc": float(roc_auc_score(labels, scores)),
        "matrix": [[tn, fp], [fn, tp]],
        "actual_normal": tn + fp,
        "actual_defect": fn + tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def draw_matrix(ax, summary: dict, title: str, n_samples: int, show_metrics: bool = True) -> None:
    matrix = np.asarray(summary["matrix"], dtype=np.int64)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
    ax.set_xticks([0, 1], labels=["Normal", "Defect"])
    ax.set_yticks([0, 1], labels=["Normal", "Defect"])
    ax.set_xlabel("Predicted class", fontsize=14, labelpad=12)
    ax.set_ylabel("Actual class", fontsize=14, labelpad=12)
    ax.tick_params(axis="both", labelsize=13, length=0)
    title_text = (f"{title}\nN={n_samples:,}" if not show_metrics else
                  f"{title}\nN={n_samples:,} | threshold={summary['threshold']:.6g} | "
                  f"F1={summary['f1_at_threshold']:.4f} | AUROC={summary['auroc']:.4f}")
    ax.set_title(title_text, fontsize=15, pad=16)
    midpoint = matrix.max() / 2
    for row in range(2):
        row_total = int(matrix[row].sum())
        for col in range(2):
            value = int(matrix[row, col])
            percent = 100.0 * value / row_total if row_total else 0.0
            ax.text(col, row, f"{value:,}\n{percent:.1f}%", ha="center", va="center",
                    fontsize=19, fontweight="bold",
                    color="white" if value > midpoint else "#152238")
    for edge in (-0.5, 0.5, 1.5):
        ax.axhline(edge, color="white", linewidth=2)
        ax.axvline(edge, color="white", linewidth=2)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Sample count", rotation=270, labelpad=18, fontsize=12)
    colorbar.ax.tick_params(labelsize=10)


def main() -> None:
    args = parse_args()
    if args.report_json:
        report_path = args.report_json.expanduser().resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        summaries = report["matrices"]
        if args.image_only:
            figure, ax = plt.subplots(1, 1, figsize=(11, 9), constrained_layout=True)
            draw_matrix(ax, summaries["image_level_per_view"], "Dinomaly2 · Real-IAD Variety 1024\nImage-level confusion matrix",
                        report["test_views"], show_metrics=False)
            figure.text(0.5, -0.005,
                        "Cell percentages are row-normalized (share of actual class); each row sums to 100%.",
                        ha="center", va="top", fontsize=12)
            figure.savefig(output / "image_level_confusion_matrix.svg", format="svg", bbox_inches="tight")
            figure.savefig(output / "image_level_confusion_matrix.png", format="png", dpi=300, bbox_inches="tight")
            plt.close(figure)
            print(f"Image-only SVG: {output / 'image_level_confusion_matrix.svg'}")
            print(f"Image-only PNG: {output / 'image_level_confusion_matrix.png'}")
            return
        figure, axes = plt.subplots(1, 2, figsize=(18, 9), constrained_layout=True)
        draw_matrix(axes[0], summaries["image_level_per_view"], "Image-level (each camera view)",
                    report["test_views"])
        draw_matrix(axes[1], summaries["object_level_five_view"], "Object-level (five-view pooled score)",
                    report["test_objects"])
        figure.suptitle("Dinomaly2 · Real-IAD Variety 1024 · all 160 categories",
                        fontsize=20, fontweight="bold")
        figure.text(0.5, -0.005,
                    "Cell percentages are row-normalized (share of actual class); each row sums to 100%. "
                    "Thresholds maximize F1 on this labeled test set.",
                    ha="center", va="top", fontsize=12)
        figure.savefig(output / "confusion_matrix.svg", format="svg", bbox_inches="tight")
        figure.savefig(output / "confusion_matrix.png", format="png", dpi=300, bbox_inches="tight")
        plt.close(figure)
        print(f"Updated SVG: {output / 'confusion_matrix.svg'}")
        print(f"Updated PNG: {output / 'confusion_matrix.png'}")
        return

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
        raise ValueError(f"Expected 160 Variety category manifests, found {len(categories)} in {json_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type != "cuda":
        raise RuntimeError("This full-run Dinomaly2 evaluation path requires CUDA.")

    data_transform, gt_transform = get_data_transforms(args.image_size, args.crop_size)
    model = make_model(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    image_labels, image_scores = [], []
    object_labels, object_scores = [], []
    category_objects = 0
    for category_number, category in enumerate(categories, start=1):
        dataset = RealIADDatasetv2(
            root=str(root), category=category, transform=data_transform,
            gt_transform=gt_transform, phase="test", split=SPLIT, version=".",
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        with torch.inference_mode():
            for images, _masks, labels, _paths in loader:
                images = images.to(device, non_blocking=True)
                batch, views, channels, height, width = images.shape
                en, de = model(images.reshape(batch * views, channels, height, width))
                maps, _ = cal_anomaly_maps(en, de, width)
                maps = F.interpolate(maps, size=(args.map_size, args.map_size),
                                      mode="bilinear", align_corners=False)
                maps = gaussian(maps).view(batch, views, 1, args.map_size, args.map_size)

                view_flat = maps.flatten(2)
                view_top_count = max(1, int(view_flat.shape[-1] * args.score_top_percent / 100.0))
                scores_per_view = view_flat.topk(view_top_count, dim=2, sorted=False).values.mean(dim=2)
                image_scores.extend(scores_per_view.cpu().numpy().reshape(-1).tolist())
                image_labels.extend(labels.cpu().numpy().reshape(-1).astype(np.uint8).tolist())

                object_flat = maps.flatten(1)
                object_top_count = max(1, int(object_flat.shape[-1] * args.score_top_percent / 100.0))
                scores_per_object = object_flat.topk(object_top_count, dim=1, sorted=False).values.mean(dim=1)
                object_scores.extend(scores_per_object.cpu().numpy().tolist())
                object_labels.extend(labels.max(dim=1).values.cpu().numpy().astype(np.uint8).tolist())
        category_objects += len(dataset)
        print(f"[{category_number}/160] {category}: {len(dataset)} test objects")

    image_labels = np.asarray(image_labels, dtype=np.uint8)
    image_scores = np.asarray(image_scores, dtype=np.float64)
    object_labels = np.asarray(object_labels, dtype=np.uint8)
    object_scores = np.asarray(object_scores, dtype=np.float64)
    summaries = {
        "image_level_per_view": confusion_summary(image_labels, image_scores),
        "object_level_five_view": confusion_summary(object_labels, object_scores),
    }

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(18, 9), constrained_layout=True)
    draw_matrix(axes[0], summaries["image_level_per_view"], "Image-level (each camera view)", len(image_labels))
    draw_matrix(axes[1], summaries["object_level_five_view"], "Object-level (five-view pooled score)", len(object_labels))
    figure.suptitle("Dinomaly2 · Real-IAD Variety 1024 · all 160 categories", fontsize=20, fontweight="bold")
    figure.text(
        0.5, -0.005,
        "Cell percentages are row-normalized (share of actual class); each row sums to 100%. "
        "Thresholds maximize F1 on this labeled test set.",
        ha="center", va="top", fontsize=12,
    )
    figure.savefig(output / "confusion_matrix.svg", format="svg", bbox_inches="tight")
    figure.savefig(output / "confusion_matrix.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    report = {
        "checkpoint": str(checkpoint),
        "dataset": str(root),
        "dataset_type": "Real-IAD Variety",
        "category_count": len(categories),
        "categories": categories,
        "test_objects": category_objects,
        "test_views": len(image_labels),
        "score": "mean of top score_top_percent pixels after resize and 5x5 sigma-4 Gaussian smoothing",
        "preprocessing": {"image_size": args.image_size, "crop_size": args.crop_size,
                          "map_size": args.map_size, "score_top_percent": args.score_top_percent},
        "threshold_selection": "Separate F1-maximizing threshold for each matrix, selected using all labeled test samples.",
        "matrices": summaries,
    }
    (output / "confusion_matrix.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Image-level confusion matrix: {summaries['image_level_per_view']['matrix']}")
    print(f"Object-level confusion matrix: {summaries['object_level_five_view']['matrix']}")
    print(f"SVG: {output / 'confusion_matrix.svg'}")
    print(f"PNG: {output / 'confusion_matrix.png'}")
    print(f"Details: {output / 'confusion_matrix.json'}")


if __name__ == "__main__":
    main()
