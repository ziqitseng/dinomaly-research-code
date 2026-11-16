#!/usr/bin/env python3
"""Visualize a Real-IAD test view, its nearest normal training image, and anomaly map."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset import RealIADDatasetv2, get_data_transforms
from dinomaly2_inference import DEFAULT_CHECKPOINT, DEFAULT_DATASET, make_model
from utils import cal_anomaly_maps, get_gaussian_kernel


class ImagePaths(Dataset):
    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a Real-IAD test image, closest normal train image, mask, and Dinomaly2 anomaly map."
    )
    parser.add_argument("--category", required=True, help="Real-IAD category, e.g. usb or bottle_cap.")
    parser.add_argument("--object-index", type=int,
                        help="Index into the category's grouped test objects (use --list to see indices).")
    parser.add_argument("--view-index", type=int, default=0, help="View within the selected object, usually 0–4.")
    parser.add_argument("--list", action="store_true", help="List test object indices and their view paths, then exit.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=392)
    parser.add_argument("--map-size", type=int, default=256)
    parser.add_argument("--score-top-percent", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for scanning the category's normal training images.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--save-path", type=Path,
                        help="Optional path to save the figure. The plot is also shown unless --no-display is set.")
    parser.add_argument("--no-display", action="store_true", help="Save only; do not open an interactive window.")
    return parser.parse_args()


def descriptor(encoder_features: list[torch.Tensor]) -> torch.Tensor:
    """Pool fused encoder maps to one cosine-comparable feature vector per image."""
    pooled = [feature.mean(dim=(-2, -1)) for feature in encoder_features]
    return F.normalize(torch.cat(pooled, dim=1), dim=1)


def get_normal_training_match(model, train_paths, transform, query_descriptor, device, batch_size, workers):
    loader = DataLoader(
        ImagePaths(train_paths, transform), batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=device.type == "cuda",
    )
    descriptors = []
    with torch.inference_mode():
        for tensors, _ in loader:
            encoder_features, _ = model(tensors.to(device, non_blocking=True))
            descriptors.append(descriptor(encoder_features).cpu())
    train_descriptors = torch.cat(descriptors, dim=0)
    similarities = train_descriptors @ query_descriptor.cpu().T
    nearest_index = int(similarities.argmax().item())
    return train_paths[nearest_index], float(similarities[nearest_index].item())


def main() -> None:
    args = parse_args()
    if args.object_index is None and not args.list:
        raise ValueError("Provide --object-index, or use --list to inspect a category's test objects.")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers cannot be negative.")
    if min(args.image_size, args.crop_size, args.map_size) <= 0 or args.crop_size % 14:
        raise ValueError("Image/map sizes must be positive and crop-size must be divisible by 14.")
    if not 0 < args.score_top_percent <= 100:
        raise ValueError("score-top-percent must be in the range (0, 100].")

    data_root = args.data_path.expanduser().resolve()
    data_transform, gt_transform = get_data_transforms(args.image_size, args.crop_size)
    split = "realiad_jsons/realiad_jsons"
    train_data = RealIADDatasetv2(
        root=str(data_root), category=args.category, transform=data_transform,
        gt_transform=gt_transform, phase="train", split=split, version="realiad_1024",
    )
    test_data = RealIADDatasetv2(
        root=str(data_root), category=args.category, transform=data_transform,
        gt_transform=gt_transform, phase="test", split=split, version="realiad_1024",
    )

    if args.list:
        for index, view_paths in enumerate(test_data.img_paths):
            print(f"{index}: {len(view_paths)} views | " + " | ".join(map(str, view_paths)))
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to this process.")
    device = torch.device(
        "cuda:0" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    train_paths = [
        str(path) for path, label in zip(train_data.img_paths, train_data.labels)
        if not bool(label)
    ]
    if not train_paths:
        raise RuntimeError(f"No normal training images found for category {args.category!r}.")
    if not 0 <= args.object_index < len(test_data):
        raise IndexError(f"object-index must be in [0, {len(test_data) - 1}]. Use --list to inspect objects.")

    image_paths = list(test_data.img_paths[args.object_index])
    mask_paths = list(test_data.gt_paths[args.object_index])
    labels = list(test_data.labels[args.object_index])
    if not 0 <= args.view_index < len(image_paths):
        raise IndexError(f"view-index must be in [0, {len(image_paths) - 1}] for this object.")
    query_path = str(image_paths[args.view_index])
    mask_path = mask_paths[args.view_index]
    is_defective = bool(labels[args.view_index])

    model = make_model(device)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with Image.open(query_path) as image:
        query_visual = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.CenterCrop(args.crop_size),
        ])(image.convert("RGB"))
        query_tensor = data_transform(image.convert("RGB")).unsqueeze(0).to(device)
    if is_defective and mask_path is not None:
        with Image.open(str(mask_path)) as mask_image:
            mask = gt_transform(mask_image.convert("L")).squeeze(0).numpy()
    else:
        mask = np.zeros((args.crop_size, args.crop_size), dtype=np.float32)

    with torch.inference_mode():
        encoder_features, decoder_features = model(query_tensor)
        query_descriptor = descriptor(encoder_features)
        anomaly_map, _ = cal_anomaly_maps(encoder_features, decoder_features, args.crop_size)
        if args.map_size != args.crop_size:
            anomaly_map = F.interpolate(
                anomaly_map, size=(args.map_size, args.map_size), mode="bilinear", align_corners=False
            )
        anomaly_map = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)(anomaly_map)
        score_flat = anomaly_map[0, 0].flatten()
        top_count = max(1, int(score_flat.numel() * args.score_top_percent / 100.0))
        image_score = float(torch.topk(score_flat, k=top_count, sorted=False).values.mean().item())
        heatmap = F.interpolate(
            anomaly_map, size=(args.crop_size, args.crop_size), mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()

    nearest_path, similarity = get_normal_training_match(
        model, train_paths, data_transform, query_descriptor, device, args.batch_size, args.workers
    )
    with Image.open(nearest_path) as image:
        nearest_visual = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.CenterCrop(args.crop_size),
        ])(image.convert("RGB"))

    heat_low, heat_high = float(heatmap.min()), float(heatmap.max())
    normalized_heat = (heatmap - heat_low) / (heat_high - heat_low + 1e-12)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), constrained_layout=True)
    axes[0].imshow(query_visual)
    axes[0].set_title(f"Test view ({'defect' if is_defective else 'normal'})")
    axes[1].imshow(nearest_visual)
    axes[1].set_title(f"Closest normal train image\ncosine similarity {similarity:.3f}")
    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground-truth mask")
    axes[3].imshow(query_visual)
    color_overlay = axes[3].imshow(normalized_heat, cmap="jet", alpha=0.55, vmin=0, vmax=1)
    axes[3].set_title(f"Normalized anomaly score\ntop {args.score_top_percent:g}% score={image_score:.4f}")
    fig.colorbar(color_overlay, ax=axes[3], fraction=0.046, pad=0.04)
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"{args.category} | object {args.object_index}, view {args.view_index}\n{query_path}")

    if args.save_path:
        save_path = args.save_path.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved visualization: {save_path}")
    print(f"Query: {query_path}")
    print(f"Nearest normal training image: {nearest_path}")
    print(f"Anomaly score (top {args.score_top_percent:g}%): {image_score:.6f}")
    if args.no_display:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
