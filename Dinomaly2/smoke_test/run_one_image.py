#!/usr/bin/env python3
"""Run one real Dinomaly2 training step on a single normal image."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


REPO_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = REPO_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from dataset import RealIADDatasetv2, get_data_transforms  # noqa: E402
from dinov1.utils import trunc_normal_  # noqa: E402
from models import vit_encoder  # noqa: E402
from models.uad import Dinomaly  # noqa: E402
from models.vision_transformer import Attention, Block as VitBlock, LinearAttention2  # noqa: E402
from optimizers import StableAdamW  # noqa: E402
from utils import global_cosine_hm_percent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=WORKSPACE_DIR / "dataset" / "Real-IAD",
        help="Real-IAD root containing realiad_1024/ and realiad_jsons/.",
    )
    parser.add_argument("--category", default="audiojack")
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional single image to use instead of the first normal train image.",
    )
    parser.add_argument("--backbone", default="dinov2reg_vit_base_14")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=392)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--linear-attention", type=int, choices=(0, 1), default=1)
    parser.add_argument("--context-aware-recenter", type=int, choices=(0, 1), default=1)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index; CPU is used when CUDA is unavailable.")
    return parser.parse_args()


def get_one_image(args: argparse.Namespace, transform) -> tuple[torch.Tensor, str]:
    if args.image is not None:
        from PIL import Image

        image_path = args.image.expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0), str(image_path)

    dataset = RealIADDatasetv2(
        root=str(args.data_path),
        category=args.category,
        transform=transform,
        gt_transform=None,
        phase="train",
        split="realiad_jsons/realiad_jsons",
        version="realiad_1024",
    )
    normal_indices = np.flatnonzero(dataset.labels == 0)
    if normal_indices.size == 0:
        raise RuntimeError(f"No normal training images found for category {args.category!r}.")

    index = int(normal_indices[0])
    image, _ = dataset[index]
    return image.unsqueeze(0), str(dataset.img_paths[index])


def build_model(args: argparse.Namespace, device: torch.device) -> tuple[Dinomaly, list[nn.Module]]:
    backbone_parts = args.backbone.split("_")
    arch, patch_size = backbone_parts[-2], int(backbone_parts[-1])
    if arch not in {"small", "base", "large"}:
        raise ValueError(f"Unsupported backbone architecture in {args.backbone!r}.")

    encoder = vit_encoder.load(
        args.backbone,
        WEIGHTS_DIR=str(REPO_DIR / "backbones" / "weights"),
    )
    dimensions = {"small": (384, 6), "base": (768, 12), "large": (1024, 16)}
    embed_dim, num_heads = dimensions[arch]
    target_layers = [4, 6, 8, 10, 12, 14, 16, 18] if arch == "large" else list(range(2, 10))

    bottleneck = nn.ModuleList([
        nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=args.dropout)),
        nn.Sequential(
            nn.Linear(256, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(p=args.dropout),
        ),
    ])
    attention = partial(LinearAttention2, eps=1e-8) if args.linear_attention else Attention
    decoder = nn.ModuleList([
        VitBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=attention,
        )
        for _ in range(8)
    ])
    model = Dinomaly(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        remove_class_token=False,
        fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        context_aware_recenter=bool(args.context_aware_recenter),
    ).to(device)

    model.init_weights()
    return model, [bottleneck, decoder]


def main() -> None:
    args = parse_args()
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)

    if args.image_size % 14 or args.crop_size % 14:
        raise ValueError("image-size and crop-size must both be divisible by 14 for a ViT/14 backbone.")

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    transform, _ = get_data_transforms(args.image_size, args.crop_size)
    image, image_path = get_one_image(args, transform)
    image = image.to(device)

    model, trainable_modules = build_model(args, device)
    model.train()
    optimizer = StableAdamW(
        [
            {"params": model.bottleneck.parameters()},
            {"params": model.decoder.parameters()},
        ],
        lr=2e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=False,
        eps=1e-10,
    )

    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    encoder_features, decoder_features = model(image)
    loss = global_cosine_hm_percent(encoder_features, decoder_features, p=0.0, factor=0.1)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Training loss is not finite: {loss.item()}")
    loss.backward()

    trainable_parameters = [p for module in trainable_modules for p in module.parameters()]
    gradients_with_values = [p.grad for p in trainable_parameters if p.grad is not None]
    if not gradients_with_values:
        raise RuntimeError("Backward pass produced no gradients for the trainable layers.")
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=0.1)
    optimizer.step()
    elapsed = time.perf_counter() - start

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "one_step_checkpoint.pth"
    torch.save(
        {
            "backbone": args.backbone,
            "target_layers": model.target_layers,
            "bottleneck": model.bottleneck.state_dict(),
            "decoder": model.decoder.state_dict(),
            "training_step": 1,
        },
        checkpoint_path,
    )
    summary = {
        "status": "passed",
        "image": image_path,
        "image_shape": list(image.shape),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "backbone": args.backbone,
        "training_step": 1,
        "optimizer_step_completed": True,
        "loss": float(loss.detach().item()),
        "gradient_norm_before_clipping": float(grad_norm.detach().item()),
        "elapsed_seconds_forward_backward_update": elapsed,
        "checkpoint": str(checkpoint_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
