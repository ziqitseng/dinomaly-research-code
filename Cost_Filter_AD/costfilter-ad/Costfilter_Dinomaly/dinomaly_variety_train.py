"""Train original ViTill Dinomaly-B on the original Real-IAD Variety split.

The architecture and loss follow dinomaly_realiad_uni.py. This runner adds
Variety manifest loading, step loss logging, and resumable checkpoints.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn.init import trunc_normal_
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sys.dont_write_bytecode = True
from dataset import get_data_transforms, realiad_variety_category_root, realiad_variety_manifest
from dinov2.models.vision_transformer import vit_base
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from optimizers import StableAdamW
from utils import WarmCosineScheduler, global_cosine_hm_percent


HERE = Path(__file__).resolve().parent
DEFAULT_BACKBONE = HERE / 'backbones/weights/dinov2_vitb14_reg4_pretrain.pth'
LOSS_COLUMNS = ('step', 'epoch', 'batch', 'loss', 'lr', 'unix_time')


class VarietyNormalDataset(Dataset):
    def __init__(self, root, category, transform):
        manifest = Path(realiad_variety_manifest(str(root), category))
        rows = json.loads(manifest.read_text())['train']
        if not rows or any(row['anomaly_class'] != 'OK' for row in rows):
            raise ValueError('Expected a nonempty normal-only training split: %s' % manifest)
        category_root = Path(realiad_variety_category_root(str(root), category))
        self.paths = [category_root / row['image_path'] for row in rows]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert('RGB'))


def categories_and_metadata(args):
    manifest_dir = args.data_path / 'realiadvariety_jsons/Real-IAD_Variety_jsons'
    files = sorted(manifest_dir.glob('*.json'))
    if len(files) != 160:
        raise ValueError('Expected 160 original Variety manifests at %s; found %d' %
                         (manifest_dir, len(files)))
    categories = [path.stem for path in files]
    metadata = {
        'protocol': 'Real-IAD Variety original 160-category normal train split',
        'categories': categories,
        'manifest_sha256': {path.stem: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in files},
        'backbone_checkpoint': str(args.backbone_checkpoint.resolve()),
        'backbone_sha256': hashlib.sha256(args.backbone_checkpoint.read_bytes()).hexdigest(),
        'architecture': 'original ViTill Dinomaly-B DINOv2-R ViT-B/14, dropout 0.4',
        'geometry': 'resize 448, center crop 392',
        'total_steps': args.total_steps,
        'batch_size': args.batch_size,
        'seed': args.seed,
    }
    return categories, metadata


def build_model(backbone_checkpoint, device):
    encoder = vit_base(patch_size=14, img_size=518, block_chunks=0,
                       init_values=1e-8, num_register_tokens=4,
                       interpolate_antialias=False, interpolate_offset=0.1)
    encoder.load_state_dict(torch.load(backbone_checkpoint, map_location='cpu'), strict=True)
    bottleneck = nn.ModuleList([bMlp(768, 3072, 768, drop=0.4)])
    decoder = nn.ModuleList([
        VitBlock(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.,
                 attn=LinearAttention2)
        for _ in range(8)])
    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
                   target_layers=[2, 3, 4, 5, 6, 7, 8, 9], mask_neighbor_size=0,
                   fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
                   fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]])
    trainable = nn.ModuleList([bottleneck, decoder])
    for module in trainable.modules():
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    model.encoder.requires_grad_(False)
    return model.to(device), trainable


def save_checkpoint(args, model, optimizer, scheduler, metadata, step, epoch, batch):
    checkpoint = args.output_dir / ('trainer_step_%d.pth' % step)
    if checkpoint.exists():
        raise FileExistsError('Refusing to overwrite: %s' % checkpoint)
    temporary = checkpoint.with_suffix('.pth.tmp')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metadata': metadata, 'step': step, 'epoch': epoch, 'batch': batch,
        'rng_python': random.getstate(),
        'rng_numpy': np.random.get_state(),
        'rng_torch': torch.get_rng_state(),
        'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }, temporary)
    temporary.replace(checkpoint)
    print('Saved training checkpoint:', checkpoint, flush=True)


def train(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    categories, metadata = categories_and_metadata(args)
    (args.output_dir / 'protocol.json').write_text(json.dumps(metadata, indent=2) + '\n')
    transform, _ = get_data_transforms(448, 392)
    dataset = ConcatDataset([VarietyNormalDataset(args.data_path, name, transform)
                             for name in categories])
    batches_per_epoch = len(dataset) // args.batch_size
    if batches_per_epoch == 0:
        raise ValueError('Batch size exceeds training split')
    print('Normal training views:', len(dataset), 'categories:', len(categories),
          'batches per epoch:', batches_per_epoch, flush=True)
    model, trainable = build_model(args.backbone_checkpoint, args.device)
    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4,
                            amsgrad=True, eps=1e-10)
    scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4,
                                    total_iters=args.total_steps, warmup_iters=100)
    step = start_epoch = start_batch = 0
    loss_path = args.output_dir / 'step_losses.csv'
    if args.resume_checkpoint:
        saved = torch.load(args.resume_checkpoint, map_location='cpu')
        if saved['metadata'] != metadata:
            raise ValueError('Resume checkpoint protocol or recipe differs')
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        step = saved['step']
        start_epoch = saved['epoch']
        start_batch = saved['batch'] + 1
        if start_batch == batches_per_epoch:
            start_epoch += 1
            start_batch = 0
        if not loss_path.exists():
            raise FileNotFoundError('Missing loss history for resume: %s' % loss_path)
        with loss_path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) < step:
            raise ValueError('Loss history ends before checkpoint step %d' % step)
        with loss_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=LOSS_COLUMNS)
            writer.writeheader()
            writer.writerows(row for row in rows if int(row['step']) <= step)
        random.setstate(saved['rng_python'])
        np.random.set_state(saved['rng_numpy'])
        torch.set_rng_state(saved['rng_torch'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved['rng_cuda'])
        print('Resuming from step %d, epoch %d, batch %d' %
              (step, start_epoch, start_batch), flush=True)
    elif loss_path.exists():
        raise FileExistsError('Refusing to overwrite loss history: %s' % loss_path)
    if step >= args.total_steps:
        raise ValueError('Training has already reached total steps')
    stop_step = min(args.total_steps, args.max_steps or args.total_steps)
    start_time = time.time()
    with loss_path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=LOSS_COLUMNS)
        if step == 0:
            writer.writeheader()
        for epoch in range(start_epoch, math.ceil(args.total_steps / batches_per_epoch)):
            generator = torch.Generator().manual_seed(args.seed + epoch)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                drop_last=True, num_workers=args.num_workers,
                                generator=generator, pin_memory=True)
            model.train()
            model.encoder.eval()
            for batch, images in enumerate(loader):
                if epoch == start_epoch and batch < start_batch:
                    continue
                images = images.to(args.device, non_blocking=True)
                encoder_features, decoder_features = model(images)
                hard_fraction = min(0.9 * step / 1000, 0.9)
                loss = global_cosine_hm_percent(encoder_features, decoder_features,
                                                p=hard_fraction, factor=0.1)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite loss at step %d' % (step + 1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                scheduler.step()
                step += 1
                row = {'step': step, 'epoch': epoch, 'batch': batch,
                       'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr'],
                       'unix_time': time.time()}
                writer.writerow(row)
                stream.flush()
                if step == 1 or step % args.log_every == 0:
                    print('step=%d/%d epoch=%d loss=%.6f lr=%.6g' %
                          (step, args.total_steps, epoch, row['loss'], row['lr']), flush=True)
                if step % args.save_every == 0 or step == stop_step:
                    save_checkpoint(args, model, optimizer, scheduler, metadata,
                                    step, epoch, batch)
                if step >= stop_step:
                    break
            if step >= stop_step:
                break
    if step == args.total_steps:
        final_path = args.output_dir / 'model.pth'
        if final_path.exists():
            raise FileExistsError('Refusing to overwrite final model: %s' % final_path)
        temporary = final_path.with_suffix('.pth.tmp')
        torch.save(model.state_dict(), temporary)
        temporary.replace(final_path)
        print('Final original Dinomaly-B checkpoint:', final_path, flush=True)
    (args.output_dir / 'training.json').write_text(json.dumps({
        'metadata': metadata, 'step': step, 'total_steps': args.total_steps,
        'seconds': time.time() - start_time,
        'final_model': str((args.output_dir / 'model.pth').resolve())
                       if step == args.total_steps else None,
    }, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--backbone-checkpoint', type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--total-steps', type=int, default=50000)
    parser.add_argument('--max-steps', type=int, help='Stop early for a smoke run')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--resume-checkpoint', type=Path)
    args = parser.parse_args()
    if any(value < 1 for value in (args.total_steps, args.batch_size, args.save_every,
                                   args.log_every, args.cpu_threads)):
        parser.error('Step counts, batch size, and CPU threads must be positive')
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be positive')
    if not args.backbone_checkpoint.is_file():
        parser.error('Missing DINOv2-R ViT-B backbone: %s' % args.backbone_checkpoint)
    train(args)


if __name__ == '__main__':
    main()
