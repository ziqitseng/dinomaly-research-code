"""Train or evaluate CostFilter on the original Real-IAD Variety split.

Run from any directory. Model input geometry and CostFilter losses follow the
MVTec runner; Variety has its own 160-class head and synthetic anomaly policy.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score

sys.dont_write_bytecode = True
from dataset import (RealIADVarietyDataset, RealIADVarietyCostFilterTrainDataset,
                     get_data_transforms)
from loss import FocalLoss_gamma, SSIM, SoftIoULoss
from dinov2.models.vision_transformer import vit_base
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from unet3d_att_dino_channel_test_48_min import DiscriminativeSubNetwork_3d_att_dino_channel
from utils import cal_anomaly_maps, cal_anomaly_maps_28, compute_pro, f1_score_max
from realiad_metrics import classification_metrics, sample_scores


HERE = Path(__file__).resolve().parent
def sample_id(path):
    # Variety names include mixed-case defect labels and two exceptional
    # filenames. The manifest's object directory is the stable five-view ID.
    return str(Path(path).parent)


def all_categories(data_path):
    manifest_dir = data_path / 'realiadvariety_jsons/Real-IAD_Variety_jsons'
    categories = sorted(path.stem for path in manifest_dir.glob('*.json'))
    if len(categories) != 160:
        raise ValueError('Expected 160 original Variety category manifests in %s; got %d' %
                         (manifest_dir, len(categories)))
    return categories


def metadata(args, categories):
    manifest_dir = args.data_path / 'realiadvariety_jsons/Real-IAD_Variety_jsons'
    return {
        'protocol': 'Real-IAD Variety original normal-only train/test',
        'categories': {name: index for index, name in enumerate(categories)},
        'manifest_sha256': {name: hashlib.sha256(
            (manifest_dir / (name + '.json')).read_bytes()).hexdigest()
            for name in categories},
        'dinomaly_checkpoint': str(args.dinomaly_checkpoint.resolve()),
        'teacher_description': args.teacher_description,
        'dinomaly_sha256': hashlib.sha256(
            args.dinomaly_checkpoint.read_bytes()).hexdigest(),
        'model_input': 'resize 448, center crop 392',
        'synthetic_foreground': 'full image',
        'filter_blend_weight': 0.05,
        'args': {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
    }


def build_dinomaly(checkpoint, device):
    # The full Dinomaly checkpoint includes encoder weights. Construct the
    # same architecture directly so invocation outside the repo does not
    # trigger vit_encoder.load's relative-path backbone download.
    encoder = vit_base(patch_size=14, img_size=518, block_chunks=0,
                       init_values=1e-8, num_register_tokens=4,
                       interpolate_antialias=False, interpolate_offset=0.1)
    bottleneck = nn.ModuleList([bMlp(768, 3072, 768, drop=0.2)])
    decoder = nn.ModuleList([
        VitBlock(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=LinearAttention2)
        for _ in range(8)])
    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
                   target_layers=[2, 3, 4, 5, 6, 7, 8, 9], mask_neighbor_size=0,
                   fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
                   fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]])
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def build_filter(device):
    model = DiscriminativeSubNetwork_3d_att_dino_channel(
        in_channels=768, out_channels=2, base_channels=48,
        cls_classes=160).float()
    model.apply(weights_init)
    return model.to(device)


def weights_init(module):
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


@torch.no_grad()
def costfilter_features(model, images):
    encoder, decoder = model(images)
    return costfilter_features_from_outputs(encoder, decoder)


def costfilter_features_from_outputs(encoder, decoder):
    batch = encoder[0].shape[0]
    volumes = []
    for rec_feat, org_feat in zip(encoder, decoder):
        assert rec_feat.shape[1:] == (768, 28, 28)
        assert org_feat.shape[1:] == (768, 28, 28)
        pi = org_feat.reshape(batch, 768, 784).transpose(1, 2)
        pr = rec_feat.reshape(batch, 768, 784).transpose(1, 2)
        pi = pi / (torch.norm(pi, p=2, dim=-1, keepdim=True) + 1e-8)
        pr = pr / (torch.norm(pr, p=2, dim=-1, keepdim=True) + 1e-8)
        costs = 1 - torch.bmm(pi, pr.transpose(1, 2))
        indices = torch.topk(costs, 768, dim=-1, largest=False,
                             sorted=False).indices.sort(dim=-1).values
        volumes.append(torch.gather(costs, -1, indices).reshape(batch, 28, 28, 768))
    volume = torch.stack(volumes, dim=0).permute(0, 1, 4, 2, 3)
    volume = F.interpolate(volume.reshape(2 * batch, 768, 28, 28),
                           size=(64, 64), mode='bilinear', align_corners=True)
    volume = volume.reshape(2, batch, 768, 64, 64).permute(1, 2, 0, 3, 4)
    baseline_map, _ = cal_anomaly_maps_28(encoder, decoder, 392)
    guidance = F.interpolate(baseline_map, size=(64, 64), mode='bilinear',
                             align_corners=True).repeat(1, 2, 1, 1)
    features = [feat.reshape(batch, 768, 784).transpose(1, 2) for feat in encoder]
    return volume.float(), features, guidance.float()


def dynamic_gamma(logits, labels):
    # Same confidence and misclassification rule as the MVTec runner, without
    # its MVTec-specific category overrides (which do not define Variety).
    gamma = torch.clamp(3.0 - logits.max(dim=1).values, min=1.5)
    return torch.where(logits.argmax(dim=1) == labels, gamma,
                       torch.full_like(gamma, 3.5))


def train(args, baseline, selected, run_metadata):
    data_transform, gt_transform = get_data_transforms(448, 392)
    ids = run_metadata['categories']
    datasets = [RealIADVarietyCostFilterTrainDataset(
        str(args.data_path), name, ids[name], data_transform, gt_transform,
        str(args.dtd_path)) for name in selected]
    dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)
    if not len(loader):
        raise ValueError('Training split is smaller than batch size')
    print('Training images:', len(dataset), 'categories:', len(selected), flush=True)

    filter_model = build_filter(args.device).train()
    optimizer = torch.optim.Adam(filter_model.parameters(), lr=1e-4,
                                 betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    focal, ssim = FocalLoss_gamma(), SSIM()
    step = 0
    records = []
    start_epoch = 0
    recipe = {'selected_categories': selected, 'batch_size': args.batch_size,
              'num_workers': args.num_workers, 'seed': args.seed,
              'dtd_path': str(args.dtd_path.resolve())}
    loss_log = args.output_dir / 'step_losses.csv'
    if args.resume_checkpoint:
        saved = torch.load(args.resume_checkpoint, map_location='cpu')
        if saved['metadata']['manifest_sha256'] != run_metadata['manifest_sha256']:
            raise ValueError('Resume checkpoint has different Variety manifests')
        if saved['metadata']['dinomaly_sha256'] != run_metadata['dinomaly_sha256']:
            raise ValueError('Resume checkpoint has a different Dinomaly teacher')
        if saved['recipe'] != recipe:
            raise ValueError('Resume checkpoint has a different training recipe')
        filter_model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        step = saved['step']
        records = saved['epoch_records']
        start_epoch = saved['epoch'] + 1
        if not loss_log.exists():
            raise FileNotFoundError('Step-loss history required to resume: %s' % loss_log)
        with loss_log.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) < step:
            raise ValueError('Step-loss history ends before checkpoint step %d' % step)
        with loss_log.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=LOSS_COLUMNS)
            writer.writeheader()
            writer.writerows(row for row in rows if int(row['step']) <= step)
        random.setstate(saved['rng_python'])
        np.random.set_state(saved['rng_numpy'])
        torch.set_rng_state(saved['rng_torch'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved['rng_cuda'])
        print('Resuming after epoch %d, step %d' % (saved['epoch'], step), flush=True)
    elif loss_log.exists():
        raise FileExistsError('Refusing to overwrite step-loss history: %s' % loss_log)
    if start_epoch >= args.epochs:
        raise ValueError('Checkpoint has already reached requested epoch count')
    with loss_log.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=LOSS_COLUMNS)
        if start_epoch == 0:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs):
            step = train_epoch(args, baseline, filter_model, optimizer, focal, ssim,
                               loader, epoch, step, writer, stream, scheduler,
                               run_metadata, recipe, records)
            if args.max_steps and step >= args.max_steps:
                break


LOSS_COLUMNS = ('step', 'epoch', 'total_loss', 'segmentation_loss',
                'classification_loss', 'lr', 'unix_time')


def train_epoch(args, baseline, filter_model, optimizer, focal, ssim, loader,
                epoch, step, writer, stream, scheduler, run_metadata, recipe,
                records):
        start = time.time()
        losses = []
        filter_model.train()
        for images, masks, _, names, labels in tqdm(
                loader, desc='epoch %d' % epoch, disable=not sys.stderr.isatty()):
            images = images.to(args.device)
            masks = masks.to(args.device)
            labels = labels.to(args.device, dtype=torch.long)
            volume, features, guidance = costfilter_features(baseline, images)
            optimizer.zero_grad(set_to_none=True)
            logits, class_logits = filter_model(volume, features, guidance)
            if class_logits.shape != (len(labels), 160):
                raise AssertionError('Expected a 160-class Variety head')
            probabilities = F.interpolate(torch.softmax(logits, dim=1),
                                           size=(392, 392), mode='bilinear',
                                           align_corners=True)
            anomaly_probability = probabilities[:, 1:2]
            gammas = dynamic_gamma(class_logits, labels)
            focal_loss = torch.stack([
                focal(probabilities[i:i + 1], masks[i:i + 1], gammas[i])
                for i in range(len(labels))]).mean()
            segmentation_loss = (focal_loss + 0.1 * ssim(anomaly_probability, masks)
                                 + F.mse_loss(anomaly_probability, masks)
                                 + 0.1 * SoftIoULoss(anomaly_probability, masks))
            classification_loss = F.cross_entropy(class_logits, labels)
            loss = segmentation_loss + 0.5 * classification_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Invalid loss at step %d' % (step + 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter_model.parameters(), 2.0)
            optimizer.step()
            step += 1
            row = {'step': step, 'epoch': epoch, 'total_loss': loss.item(),
                   'segmentation_loss': segmentation_loss.item(),
                   'classification_loss': classification_loss.item(),
                   'lr': optimizer.param_groups[0]['lr'],
                   'unix_time': time.time()}
            writer.writerow(row)
            stream.flush()
            losses.append(row['total_loss'])
            if step == 1 or step % 100 == 0:
                print('step=%d epoch=%d loss=%.6f' %
                      (step, epoch, losses[-1]), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        mean_loss = float(np.mean(losses))
        scheduler.step(mean_loss)
        checkpoint = args.output_dir / ('epoch_%d_step_%d_variety.pth' % (epoch, step))
        if checkpoint.exists():
            raise FileExistsError('Refusing to overwrite checkpoint: %s' % checkpoint)
        records.append({'epoch': epoch, 'step': step, 'mean_loss': mean_loss,
                        'seconds': time.time() - start,
                        'checkpoint': str(checkpoint.resolve())})
        temporary_checkpoint = checkpoint.with_suffix('.pth.tmp')
        torch.save({'model_state_dict': filter_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch, 'step': step,
                    'epoch_records': records,
                    'recipe': recipe,
                    'rng_python': random.getstate(),
                    'rng_numpy': np.random.get_state(),
                    'rng_torch': torch.get_rng_state(),
                    'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    'metadata': run_metadata}, temporary_checkpoint)
        temporary_checkpoint.replace(checkpoint)
        (args.output_dir / 'training.json').write_text(json.dumps(
            {'metadata': run_metadata, 'epochs': records}, indent=2) + '\n')
        print('Saved:', checkpoint, 'mean loss:', mean_loss, flush=True)
        return step


def eval_subset(dataset, per_label, objects_per_label):
    if per_label is not None and objects_per_label is not None:
        raise ValueError('Choose view or object subset, not both')
    if per_label is None and objects_per_label is None:
        return dataset
    chosen = []
    if per_label is not None:
        for label in (False, True):
            indices = np.flatnonzero(dataset.labels == label)[:per_label]
            if len(indices) != per_label:
                raise ValueError('Not enough Variety test views of label %s' % label)
            chosen.extend(indices.tolist())
    else:
        groups = sample_scores(dataset.img_paths, [0.] * len(dataset),
                               dataset.labels, expected_views=5, key_fn=sample_id)
        for label in (0, 1):
            matches = [row for row in groups if row['label'] == label][:objects_per_label]
            if len(matches) != objects_per_label:
                raise ValueError('Not enough Variety test objects of label %d' % label)
            for row in matches:
                chosen.extend(row['indices'])
    return Subset(dataset, chosen)


def pixel_metrics(masks, maps):
    masks = np.concatenate(masks, axis=0).astype(np.uint8)
    maps = np.concatenate(maps, axis=0).astype(np.float32)
    if set(np.unique(masks)) != {0, 1}:
        raise ValueError('Pixel metrics require both normal and anomalous pixels')
    return {
        'P_AUROC': float(roc_auc_score(masks.ravel(), maps.ravel())),
        'P_AP': float(average_precision_score(masks.ravel(), maps.ravel())),
        'P_F1max': float(f1_score_max(masks.ravel(), maps.ravel())),
        'AUPRO': float(compute_pro(masks, maps)),
    }


@torch.no_grad()
def evaluate_category(baseline, filter_model, loader, device, metric_size,
                      with_pixel_metrics, pixel_backend):
    branches = ['baseline'] + (['costfilter'] if filter_model is not None else [])
    paths, labels = [], []
    scores = {branch: [] for branch in branches}
    predictions = {branch: [] for branch in branches}
    masks = []
    pixel_annotated_views = 0
    unannotated_anomalous_views = 0
    pixel_accumulators = {}
    if with_pixel_metrics and pixel_backend == 'adeval':
        from adeval import EvalAccumulatorCuda
        if not str(device).startswith('cuda'):
            raise ValueError('ADEval CUDA pixel backend requires a CUDA device')
        pixel_accumulators = {
            branch: EvalAccumulatorCuda(-0.1, 2.1, -0.1, 2.1,
                                        nstrips=10000)
            for branch in branches}
    for images, gt, batch_labels, batch_paths, batch_pixel_annotated in tqdm(
            loader, desc='inference', disable=not sys.stderr.isatty()):
        valid_pixels = batch_pixel_annotated.bool()
        pixel_annotated_views += int(valid_pixels.sum())
        unannotated_anomalous_views += int((~valid_pixels).sum())
        images = images.to(device)
        encoder, decoder = baseline(images)
        baseline_map, _ = cal_anomaly_maps(encoder, decoder, 392)
        maps = {'baseline': baseline_map}
        if filter_model is not None:
            volume, features, guidance = costfilter_features_from_outputs(encoder, decoder)
            logits, _ = filter_model(volume, features, guidance)
            filtered_map = F.interpolate(torch.softmax(logits, dim=1)[:, 1:2],
                                         size=(392, 392), mode='bilinear',
                                         align_corners=True)
            maps['costfilter'] = 0.95 * baseline_map + 0.05 * filtered_map
        if with_pixel_metrics:
            resized_gt = F.interpolate(gt, size=(metric_size, metric_size),
                                       mode='nearest').bool()
            resized_gt = resized_gt.any(dim=1)
            if valid_pixels.any():
                if pixel_backend == 'exact':
                    masks.append(resized_gt[valid_pixels].cpu().numpy())
                else:
                    resized_gt = resized_gt.to(device=device, dtype=torch.uint8)
        for branch, anomaly_map in maps.items():
            anomaly_map = F.interpolate(anomaly_map, size=(metric_size, metric_size),
                                        mode='bilinear', align_corners=False)
            flat = anomaly_map.flatten(1)
            k = min(flat.shape[1], max(int(flat.shape[1] * 0.01), 250))
            score = torch.topk(flat, k=k, dim=1).values.mean(dim=1)
            scores[branch].extend(score.cpu().tolist())
            if with_pixel_metrics and valid_pixels.any() and pixel_backend == 'exact':
                predictions[branch].append(anomaly_map[valid_pixels, 0].cpu().numpy())
            elif with_pixel_metrics and valid_pixels.any():
                pixel_accumulators[branch].add_anomap_batch(
                    anomaly_map[valid_pixels.to(device), 0], resized_gt[valid_pixels.to(device)])
        paths.extend(batch_paths)
        labels.extend(batch_labels.int().tolist())

    metrics = {}
    object_rows = {}
    for branch in branches:
        view = classification_metrics(labels, scores[branch])
        metrics[branch] = {'I_' + key: value for key, value in view.items()}
        object_rows[branch] = sample_scores(paths, scores[branch], labels,
                                           expected_views=None, key_fn=sample_id)
        if all(row['view_count'] == 5 for row in object_rows[branch]):
            obj = classification_metrics([row['label'] for row in object_rows[branch]],
                                         [row['score'] for row in object_rows[branch]])
            metrics[branch].update({'S_' + key: value for key, value in obj.items()})
        if with_pixel_metrics:
            if pixel_backend == 'exact':
                metrics[branch].update(pixel_metrics(masks, predictions[branch]))
            else:
                accumulator = pixel_accumulators[branch]
                accumulator.add_image(scores[branch], labels)
                accumulator.add_sample(
                    [row['score'] for row in object_rows[branch]],
                    [row['label'] for row in object_rows[branch]])
                summary = accumulator.summary()
                precision_recall = accumulator._pixel_acc.pr()
                recall, precision = precision_recall[:2]
                f1 = 2 * precision * recall / (precision + recall + 1e-7)
                metrics[branch].update({
                    'P_AUROC': summary['p_auroc'],
                    'P_AP': summary['p_aupr'],
                    'P_F1max_approx': float(np.nanmax(f1)),
                    'AUPRO': summary['p_aupro'],
                })
    return {'metrics': metrics, 'paths': paths, 'labels': labels,
            'scores': scores, 'object_rows': object_rows,
            'complete_objects': all(row['view_count'] == 5
                                    for row in object_rows['baseline']),
            'pixel_annotated_views': pixel_annotated_views,
            'unannotated_anomalous_views': unannotated_anomalous_views}


def write_score_tables(output_dir, name, result):
    branches = list(result['scores'])
    with (output_dir / (name + '_views.csv')).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'sample_id', 'label'] + branches)
        for index, path in enumerate(result['paths']):
            writer.writerow([path, sample_id(path), result['labels'][index]] +
                            [result['scores'][branch][index] for branch in branches])
    with (output_dir / (name + '_objects.csv')).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'view_count', 'label'] + branches)
        groups = result['object_rows']['baseline']
        for index, row in enumerate(groups):
            writer.writerow([row['sample_id'], row['view_count'], row['label']] +
                            [result['object_rows'][branch][index]['score']
                             for branch in branches])


def evaluate(args, baseline, selected, run_metadata):
    filter_model = None
    if not args.baseline_only:
        if not args.costfilter_checkpoint:
            raise ValueError('--costfilter-checkpoint is required for paired evaluation')
        saved = torch.load(args.costfilter_checkpoint, map_location='cpu')
        if saved.get('metadata', {}).get('protocol') != run_metadata['protocol']:
            raise ValueError('Checkpoint was not trained with the original Variety protocol')
        for key in ('categories', 'manifest_sha256', 'dinomaly_sha256'):
            if saved['metadata'][key] != run_metadata[key]:
                raise ValueError('Checkpoint %s does not match this run' % key)
        filter_model = build_filter(args.device)
        filter_model.load_state_dict(saved['model_state_dict'], strict=True)
        filter_model.eval()
    data_transform, gt_transform = get_data_transforms(448, 392)
    metrics_path = args.output_dir / 'metrics.json'
    results = {}
    payload = None
    if args.resume_eval and metrics_path.exists():
        previous = json.loads(metrics_path.read_text())
        for key in ('categories', 'manifest_sha256', 'dinomaly_sha256'):
            if previous['metadata'][key] != run_metadata[key]:
                raise ValueError('Previous evaluation has different %s' % key)
        for key in ('metric_size', 'eval_per_label', 'eval_objects_per_label',
                    'skip_pixel_metrics', 'pixel_backend'):
            if previous[key] != getattr(args, key):
                raise ValueError('Previous evaluation has different %s' % key)
        if previous['checkpoint'] != str(args.costfilter_checkpoint):
            raise ValueError('Previous evaluation used a different filter checkpoint')
        results = previous['categories']
        payload = previous
    elif metrics_path.exists():
        raise FileExistsError('Evaluation results exist; use --resume-eval: %s' % metrics_path)
    start = time.time()
    for name in selected:
        if name in results:
            print('Skipping completed category:', name, flush=True)
            continue
        dataset = RealIADVarietyDataset(str(args.data_path), name, data_transform,
                                 gt_transform, phase='test')
        subset = eval_subset(dataset, args.eval_per_label, args.eval_objects_per_label)
        loader = DataLoader(subset, batch_size=args.eval_batch_size,
                            num_workers=args.num_workers, shuffle=False)
        result = evaluate_category(baseline, filter_model, loader, args.device,
                                   args.metric_size, not args.skip_pixel_metrics,
                                   args.pixel_backend)
        write_score_tables(args.output_dir, name, result)
        metrics = result['metrics']
        results[name] = {'baseline': metrics['baseline'],
                         'views': len(result['paths']),
                         'objects': len(result['object_rows']['baseline']),
                         'complete_objects': result['complete_objects'],
                         'pixel_annotated_views': result['pixel_annotated_views'],
                         'unannotated_anomalous_views': result['unannotated_anomalous_views'],
                         'unannotated_anomalous_views_in_category': dataset.missing_mask_annotations}
        if filter_model is not None:
            results[name]['costfilter'] = metrics['costfilter']
            results[name]['delta'] = {key: metrics['costfilter'][key] - value
                                      for key, value in metrics['baseline'].items()}
        payload = {'metadata': run_metadata, 'checkpoint': str(args.costfilter_checkpoint),
                   'metric_size': args.metric_size, 'eval_per_label': args.eval_per_label,
                   'eval_objects_per_label': args.eval_objects_per_label,
                   'skip_pixel_metrics': args.skip_pixel_metrics,
                   'pixel_backend': args.pixel_backend,
                   'categories': results, 'complete': False}
        metrics_path.write_text(json.dumps(payload, indent=2) + '\n')
        print(name, results[name], flush=True)
    payload['mean'] = {}
    for branch in ('baseline', 'costfilter', 'delta'):
        if branch in next(iter(results.values())):
            keys = results[selected[0]][branch]
            payload['mean'][branch] = {
                key: float(np.mean([result[branch][key] for result in results.values()]))
                for key in keys}
    payload['seconds'] = time.time() - start
    payload['complete'] = (args.eval_per_label is None and
                           args.eval_objects_per_label is None and len(selected) == 160)
    metrics_path.write_text(json.dumps(payload, indent=2) + '\n')
    print('Mean:', payload['mean'], 'complete:', payload['complete'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true', help='Train; default is evaluation')
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--dtd-path', type=Path, help='DTD images directory, required for training')
    parser.add_argument('--dinomaly-checkpoint', type=Path, required=True,
                        help='Original ViTill Dinomaly-B checkpoint; Dinomaly2 weights are incompatible')
    parser.add_argument('--teacher-description', required=True,
                        help='Record teacher training dataset, e.g. Real-IAD cross-dataset transfer')
    parser.add_argument('--costfilter-checkpoint', type=Path)
    parser.add_argument('--baseline-only', action='store_true',
                        help='Evaluate the frozen Dinomaly-B teacher without CostFilter')
    parser.add_argument('--resume-eval', action='store_true',
                        help='Skip categories already saved in metrics.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--categories', nargs='+', help='Selected category names; default all 160')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=17)
    parser.add_argument('--max-steps', type=int, help='Stop training after this many steps')
    parser.add_argument('--resume-checkpoint', type=Path,
                        help='Resume at the epoch after this checkpoint')
    parser.add_argument('--eval-per-label', type=int,
                        help='Smoke evaluation: first N normal and N anomalous views per category')
    parser.add_argument('--eval-objects-per-label', type=int,
                        help='Smoke evaluation: N normal and N anomalous complete five-view objects')
    parser.add_argument('--metric-size', type=int, default=256)
    parser.add_argument('--skip-pixel-metrics', action='store_true',
                        help='Compute image and sample metrics only')
    parser.add_argument('--pixel-backend', choices=('exact', 'adeval'),
                        default='exact', help='Exact MVTec-style or streaming ADEval pixel metrics')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if args.train and (args.dtd_path is None or args.batch_size < 2):
        parser.error('Training requires --dtd-path and --batch-size >= 2')
    if args.epochs < 1 or args.eval_batch_size < 1 or args.metric_size < 1:
        parser.error('Epochs, evaluation batch size, and metric size must be positive')
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be positive')
    if args.eval_per_label is not None and args.eval_per_label < 1:
        parser.error('--eval-per-label must be positive')
    if args.eval_objects_per_label is not None and args.eval_objects_per_label < 1:
        parser.error('--eval-objects-per-label must be positive')
    if args.eval_per_label is not None and args.eval_objects_per_label is not None:
        parser.error('Choose --eval-per-label or --eval-objects-per-label')
    if not args.dinomaly_checkpoint.is_file():
        parser.error('Missing Dinomaly checkpoint: %s' % args.dinomaly_checkpoint)
    categories = all_categories(args.data_path)
    selected = args.categories or categories
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(categories):
        parser.error('--categories must contain unique names from the original manifests')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    run_metadata = metadata(args, categories)
    (args.output_dir / 'protocol.json').write_text(json.dumps(run_metadata, indent=2) + '\n')
    baseline = build_dinomaly(args.dinomaly_checkpoint, args.device)
    if args.train:
        train(args, baseline, selected, run_metadata)
    else:
        evaluate(args, baseline, selected, run_metadata)


if __name__ == '__main__':
    main()
