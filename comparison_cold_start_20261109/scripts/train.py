"""MANUAL stage: fresh matched-budget training, no test evaluation/selection."""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

from common import ROOT, check_prepared, config, load_cached_mask, load_records, now, run_dir, sha256, write_json


class BatchStream:
    """Serializable independent RNG: both methods receive identical normal batches."""
    def __init__(self, size, batch_size, seed):
        if size < batch_size:
            raise ValueError('Training partition smaller than batch')
        self.size, self.batch_size = size, batch_size
        self.rng = random.Random(seed)
        self.order, self.offset = [], 0

    def next(self):
        if self.offset + self.batch_size > len(self.order):
            self.order = list(range(self.size))
            self.rng.shuffle(self.order)
            self.offset = 0
        result = self.order[self.offset:self.offset + self.batch_size]
        self.offset += self.batch_size
        return result

    def state_dict(self):
        return dict(size=self.size, batch_size=self.batch_size, order=self.order,
                    offset=self.offset, rng=self.rng.getstate())

    def load_state_dict(self, state):
        if (state['size'], state['batch_size']) != (self.size, self.batch_size):
            raise RuntimeError('Sampler budget changed')
        self.order, self.offset = state['order'], state['offset']
        self.rng.setstate(state['rng'])


def batch(records, indices, device, masks=False):
    import numpy as np
    import torch
    from PIL import Image
    images, targets = [], []
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    for index in indices:
        r = records[index]
        with Image.open(ROOT / 'prepared' / r['cache_image']) as image:
            array = np.asarray(image).copy()
            if hashlib.sha256(array.tobytes()).hexdigest() != r['input_pixel_sha256']:
                raise RuntimeError('RGB cache differs from prepared pixels: ' + r['id'])
            value = torch.from_numpy(array).permute(2, 0, 1).float() / 255.
        images.append((value - mean) / std)
        if masks:
            targets.append(torch.from_numpy(load_cached_mask(r, 'cache_train_mask')).float()[None])
    result = torch.stack(images).to(device)
    return (result, torch.stack(targets).to(device)) if masks else result


def initialize_prototypes(model, learner, normals, cfg, seed, target, device):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.cluster import MiniBatchKMeans
    model.eval()
    generator = torch.Generator().manual_seed(seed + 12345)
    per_view = min(cfg['prototype_patches_per_view'], cfg['prototype_patch_cap'] // len(normals))
    if per_view < 1:
        raise RuntimeError('Prototype cap cannot cover every normal view')
    features = []
    with torch.no_grad():
        for start in range(0, len(normals), cfg['normal_batch_size']):
            indices = list(range(start, min(start + cfg['normal_batch_size'], len(normals))))
            _, _, tokens = model(batch(normals, indices, device))
            spatial = tokens[:, 5:]
            for view in spatial:
                selected = torch.randperm(view.shape[0], generator=generator)[:per_view].to(device)
                features.append(F.normalize(view[selected], dim=1).cpu().numpy())
    features = np.concatenate(features)
    if len(features) < cfg['prototype_count']:
        raise RuntimeError('Too few patches for requested prototype count')
    kmeans = MiniBatchKMeans(n_clusters=cfg['prototype_count'], batch_size=16384,
        n_init=1, max_no_improvement=20, random_state=seed)
    kmeans.fit(features)
    prototypes = F.normalize(torch.from_numpy(kmeans.cluster_centers_), dim=1).to(device)
    if not torch.isfinite(prototypes).all() or (prototypes.norm(dim=1) < 0.99).any():
        raise RuntimeError('Invalid prototype initialization')
    learner.prototypes.copy_(prototypes)
    learner.is_initialized = True
    torch.save(prototypes.cpu(), target / 'prototypes_init.pth')
    write_json(target / 'prototype_provenance.json', {
        'normal_ids': [r['id'] for r in normals], 'normal_views': len(normals),
        'patches': len(features), 'patches_per_view': per_view, 'seed': seed,
        'initial_model_sha256': sha256(target / 'initial_model.pth'),
        'prototypes_sha256': sha256(target / 'prototypes_init.pth'),
        'method': 'same initialized training bottleneck; eval dropout; all selected normals; normalized MiniBatchKMeans'})


def atomic_checkpoint(path, state):
    import torch
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=['arcad', 'dinomaly2'])
    parser.add_argument('seed', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true', help='Continue latest full state in this same run')
    args = parser.parse_args()
    prepared = check_prepared()
    cfg = config()
    if args.seed not in cfg['training_seeds']:
        raise ValueError('Seed not declared in frozen config')
    # cuBLAS deterministic configuration must precede CUDA initialization.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    import torch.nn.functional as F
    from model_adapter import build, torch_load
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or not args.device.startswith('cuda'):
        raise RuntimeError('Training stage requires a CUDA GPU')
    target = run_dir(args.model, args.seed)
    if args.resume:
        if (target / 'TRAIN_COMPLETE.json').exists():
            raise RuntimeError('Training already complete')
        previous = torch_load(target / 'latest.pth', args.device)
        if previous['model_name'] != args.model or previous['seed'] != args.seed:
            raise RuntimeError('Resume model/seed identity mismatch')
        if previous['config_sha256'] != sha256(ROOT / 'config/experiment.json'):
            raise RuntimeError('Resume config mismatch')
        if previous['prepared_sha256'] != sha256(ROOT / 'prepared/COMPLETE.json'):
            raise RuntimeError('Resume prepared input mismatch')
    else:
        target.mkdir(parents=True, exist_ok=False)
        previous = None
    records = load_records()
    normals = [r for r in records if r['partition'] == 'normal_train']
    support = [r for r in records if r['partition'] == 'arcad_support' and r['support_eligible']]
    if any(r['is_ng_without_mask'] or r['protected_test_group'] or not r['raw_mask'] for r in support):
        raise RuntimeError('Unannotated NG or protected test object entered ArcAD support')
    model, trainable, extra = build(args.model, cfg, args.device)
    from optimizers import StableAdamW
    from utils import global_cosine_hm_percent
    optimizer = StableAdamW([{'params': trainable.parameters()}], lr=2e-3,
        betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=args.model == 'arcad', eps=1e-10)
    normal_stream = BatchStream(len(normals), cfg['normal_batch_size'], args.seed)
    support_stream = BatchStream(len(support), cfg['support_batch_size'], args.seed + 10000) if args.model == 'arcad' else None
    start_step = 0
    if previous:
        model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        for name in ('learner', 'discriminator'):
            if name in extra:
                extra[name].load_state_dict(previous[name])
        normal_stream.load_state_dict(previous['normal_stream'])
        if support_stream:
            support_stream.load_state_dict(previous['support_stream'])
        random.setstate(previous['python_rng'])
        np.random.set_state(previous['numpy_rng'])
        torch.set_rng_state(previous['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in previous['cuda_rng']])
        start_step = previous['step']
    else:
        write_json(target / 'run_config.json', dict(cfg, model=args.model, seed=args.seed,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            arcad_candidate_selection='no-grad search; differentiable selected-candidate recomputation' if args.model == 'arcad' else None,
            created_at=now(), normal_views=len(normals), support_views=len(support) if support_stream else 0,
            prepared_sha256=sha256(ROOT / 'prepared/COMPLETE.json')))
        torch.save(model.state_dict(), target / 'initial_model.pth')
        if args.model == 'arcad':
            initialize_prototypes(model, extra['learner'], normals, cfg, args.seed, target, args.device)
    model.train()
    model.encoder.eval()
    if args.model == 'arcad':
        extra['discriminator'].train()
    started = time.monotonic()
    log = (target / 'training.jsonl').open('a', buffering=1)
    for step in range(start_step, cfg['total_steps']):
        # Same schedule for both models; native schedule differences are explicitly harmonized.
        warmup = 100
        if step < warmup:
            lr = 2e-3 * (step + 1) / warmup
        else:
            phase = (step - warmup) / max(1, cfg['total_steps'] - warmup - 1)
            lr = 2e-4 + 0.5 * (2e-3 - 2e-4) * (1 + math.cos(math.pi * phase))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad()
        indices = normal_stream.next()
        output = model(batch(normals, indices, args.device))
        recon = global_cosine_hm_percent(output[0], output[1], p=min(0.9 * step / 1000, 0.9), factor=0.1)
        values = {'reconstruction': recon}
        normal_loss = recon
        if args.model == 'arcad':
            learner, disc, focal = extra['learner'], extra['discriminator'], extra['focal']
            comp, features, assignments = learner(output[2])
            learner.update_prototypes(features, assignments)
            fake = learner.augment_features(features)
            synthetic_pool = fake.detach()
            cls_normal = 0.5 * (focal(disc(features), torch.zeros((features.shape[0], 1), device=args.device))
                + focal(disc(fake), torch.ones((fake.shape[0], 1), device=args.device)))
            normal_loss = normal_loss + 0.1 * comp + 0.1 * cls_normal
            values.update(clustering=comp, classification_normal=cls_normal)
        if not torch.isfinite(normal_loss):
            raise FloatingPointError('Nonfinite normal loss at step %d' % (step + 1))
        normal_loss.backward()
        # Free the normal forward graph before support forward; normal sampling is identical to Dinomaly2.
        del output
        if args.model == 'arcad':
            support_indices = support_stream.next()
            images, masks = batch(support, support_indices, args.device, masks=True)
            _, _, tokens = model(images)
            flat = F.normalize(tokens[:, 5:].reshape(-1, tokens.shape[-1]), dim=1)
            cls_support = focal(disc(flat), torch.ones((flat.shape[0], 1), device=args.device))
            calibration = learner.calculate_contrastive_calibration_loss(tokens, masks, synthetic_pool)
            support_loss = 0.1 * (cls_support + calibration)
            if not torch.isfinite(support_loss):
                raise FloatingPointError('Nonfinite support loss at step %d' % (step + 1))
            support_loss.backward()
            values.update(classification_support=cls_support, calibration=calibration)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable.parameters(), 0.1)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError('Nonfinite gradients at step %d' % (step + 1))
        optimizer.step()
        if (step + 1) % cfg['log_interval'] == 0 or step == start_step:
            if not all(torch.isfinite(p).all() for p in trainable.parameters()):
                raise FloatingPointError('Nonfinite trained parameters')
            report = dict(step=step + 1, lr=lr, gradient_norm=float(grad_norm), timestamp=now(),
                elapsed_segment_seconds=time.monotonic() - started,
                normal_ids=[normals[i]['id'] for i in indices],
                losses={key: float(value.detach()) for key, value in values.items()})
            if args.model == 'arcad':
                norms = learner.prototypes.norm(dim=1)
                if not torch.isfinite(norms).all() or norms.min() < 0.99:
                    raise FloatingPointError('Prototype collapse/nonfinite values')
                report.update(prototype_norm_min=float(norms.min()), prototype_norm_max=float(norms.max()),
                    support_ids=[support[i]['id'] for i in support_indices])
            log.write(json.dumps(report, allow_nan=False) + '\n')
            print(json.dumps(report), flush=True)
        if (step + 1) % cfg['checkpoint_interval'] == 0 or step + 1 == cfg['total_steps']:
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step + 1,
                model_name=args.model, seed=args.seed, normal_stream=normal_stream.state_dict(),
                python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(), config_sha256=sha256(ROOT / 'config/experiment.json'),
                prepared_sha256=sha256(ROOT / 'prepared/COMPLETE.json'))
            if args.model == 'arcad':
                state.update(learner=learner.state_dict(), discriminator=disc.state_dict(),
                             support_stream=support_stream.state_dict())
            atomic_checkpoint(target / 'latest.pth', state)
    log.close()
    atomic_checkpoint(target / 'model.pth', model.state_dict())
    write_json(target / 'TRAIN_COMPLETE.json', dict(completed_at=now(), steps=cfg['total_steps'],
        model=args.model, seed=args.seed,
        model_sha256=sha256(target / 'model.pth'), config_sha256=sha256(ROOT / 'config/experiment.json'),
        prepared_sha256=sha256(ROOT / 'prepared/COMPLETE.json')))


if __name__ == '__main__':
    main()
