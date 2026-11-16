"""MANUAL GPU check on synthetic tensors only; never produces benchmark results."""
import argparse
import time

from common import ROOT, check_lock, config, now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=['arcad', 'dinomaly2'])
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    lock_hash = check_lock()
    cfg = config()
    import torch
    import torch.nn.functional as F
    from model_adapter import build
    torch.manual_seed(9001)
    torch.cuda.manual_seed_all(9001)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model, trainable, extra = build(args.model, cfg, args.device)
    from optimizers import StableAdamW
    from utils import global_cosine_hm_percent
    optimizer = StableAdamW([{'params': trainable.parameters()}], lr=2e-3,
        betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=args.model == 'arcad', eps=1e-10)
    if args.model == 'arcad':
        extra['learner'].prototypes.copy_(F.normalize(torch.randn_like(extra['learner'].prototypes), dim=1))
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    model.encoder.eval()
    for step in range(2):
        optimizer.zero_grad()
        images = torch.randn(cfg['normal_batch_size'], 3, cfg['image_size'], cfg['image_size'], device=args.device)
        output = model(images)
        loss = global_cosine_hm_percent(output[0], output[1], p=0.1, factor=0.1)
        if args.model == 'arcad':
            learner, disc, focal = extra['learner'], extra['discriminator'], extra['focal']
            comp, features, assignments = learner(output[2])
            learner.update_prototypes(features, assignments)
            synthetic = learner.augment_features(features)
            pool = synthetic.detach()
            loss = loss + 0.1 * comp + 0.05 * (
                focal(disc(features), torch.zeros((features.shape[0], 1), device=args.device)) +
                focal(disc(synthetic), torch.ones((synthetic.shape[0], 1), device=args.device)))
        loss.backward()
        del output
        if args.model == 'arcad':
            support = torch.randn(cfg['support_batch_size'], 3, cfg['image_size'], cfg['image_size'], device=args.device)
            _, _, tokens = model(support)
            masks = torch.zeros(cfg['support_batch_size'], 1, cfg['image_size'], cfg['image_size'], device=args.device)
            masks[:, :, 100:200, 100:200] = 1
            flat = F.normalize(tokens[:, 5:].reshape(-1, tokens.shape[-1]), dim=1)
            support_loss = 0.1 * (focal(disc(flat), torch.ones((flat.shape[0], 1), device=args.device)) +
                learner.calculate_contrastive_calibration_loss(tokens, masks, pool))
            support_loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable.parameters(), 0.1)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError('Synthetic smoke check failed')
        optimizer.step()
    model.eval()
    with torch.no_grad():
        result = model(torch.randn(5, 3, cfg['image_size'], cfg['image_size'], device=args.device))
        for value in result[0] + result[1]:
            if not torch.isfinite(value).all():
                raise FloatingPointError('Nonfinite synthetic inference')
    report = dict(status='SMOKE_PASSED', model=args.model, timestamp=now(), steps=2,
        input_lock_sha256=lock_hash,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        arcad_candidate_selection='no-grad search; differentiable selected-candidate recomputation' if args.model == 'arcad' else None,
        input='synthetic random tensors; random unit prototypes for smoke only',
        elapsed_seconds=time.monotonic() - started,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
        warning='Not benchmark evidence; real prototype extraction and full run still required')
    write_json(ROOT / 'logs' / ('smoke_%s_%s.json' % (args.model, now().replace(':', '').replace('+', '_'))), report)
    print(report)


if __name__ == '__main__':
    main()
