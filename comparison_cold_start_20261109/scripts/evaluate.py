"""MANUAL stage: shared held-out predictions, separate object/view/pixel targets."""
import argparse
import collections
import hashlib
import json
import time

from common import ROOT, check_prepared, config, load_cached_mask, load_records, now, protocol_b_ids, protocol_b_label, read_json, run_dir, sha256, write_csv, write_json


def predict_maps(model, records, cfg, device, kernel):
    import torch
    import torch.nn.functional as F
    from train import batch
    output = model(batch(records, list(range(len(records))), device))
    maps = []
    for en, de in zip(output[0], output[1]):
        cosine = (1 - F.cosine_similarity(en, de, dim=1))[:, None]
        maps.append(F.interpolate(cosine, size=cfg['image_size'], mode='bilinear', align_corners=True))
    maps = torch.stack(maps).mean(dim=0)
    maps = F.interpolate(maps, size=cfg['map_size'], mode='bilinear', align_corners=False)
    return F.conv2d(maps, kernel, padding=2)[:, 0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=['arcad', 'dinomaly2'])
    parser.add_argument('seed', type=int)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    check_prepared()
    cfg = config()
    listed_ids = protocol_b_ids(cfg)  # Read the explicit frozen list at scoring time.
    import numpy as np
    import torch
    from error_analysis import analyze_errors, fit_normal_threshold
    from metrics import PixelMetrics, binary_metrics, top_score
    from model_adapter import build, torch_load
    target = run_dir(args.model, args.seed)
    completion = read_json(target / 'TRAIN_COMPLETE.json')
    if completion['model'] != args.model or completion['seed'] != args.seed:
        raise RuntimeError('Completed training model/seed identity mismatch')
    if completion['steps'] != cfg['total_steps']:
        raise RuntimeError('Training budget incomplete')
    if completion['config_sha256'] != sha256(ROOT / 'config/experiment.json'):
        raise RuntimeError('Training config changed')
    if completion['prepared_sha256'] != sha256(ROOT / 'prepared/COMPLETE.json'):
        raise RuntimeError('Training/test membership changed')
    if completion['model_sha256'] != sha256(target / 'model.pth'):
        raise RuntimeError('Checkpoint differs from completed training')
    evaluation = target / 'evaluation'
    evaluation.mkdir(exist_ok=False)
    (evaluation / 'pixel_statistics').mkdir()
    if cfg['save_maps']:
        (evaluation / 'maps').mkdir()
    model, _, _ = build(args.model, cfg, args.device)
    model.load_state_dict(torch_load(target / 'model.pth', args.device), strict=True)
    model.eval()
    coordinates = torch.arange(5, device=args.device).float() - 2
    axis = torch.exp(-coordinates.square() / (2 * 4. ** 2))
    kernel = (axis[:, None] * axis[None, :])
    kernel = (kernel / kernel.sum())[None, None]
    groups, calibration = collections.defaultdict(list), collections.defaultdict(list)
    all_records = load_records()
    for record in all_records:
        if record['partition'] == 'test':
            groups[record['group']].append(record)
        elif record['partition'] == 'normal_calibration':
            calibration[record['category']].append(record)
    if {r['id'] for rs in groups.values() for r in rs if r['id'] in listed_ids} != listed_ids:
        raise RuntimeError('All listed Protocol B views must remain in the shared test set')
    categories = sorted({rs[0]['category'] for rs in groups.values()})
    view_rows, object_rows, category_rows = [], [], []
    started = time.monotonic()
    thresholds, calibration_rows = {}, []
    with torch.no_grad():
        for category in categories:
            normal_rows = []
            for start in range(0, len(calibration[category]), cfg['normal_batch_size']):
                rs = calibration[category][start:start + cfg['normal_batch_size']]
                maps = predict_maps(model, rs, cfg, args.device, kernel)
                for r, amap in zip(rs, maps):
                    normal_rows.append(dict(id=r['id'], category=category, source_split=r['source_split'],
                        partition=r['partition'], object_label=r['object_label'], is_ng_without_mask=r['is_ng_without_mask'],
                        quality_flags='|'.join(r['exclude_reasons']), score=top_score(amap, cfg['top_pixel_fraction'])))
            thresholds[category] = fit_normal_threshold(normal_rows, cfg['false_alarm_target_fpr'])
            calibration_rows.extend(normal_rows)
            pixel = PixelMetrics(cfg['pixel_histogram_bins'])
            category_views, category_objects = [], []
            for group, records in sorted(groups.items()):
                if records[0]['category'] != category:
                    continue
                records.sort(key=lambda r: r['camera'])
                if [r['camera'] for r in records] != [1, 2, 3, 4, 5]:
                    raise RuntimeError('Evaluation object lacks five views')
                maps = predict_maps(model, records, cfg, args.device, kernel)
                if not np.isfinite(maps).all():
                    raise FloatingPointError('Nonfinite held-out map: ' + group)
                for r, amap in zip(records, maps):
                    score = top_score(amap, cfg['top_pixel_fraction'])
                    image_label = protocol_b_label(r, listed_ids)
                    if image_label != r['protocol_b_image_label'] or (r['id'] in listed_ids) != r['is_ng_without_mask']:
                        raise RuntimeError('Prepared labels disagree with scoring-time Protocol B list')
                    if r['id'] in listed_ids and not r['pixel_eligible']:
                        raise RuntimeError('Protocol B missing-mask pixels must be included')
                    row = dict(id=r['id'], group=group, category=category, camera=r['camera'],
                        source_class=r['source_class'], object_label=r['object_label'], score=score,
                        protocol_b_image_label=image_label, is_ng_without_mask=r['id'] in listed_ids,
                        quality_flags='|'.join(r['exclude_reasons']),
                        visible_eligible=r['visible_eligible'], visible_label=r.get('visible_label'),
                        pixel_eligible=r['pixel_eligible'], annotation_issues='|'.join(r['annotation_issues']))
                    view_rows.append(row)
                    category_views.append(row)
                    if r['pixel_eligible']:
                        if r['object_label'] == 0:
                            mask = np.zeros(amap.shape, dtype=bool)
                        else:
                            mask = load_cached_mask(r, 'cache_eval_mask')
                            if r['id'] in listed_ids and mask.any():
                                raise RuntimeError('Protocol B EMPTY mask must contain zero foreground')
                        pixel.update(mask, amap)
                object_row = dict(group=group, category=category, label=records[0]['object_label'],
                                  score=top_score(maps, cfg['top_pixel_fraction']))
                object_rows.append(object_row)
                category_objects.append(object_row)
                if cfg['save_maps']:
                    filename = hashlib.sha256(group.encode()).hexdigest() + '.npz'
                    np.savez_compressed(evaluation / 'maps' / filename, maps=maps,
                                        ids=np.array([r['id'] for r in records]), group=group)
            visible = [r for r in category_views if r['visible_eligible']]
            obj_metrics = binary_metrics([r['label'] for r in category_objects], [r['score'] for r in category_objects])
            view_metrics = binary_metrics([r['visible_label'] for r in visible], [r['score'] for r in visible])
            source_view = binary_metrics([r['object_label'] for r in category_views], [r['score'] for r in category_views])
            b_view = binary_metrics([r['protocol_b_image_label'] for r in category_views], [r['score'] for r in category_views])
            result = dict(category=category, I_B_AUROC=b_view['auroc'], I_B_AP=b_view['ap'],
                O_source_AUROC=obj_metrics['auroc'], O_source_AP=obj_metrics['ap'],
                I_visible_AUROC=view_metrics['auroc'], I_visible_AP=view_metrics['ap'],
                I_source_AUROC=source_view['auroc'], I_source_AP=source_view['ap'],
                objects=len(category_objects), views=len(category_views), visible_views=len(visible),
                ng_without_mask_views=sum(r['is_ng_without_mask'] for r in category_views),
                ng_without_mask_background_pixels=sum(r['is_ng_without_mask'] for r in category_views) * cfg['map_size'] ** 2,
                b_normal_views=sum(r['protocol_b_image_label'] == 0 for r in category_views),
                b_anomaly_views=sum(r['protocol_b_image_label'] == 1 for r in category_views),
                normal_objects=sum(r['label'] == 0 for r in category_objects),
                defective_objects=sum(r['label'] == 1 for r in category_objects),
                visible_normal_views=sum(r['visible_label'] == 0 for r in visible),
                visible_defective_views=sum(r['visible_label'] == 1 for r in visible),
                pixel_b_normal_views=sum(r['pixel_eligible'] and r['protocol_b_image_label'] == 0 for r in category_views),
                pixel_b_anomaly_views=sum(r['pixel_eligible'] and r['protocol_b_image_label'] == 1 for r in category_views),
                unknown_views=len(category_views) - len(visible), **pixel.summary())
            category_rows.append(result)
            pixel.save(evaluation / 'pixel_statistics' / (category + '.npz'))
            print(json.dumps(result, allow_nan=False), flush=True)
    annotated_rows, errors = analyze_errors(view_rows, thresholds)
    if errors['overall']['ng_without_mask_views'] != cfg['expected_ng_without_masks']:
        raise RuntimeError('Missing-mask error bucket lost Protocol B test views')
    write_csv(evaluation / 'view_scores.csv', annotated_rows, list(annotated_rows[0]))
    write_csv(evaluation / 'calibration_scores.csv', calibration_rows, list(calibration_rows[0]))
    write_json(evaluation / 'false_alarm_thresholds.json', thresholds)
    write_json(evaluation / 'error_analysis.json', errors)
    for bucket in ('true_normal', 'ng_without_mask'):
        alarm_rows = sorted((r for r in annotated_rows if r['error_bucket'] == bucket and r['alarm']), key=lambda r: -r['score'])
        write_csv(evaluation / (bucket + '_false_alarms.csv'), alarm_rows, list(annotated_rows[0]))
    write_csv(evaluation / 'object_scores.csv', object_rows, list(object_rows[0]))
    write_csv(evaluation / 'category_metrics.csv', category_rows, list(category_rows[0]))
    write_json(evaluation / 'category_metrics.json', category_rows)
    artifact_hashes = {p.name: sha256(p) for p in evaluation.iterdir() if p.is_file()}
    write_json(evaluation / 'COMPLETE.json', dict(completed_at=now(), model=args.model, seed=args.seed,
        prepared_sha256=sha256(ROOT / 'prepared/COMPLETE.json'),
        checkpoint_sha256=completion['model_sha256'], artifacts=artifact_hashes,
        elapsed_seconds=time.monotonic() - started,
        scoring_protocol='B', protocol_b_list_sha256=sha256(cfg['protocol_b_null_mask_list']),
        ng_without_mask_test_views=cfg['expected_ng_without_masks'],
        pixel_metric_method='4096 fixed bins over [0,2]; 8-connected regions; PRO FPR <=0.3',
        score_method='mean of top floor(1%) map pixels; object pools all five maps; no per-image normalization'))


if __name__ == '__main__':
    main()
