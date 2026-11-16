"""MANUAL stage: validate complete paired runs and write final comparison here."""
from common import ROOT, check_prepared, config, now, protocol_b_ids, protocol_b_label, read_json, rows, run_dir, sha256, write_csv, write_json

METRICS = ['I_B_AUROC', 'I_B_AP', 'P_AUROC', 'P_AP', 'P_AUPRO',
           'O_source_AUROC', 'O_source_AP', 'I_visible_AUROC', 'I_visible_AP', 'I_source_AUROC', 'I_source_AP']


def main():
    import numpy as np
    check_prepared()
    cfg = config()
    listed_ids = protocol_b_ids(cfg)
    prepared_hash = sha256(ROOT / 'prepared/COMPLETE.json')
    output = ROOT / 'results'
    if (output / 'COMPLETE.json').exists():
        raise RuntimeError('Final report already written; preserve it')
    datasets, identities, error_summaries = {}, {}, {}
    expected_views, expected_objects = [], []
    # Expected identities come from frozen manifests, independently of either model's output.
    from common import load_records
    group_labels = {}
    for r in load_records():
        if r['partition'] == 'test':
            expected_views.append((r['id'], r['group'], str(r['camera']), str(r['object_label']),
                                   str(protocol_b_label(r, listed_ids)), str(r['id'] in listed_ids),
                                   str(r['visible_eligible']), str(r.get('visible_label') if r.get('visible_label') is not None else ''),
                                   str(r['pixel_eligible'])))
            group_labels[r['group']] = str(r['object_label'])
    expected_objects = sorted(group_labels.items())
    expected_views = sorted(expected_views)
    if {r[0] for r in expected_views if r[0] in listed_ids} != listed_ids:
        raise RuntimeError('Shared test must include every listed Protocol B NG/no-mask view')
    for model in ('arcad', 'dinomaly2'):
        for seed in cfg['training_seeds']:
            directory = run_dir(model, seed) / 'evaluation'
            complete = read_json(directory / 'COMPLETE.json')
            if complete['model'] != model or complete['seed'] != seed:
                raise RuntimeError('Evaluation model/seed identity mismatch')
            if (complete['scoring_protocol'] != 'B'
                    or complete['protocol_b_list_sha256'] != sha256(cfg['protocol_b_null_mask_list'])
                    or complete['ng_without_mask_test_views'] != cfg['expected_ng_without_masks']):
                raise RuntimeError('Evaluation did not use the frozen Protocol B population/labels')
            if complete['prepared_sha256'] != prepared_hash:
                raise RuntimeError('Evaluation input fingerprint differs')
            training = read_json(directory.parent / 'TRAIN_COMPLETE.json')
            if training['model'] != model or training['seed'] != seed:
                raise RuntimeError('Training model/seed identity mismatch')
            if training['config_sha256'] != sha256(ROOT / 'config/experiment.json'):
                raise RuntimeError('Training configuration fingerprint differs')
            if training['steps'] != cfg['total_steps'] or complete['checkpoint_sha256'] != training['model_sha256']:
                raise RuntimeError('Wrong training budget/checkpoint')
            for name, digest in complete['artifacts'].items():
                if sha256(directory / name) != digest:
                    raise RuntimeError('Evaluation artifact changed: ' + str(directory / name))
            view_identity = sorted((r['id'], r['group'], r['camera'], r['object_label'],
                                    r['protocol_b_image_label'], r['is_ng_without_mask'],
                                    r['visible_eligible'], r['visible_label'], r['pixel_eligible'])
                                   for r in rows(directory / 'view_scores.csv'))
            object_identity = sorted((r['group'], r['label']) for r in rows(directory / 'object_scores.csv'))
            if view_identity != expected_views or object_identity != expected_objects:
                raise RuntimeError('Evaluation identities/labels differ from frozen shared test set')
            values = read_json(directory / 'category_metrics.json')
            datasets[model, seed] = {r['category']: r for r in values}
            identities[model + '/seed_%d' % seed] = sha256(directory / 'COMPLETE.json')
            error_summaries[model + '/seed_%d' % seed] = read_json(directory / 'error_analysis.json')
    categories = sorted(next(iter(datasets.values())))
    if any(sorted(value) != categories for value in datasets.values()):
        raise RuntimeError('Category membership differs between runs')
    # Pixel denominator equality is required even though predictions differ.
    reference = next(iter(datasets.values()))
    for values in datasets.values():
        for category in categories:
            for key in ('objects', 'normal_objects', 'defective_objects', 'views', 'visible_views',
                        'ng_without_mask_views', 'ng_without_mask_background_pixels', 'b_normal_views', 'b_anomaly_views',
                        'visible_normal_views', 'visible_defective_views', 'pixel_b_normal_views', 'pixel_b_anomaly_views',
                        'pixel_views', 'foreground_pixels', 'background_pixels', 'regions'):
                if values[category][key] != reference[category][key]:
                    raise RuntimeError('Metric denominator differs: %s/%s' % (category, key))
    per_seed, differences, aggregate = [], [], []
    generator = np.random.RandomState(20261109)
    for metric in METRICS:
        included = [c for c in categories if all(data[c][metric] is not None for data in datasets.values())]
        skipped = sorted(set(categories) - set(included))
        if not included:
            aggregate.append(dict(metric=metric, categories=0, skipped_categories=skipped))
            continue
        method_means = {}
        for model in ('arcad', 'dinomaly2'):
            values = []
            for seed in cfg['training_seeds']:
                mean = float(np.mean([datasets[model, seed][c][metric] for c in included]))
                values.append(mean)
                per_seed.append(dict(model=model, seed=seed, metric=metric, macro=mean, categories=len(included)))
            method_means[model] = values
        category_deltas = []
        for category in included:
            delta = float(np.mean([datasets['arcad', seed][category][metric] -
                                  datasets['dinomaly2', seed][category][metric] for seed in cfg['training_seeds']]))
            category_deltas.append(delta)
            differences.append(dict(category=category, metric=metric, arcad_minus_dinomaly2=delta))
        delta_array = np.asarray(category_deltas)
        draws = generator.randint(0, len(included), size=(cfg['bootstrap_replicates'], len(included)))
        bootstrap = delta_array[draws].mean(axis=1)
        paired_seed_delta = np.asarray(method_means['arcad']) - np.asarray(method_means['dinomaly2'])
        aggregate.append(dict(metric=metric, categories=len(included), skipped_categories=skipped,
            arcad_mean=float(np.mean(method_means['arcad'])),
            arcad_seed_std=float(np.std(method_means['arcad'], ddof=1)),
            dinomaly2_mean=float(np.mean(method_means['dinomaly2'])),
            dinomaly2_seed_std=float(np.std(method_means['dinomaly2'], ddof=1)),
            paired_mean_difference=float(paired_seed_delta.mean()),
            paired_seed_differences=paired_seed_delta.tolist(),
            category_bootstrap_95ci=np.quantile(bootstrap, [0.025, 0.975]).tolist()))
    write_csv(output / 'per_seed_macro.csv', per_seed, ['model', 'seed', 'metric', 'macro', 'categories'])
    write_csv(output / 'paired_category_differences.csv', differences,
              ['category', 'metric', 'arcad_minus_dinomaly2'])
    summary = dict(completed_at=now(), status='COMPLETE', categories=len(categories),
        scoring_protocol='B', primary_metric='I_B_AUROC',
        protocol_b_list_sha256=sha256(cfg['protocol_b_null_mask_list']),
        ng_without_mask_test_views=cfg['expected_ng_without_masks'],
        false_alarm_analysis=error_summaries,
        training_seeds=cfg['training_seeds'], split_seed=cfg['split_seed'], metrics=aggregate,
        prepared_sha256=prepared_hash, evaluation_fingerprints=identities,
        dataset_summary=read_json(ROOT / 'prepared/summary.json'),
        confidence_interval_scope='paired category bootstrap of training-seed-averaged differences; conditional on one fixed split; not physical-object independence')
    write_json(output / 'comparison.json', summary)
    lines = ['# Matched cold-start comparison — Protocol B — completed', '',
        'Both methods used the same 30% per-category reviewed normal training subset and the same held-out objects. '
        'ArcAD additionally used one defective support object per retained category; those objects were excluded '
        'from both test sets. All numbers below are percentages; differences are percentage points.', '',
        '**Primary image protocol B:** all {:,} listed source-NG/no-mask views remain in test, '.format(cfg['expected_ng_without_masks']) +
        'have image target 0, and contribute synthetic zero-mask pixels. Source NG identity is preserved separately. '
        'Compare external scores only after matching both labeling protocol and population/preprocessing.', '',
        '| Metric | Categories | ArcAD mean ± seed SD | Dinomaly2 mean ± seed SD | ArcAD − Dinomaly2 | Paired category bootstrap 95% CI |',
        '|---|---:|---:|---:|---:|---:|']
    for r in aggregate:
        if not r['categories']:
            lines.append('| %s | 0 | unavailable | unavailable | — | — |' % r['metric'])
            continue
        lines.append('| %s | %d | %.3f ± %.3f | %.3f ± %.3f | %+.3f | [%+.3f, %+.3f] |' % (
            r['metric'], r['categories'], 100 * r['arcad_mean'], 100 * r['arcad_seed_std'],
            100 * r['dinomaly2_mean'], 100 * r['dinomaly2_seed_std'], 100 * r['paired_mean_difference'],
            100 * r['category_bootstrap_95ci'][0], 100 * r['category_bootstrap_95ci'][1]))
    lines += ['', '| Run | Source-OK false alarms / views | NG/no-mask alarms / views | NG/no-mask share of B false alarms |',
              '|---|---:|---:|---:|']
    for name, errors in sorted(error_summaries.items()):
        e = errors['overall']
        share = e['ng_without_mask_share_of_protocol_b_false_alarms']
        lines.append('| %s | %d / %d | %d / %d | %s |' % (name,
            e['true_normal_false_alarms'], e['true_normal_views'], e['ng_without_mask_alarms'],
            e['ng_without_mask_views'], 'undefined (no false alarms)' if share is None else '%.2f%%' % (share * 100)))
    dataset_summary = summary['dataset_summary']
    lines += ['', 'Test coverage: %d categories; %d protected five-view groups; %d retained test views carry '
              'quality-exclusion flags overridden by mandatory Protocol B retention. Flags remain visible; '
              'this is not a fully repaired official benchmark.' % (len(categories),
                  dataset_summary['protected_test_groups'], dataset_summary['protocol_b_retained_quality_flagged_views']), '',
              'I_B is the main image result. O_source metrics are supplementary and use original object labels and all five views. I_visible metrics omit annotation-unknown NG views. '
              'I_source metrics broadcast the source object label to views and are sensitivity results. '
              'Pixel metrics are histogram approximations (4096 fixed bins): missing masks are generated zero targets, '
              'valid present empty masks stay zero; misaligned, geometry-flagged, and vanished-foreground supplied masks are excluded.', '',
              'False-alarm thresholds use only reviewed source-OK calibration images held out from training. '
              'Source-OK false alarms and NG/no-mask Protocol B false alarms are separate buckets. '
              'The NG/no-mask share is measured from this run, never set to a prior example percentage. '
              'Inspect false_alarm_analysis in comparison.json and each evaluation/error_analysis.json before attributing those alarms to model errors.', '',
              'The interval resamples categories after averaging training seeds. It does not include split-selection uncertainty '
              'or prove physical-object independence. Seed SD describes initialization/training variation on the fixed split. '
              'No combined detection/localization ranking or test-tuned threshold is used.', '',
              'See comparison.json for exclusions, denominator coverage, fingerprints, skipped metric categories, and paired seed differences. '
              'See ../DATASET_POLICY.md and ../PROTOCOL.md for the preregistered interpretation.', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines))
    write_json(output / 'COMPLETE.json', dict(completed_at=now(),
        artifacts={p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name != 'STATUS.json'}))
    write_json(output / 'STATUS.json', dict(status='COMPLETE', completed_at=now(),
        comparison_sha256=sha256(output / 'comparison.json')))
    print('Final paired comparison written to ' + str(output / 'RESULTS.md'))


if __name__ == '__main__':
    main()
