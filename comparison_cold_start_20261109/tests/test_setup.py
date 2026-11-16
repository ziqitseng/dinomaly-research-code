"""CPU-only synthetic verification. Never reads real data or imports either model."""
import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import group_id, protocol_b_label
from prepare import assign_partitions, prepare
from train import BatchStream


class AdapterCompatibilityTests(unittest.TestCase):
    def test_arcad_deterministic_selection_matches_native_values_gradients_rng(self):
        from check_deterministic_selection import check_selection
        report = check_selection('cpu')
        self.assertTrue(report['strict_replacement_passed'])
        self.assertEqual(len(report['cases']), 4)

    def test_encoder_loader_calls_match_both_repository_apis(self):
        from types import SimpleNamespace
        from model_adapter import encoder_load_arguments
        cfg = dict(backbone_file='/example/weights/backbone.pth')
        def arcad_load(name):
            return name
        def dinomaly2_load(name, WEIGHTS_DIR='./backbones/weights'):
            return name, WEIGHTS_DIR
        arcad = SimpleNamespace(load=arcad_load)
        dinomaly2 = SimpleNamespace(load=dinomaly2_load)
        self.assertEqual(encoder_load_arguments(arcad, cfg), {})
        self.assertEqual(encoder_load_arguments(dinomaly2, cfg), {'WEIGHTS_DIR': '/example/weights'})
        self.assertEqual(arcad.load('dinov2reg_vit_base_14', **encoder_load_arguments(arcad, cfg)), 'dinov2reg_vit_base_14')
        self.assertEqual(dinomaly2.load('dinov2reg_vit_base_14', **encoder_load_arguments(dinomaly2, cfg))[1], '/example/weights')

    def test_unused_video_and_tiff_dependencies_not_required(self):
        from preflight import REQUIRED_MODULES
        self.assertNotIn('natsort', REQUIRED_MODULES)
        self.assertNotIn('tifffile', REQUIRED_MODULES)


def synthetic_records():
    result = []
    for category in ('alpha', 'beta'):
        for split, label, count in [('train', 0, 10), ('test', 0, 3), ('test', 1, 3)]:
            for sample in range(count):
                group = '%s/%s/S%04d' % (category, 'NG/Pit' if label else 'OK', sample + (100 if split == 'test' else 0))
                for camera in range(1, 6):
                    result.append(dict(category=category, source_split=split, object_label=label,
                        id=group + '/image_C%d.png' % camera, is_ng_without_mask=False,
                        group=group, camera=camera, exclude_reasons=[], partition='excluded',
                        support_eligible=bool(label), visible_eligible=True, pixel_eligible=True))
    return result


class SplitTests(unittest.TestCase):
    def test_group_key_preserves_components(self):
        self.assertEqual(group_id('thing/NG/Pit/S0001/file.png'), 'thing/NG/Pit/S0001')
        self.assertNotEqual(group_id('ab/c/file.png'), group_id('a/bc/file.png'))

    def test_matched_budget_and_whole_objects(self):
        records = synthetic_records()
        config = dict(normal_fraction=0.3, split_seed=1, support_objects_per_category=1)
        summary = assign_partitions(records, config)
        self.assertEqual([r['normal_train_objects'] for r in summary], [3, 3])
        for group in {r['group'] for r in records}:
            self.assertEqual(len({r['partition'] for r in records if r['group'] == group}), 1)
        self.assertEqual(sum(r['partition'] == 'arcad_support' for r in records), 10)
        self.assertTrue(all(r['source_split'] == 'test' for r in records if r['partition'] == 'test'))
        shuffled = list(reversed(synthetic_records()))
        assign_partitions(shuffled, config)
        self.assertEqual({(r['group'], r['partition']) for r in records},
                         {(r['group'], r['partition']) for r in shuffled})

    def test_batch_stream_independent_rng_and_resume(self):
        import random
        first, second = BatchStream(13, 4, 1), BatchStream(13, 4, 1)
        for i in range(7):
            random.random()
            self.assertEqual(first.next(), second.next())
        checkpoint = first.state_dict()
        resumed = BatchStream(13, 4, 99)
        resumed.load_state_dict(checkpoint)
        for i in range(9):
            self.assertEqual(first.next(), resumed.next())

    def test_null_mask_object_stays_test_even_if_quality_flagged(self):
        records = synthetic_records()
        target = next(r for r in records if r['category'] == 'alpha' and r['object_label'] == 1)
        target['is_ng_without_mask'] = True
        protected = target['group']
        for r in records:
            if r['group'] == protected:
                r['exclude_reasons'] = ['copied_or_blank_object']
        assign_partitions(records, dict(normal_fraction=.3, split_seed=1, support_objects_per_category=1))
        self.assertTrue(all(r['partition'] == 'test' for r in records if r['group'] == protected))
        self.assertTrue(all(r['group'] != protected for r in records if r['partition'] == 'arcad_support'))

    def test_protocol_b_uses_membership_not_zero_foreground(self):
        listed = {'null.png'}
        self.assertEqual(protocol_b_label(dict(id='null.png', object_label=1), listed), 0)
        self.assertEqual(protocol_b_label(dict(id='existing_empty_mask.png', object_label=1), listed), 1)
        self.assertEqual(protocol_b_label(dict(id='ok.png', object_label=0), listed), 0)


class ErrorAnalysisTests(unittest.TestCase):
    def test_threshold_refuses_ng_or_test_calibration(self):
        from error_analysis import fit_normal_threshold
        normals = [dict(object_label=0, is_ng_without_mask=False, partition='normal_calibration',
                        source_split='train', score=i / 100.) for i in range(100)]
        threshold = fit_normal_threshold(normals, .01)
        self.assertEqual(threshold['threshold'], .99)
        for bad in (dict(normals[0], object_label=1, is_ng_without_mask=True, score=1000),
                    dict(normals[0], partition='test')):
            with self.assertRaises(ValueError):
                fit_normal_threshold(normals + [bad], .01)

    def test_false_alarms_keep_annotation_bucket_separate(self):
        from error_analysis import analyze_errors
        base = dict(category='a', object_label=0, is_ng_without_mask=False, protocol_b_image_label=0)
        rs = [dict(base, score=.2), dict(base, score=.95),
              dict(base, object_label=1, is_ng_without_mask=True, score=.99),
              dict(base, object_label=1, protocol_b_image_label=1, score=.01)]
        annotated, summary = analyze_errors(rs, {'a': {'threshold': .9}})
        self.assertEqual(summary['overall']['true_normal_false_alarms'], 1)
        self.assertEqual(summary['overall']['ng_without_mask_alarms'], 1)
        self.assertEqual(summary['overall']['ng_without_mask_share_of_protocol_b_false_alarms'], .5)
        self.assertEqual(summary['overall']['ng_with_mask_misses'], 1)
        self.assertEqual(annotated[2]['error_bucket'], 'ng_without_mask')


class MetricTests(unittest.TestCase):
    def test_pixel_metrics_perfect_and_tied(self):
        import numpy as np
        from metrics import PixelMetrics
        mask = np.array([[0, 0], [0, 1]], dtype=bool)
        perfect = PixelMetrics(4096)
        perfect.update(mask, mask.astype(float))
        result = perfect.summary()
        self.assertAlmostEqual(result['P_AUROC'], 1)
        self.assertAlmostEqual(result['P_AP'], 1)
        self.assertAlmostEqual(result['P_AUPRO'], 1)
        tied = PixelMetrics(4096)
        tied.update(mask, np.ones((2, 2)))
        self.assertAlmostEqual(tied.summary()['P_AUROC'], 0.5)
        self.assertAlmostEqual(tied.summary()['P_AP'], 0.25)
        self.assertAlmostEqual(tied.summary()['P_AUPRO'], 0.15)

    def test_metric_matches_exact_at_separated_bins(self):
        import numpy as np
        from metrics import PixelMetrics, binary_metrics, top_score
        mask = np.array([[0, 1, 0], [1, 1, 0]], dtype=bool)
        scores = np.array([[0.1, 0.9, 0.8], [0.6, 0.3, 0.4]])
        pixel = PixelMetrics(4096)
        pixel.update(mask, scores)
        exact = binary_metrics(mask.ravel(), scores.ravel())
        self.assertAlmostEqual(pixel.summary()['P_AUROC'], exact['auroc'])
        self.assertAlmostEqual(pixel.summary()['P_AP'], exact['ap'])
        self.assertEqual(top_score(np.arange(10), .2), 8.5)
        self.assertIsNone(binary_metrics([0, 0], [0.1, 0.2])['auroc'])


class PreparationIntegrationTest(unittest.TestCase):
    def test_protocol_b_masks_test_reservation_and_quality_join(self):
        """Build a tiny fake audit and run full preparation against its synthetic pixels."""
        import numpy as np
        from PIL import Image
        generator = np.random.RandomState(55)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            raw, audit, out = root / 'raw', root / 'audit', root / 'prepared'
            raw.mkdir()
            audit.mkdir()
            out.mkdir()
            inventory, checks, issues, dimensions = [], [], [], []
            duplicate_pixels = None
            mode_pixels = {}
            # One retained category, plus a deliberately substituted excluded category.
            for category in ('widget', 'fuse_holder'):
                for split, cls, count in [('train', 'OK', 10), ('test', 'OK', 2), ('test', 'Pit', 4)]:
                    for sample in range(count):
                        sample_id = sample + (100 if split == 'test' else 0)
                        for camera in range(1, 6):
                            group = '%s/%s/S%04d' % (category, 'OK' if cls == 'OK' else 'NG/Pit', sample_id)
                            camera_name = '03' if camera == 3 else str(camera)
                            path = group + '/image_C%s_test.png' % camera_name
                            (raw / path).parent.mkdir(parents=True, exist_ok=True)
                            array = generator.randint(0, 255, (28, 28, 3), dtype=np.uint8)
                            # Distinct raw arrays collapse to the same decoded RGB input.
                            if category == 'widget' and cls == 'OK' and split == 'train' and sample in (8, 9):
                                if duplicate_pixels is None:
                                    duplicate_pixels = array.copy()
                                array = duplicate_pixels.copy()
                            if category == 'widget' and cls == 'OK' and split == 'train' and sample in (6, 7):
                                if sample == 6:
                                    mode_pixels[camera] = array.copy()
                                else:
                                    array = np.concatenate([mode_pixels[camera], np.full((28, 28, 1), 255, dtype=np.uint8)], axis=2)
                            if cls == 'OK' and sample == 0 and camera == 1:
                                array = array[:, :, 0]  # Grayscale must still produce identical RGB pixels.
                            image_format = 'JPEG' if cls == 'OK' and sample == 1 and camera == 1 else 'PNG'
                            Image.fromarray(array).save(raw / path, format=image_format)
                            with Image.open(raw / path) as encoded:
                                decoded = np.asarray(encoded)
                            mask_path = None
                            mask = np.zeros((28, 28), dtype=np.uint8)
                            if cls != 'OK' and (category == 'fuse_holder' or sample == 0 or (camera != 2 and sample != 3)):
                                mask_path = group + '/image_C%d_test_mask.png' % camera
                                if camera != 3:  # C3 is an explicitly empty annotation, also unknown.
                                    mask[6:20, 6:20] = 255
                                Image.fromarray(mask).save(raw / mask_path)
                            inventory.append(dict(category=category, split=split, anomaly_class=cls,
                                declared_image=path, actual_image=path, declared_mask=mask_path or '', actual_mask=mask_path or ''))
                            record = dict(path=path, pixel_sha256=hashlib.sha256(decoded.tobytes()).hexdigest(),
                                image_sha256=hashlib.sha256((raw / path).read_bytes()).hexdigest(),
                                image_shape=list(decoded.shape), mask_nonzero=int((mask > 0).sum()))
                            if mask_path:
                                record['mask_sha256'] = hashlib.sha256((raw / mask_path).read_bytes()).hexdigest()
                            checks.append(record)
                            if category == 'widget' and cls == 'Pit' and camera == 4:
                                issues.append(dict(issue='annotation_points_outside_canvas', path=path, detail='synthetic'))
            for filename, values, fields in [
                ('manifest_inventory.csv', inventory, list(inventory[0])),
                ('blank_or_nearly_uniform_images.csv', [{'path': 'widget/NG/Pit/S0103/image_C1_test.png'}], ['path']),
                ('dimension_mismatches.csv', dimensions, ['path']),
                ('pixel_issues.csv', issues, ['issue', 'path', 'detail'])]:
                with (audit / filename).open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(values)
            with (audit / 'file_checks.jsonl').open('w') as stream:
                for record in checks:
                    stream.write(json.dumps(record) + '\n')
            missing = [{'id': r['declared_image']} for r in inventory if r['anomaly_class'] != 'OK' and not r['actual_mask']]
            with (root / 'null_masks.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['id'])
                writer.writeheader()
                writer.writerows(missing)
            cfg = dict(audit_root=str(audit), raw_root=str(raw), image_size=28, map_size=14,
                protocol_b_null_mask_list=str(root / 'null_masks.csv'), expected_ng_without_masks=len(missing),
                exclude_categories=['fuse_holder'], normal_fraction=0.3, split_seed=1,
                support_objects_per_category=1, expected_source_views=len(inventory), expected_source_categories=2)
            # The archived serial implementation is the regression reference, not a live job.
            import importlib.util
            from common import read_json, rows, sha256
            archive = Path(__file__).resolve().parents[1] / 'provenance/revisions/before_fast_prepare'
            reference_path = archive / 'scripts/prepare.py'
            locked = next(e['sha256'] for e in read_json(archive / 'input_lock.json')['files'] if e['path'] == 'scripts/prepare.py')
            self.assertEqual(sha256(reference_path), locked)
            spec = importlib.util.spec_from_file_location('legacy_serial_prepare', reference_path)
            reference = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reference)
            legacy = root / 'legacy'
            legacy.mkdir()
            reference.prepare(cfg, legacy)
            prepare(cfg, out, workers=4, verify_written=True)
            records = [json.loads(line) for line in (out / 'records.jsonl').read_text().splitlines()]
            legacy_records = [json.loads(line) for line in (legacy / 'records.jsonl').read_text().splitlines()]
            self.assertEqual(records, legacy_records)
            for name in ('category_counts.csv', 'decisions.csv', 'protocol_b_ng_without_masks.csv'):
                self.assertEqual((out / name).read_bytes(), (legacy / name).read_bytes())
            for name in {r[key] for r in records for key in ('cache_image', 'cache_train_mask', 'cache_eval_mask') if key in r}:
                with Image.open(out / name) as actual, Image.open(legacy / name) as original:
                    self.assertTrue(np.array_equal(np.asarray(actual), np.asarray(original)))
            for entry in map(json.loads, (out / 'cache_checksums.jsonl').read_text().splitlines()):
                self.assertEqual(sha256(out / entry['path']), entry['sha256'])
            self.assertTrue(all(r['partition'] == 'excluded' for r in records if r['category'] == 'fuse_holder'))
            ng = [r for r in records if r['object_label'] and r['category'] == 'widget']
            self.assertTrue(all(r['object_label'] == 1 for r in ng))
            nulls = [r for r in ng if r['is_ng_without_mask']]
            self.assertEqual(len(nulls), len(missing))
            self.assertTrue(all(r['partition'] == 'test' and r['pixel_eligible']
                                and r['protocol_b_image_label'] == 0 and not r['support_eligible'] for r in nulls))
            self.assertTrue(all(not r['support_eligible'] for r in ng if r['protected_test_group']))
            for r in nulls:
                self.assertTrue(r['synthetic_empty_mask'])
                for key, size in [('cache_train_mask', 28), ('cache_eval_mask', 14)]:
                    self.assertTrue(r[key].endswith('_mask_EMPTY.png'))
                    with Image.open(out / r[key]) as mask:
                        self.assertEqual(mask.size, (size, size))
                        self.assertFalse(np.asarray(mask).any())
            empty_present = [r for r in ng if r['raw_mask'] and r['camera'] == 3]
            self.assertTrue(all(r['protocol_b_image_label'] == 1 and r['pixel_eligible'] for r in empty_present))
            unannotated = [r for r in ng if r['group'].endswith('S0103')]
            self.assertEqual(len(unannotated), 5)
            self.assertTrue(all(r['object_label'] == 1 and r['partition'] == 'test'
                                and not r['visible_eligible'] and r['pixel_eligible'] for r in unannotated))
            self.assertTrue(all('blank_image_in_object' in r['exclude_reasons'] for r in unannotated))
            self.assertIn('widget/NG/Pit/S0103', json.loads((out / 'summary.json').read_text())['object_only_test_groups'])
            repeated = [r for r in records if r['group'] in ('widget/OK/S0008', 'widget/OK/S0009')]
            self.assertTrue(all(r['partition'] == 'excluded' for r in repeated))
            processed_repeated = [r for r in records if r['group'] in ('widget/OK/S0006', 'widget/OK/S0007')]
            self.assertTrue(all('duplicate_model_input_pixels' in r['exclude_reasons'] for r in processed_repeated))
            self.assertTrue((out / 'COMPLETE.json').is_file())
            # A journal resume validates small cache bytes and source stats without decoding raw data.
            from unittest.mock import patch
            (out / 'COMPLETE.json').unlink()
            with patch('prepare_cache.audited_bytes', side_effect=AssertionError('Resume reread raw bytes')):
                prepare(cfg, out, workers=2, resume=True)
            runtime = read_json(out / 'preparation_runtime.json')
            self.assertGreater(runtime['resumed'], 0)
            self.assertEqual(runtime['decoded'], 0)
            self.assertEqual(read_json(out / 'summary.json')['ng_without_masks_in_test'], len(missing))
            # Recover a corrupted cache file by regenerating only its one source record.
            unique_normal = next(r for r in records if r['partition'] == 'normal_train')
            (out / unique_normal['cache_image']).write_bytes(b'interrupted cache write')
            (out / 'COMPLETE.json').unlink()
            prepare(cfg, out, workers=4, resume=True)
            runtime = read_json(out / 'preparation_runtime.json')
            self.assertEqual(runtime['decoded'], 1)
            self.assertEqual(runtime['repaired'], 1)
            self.assertEqual([json.loads(line) for line in (out / 'records.jsonl').read_text().splitlines()], records)
            # Adopt the legacy cache without touching any valid original PNG byte stream.
            before = {r['path']: (legacy / r['path']).stat().st_mtime_ns
                      for r in map(json.loads, (legacy / 'cache_checksums.jsonl').read_text().splitlines())}
            prepare(cfg, legacy, workers=4, resume=True)
            self.assertEqual(read_json(legacy / 'preparation_runtime.json')['written'], 0)
            self.assertEqual(before, {name: (legacy / name).stat().st_mtime_ns for name in before})
            # Actual mid-run failure still commits earlier results and never publishes COMPLETE.
            import prepare_cache
            interrupted = root / 'interrupted'
            interrupted.mkdir()
            original_process = prepare_cache.process_record
            def fail_last(record, *args, **kwargs):
                if record['id'] == 'widget/NG/Pit/S0103/image_C5_test.png':
                    raise RuntimeError('injected worker interruption')
                return original_process(record, *args, **kwargs)
            with patch('prepare_cache.process_record', side_effect=fail_last):
                with self.assertRaisesRegex(RuntimeError, 'injected worker interruption'):
                    prepare(cfg, interrupted, workers=4)
            self.assertFalse((interrupted / 'COMPLETE.json').exists())
            prepare(cfg, interrupted, workers=4, resume=True)
            self.assertGreater(read_json(interrupted / 'preparation_runtime.json')['resumed'], 0)
            self.assertEqual([json.loads(line) for line in (interrupted / 'records.jsonl').read_text().splitlines()], records)
            with self.assertRaisesRegex(RuntimeError, 'Resume recipe'):
                prepare(dict(cfg, image_size=42), interrupted, workers=4, resume=True)


class CacheSafetyTests(unittest.TestCase):
    def test_changed_raw_source_invalidates_resume_and_fails_audit(self):
        import numpy as np
        from PIL import Image
        from prepare_cache import process_record
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.png'
            Image.fromarray(np.arange(300, dtype=np.uint8).reshape(10, 10, 3)).save(source)
            record = dict(id='a/OK/S1/source.png', raw_image='source.png', raw_mask=None,
                annotation_issues=[], mask_foreground_raw=0, object_label=0,
                is_ng_without_mask=False, protected_test_group=False)
            expected = dict(image_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), image_shape=[10, 10, 3])
            cfg = dict(raw_root=str(root), image_size=28, map_size=14)
            output = root / 'cache_output'
            output.mkdir()
            result, _ = process_record(record, expected, cfg, output)
            source.write_bytes(source.read_bytes() + b'changed')
            with self.assertRaisesRegex(RuntimeError, 'audited bytes'):
                process_record(record, expected, cfg, output, saved=result)

    def test_legacy_prepare_process_is_detected_before_restart(self):
        from prepare_cache import other_prepare_pids
        script = Path(__file__).resolve().parents[1] / 'scripts/prepare.py'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            proc = root / '500001'
            proc.mkdir()
            (proc / 'cmdline').write_bytes(b'python\0scripts/prepare.py\0')
            (proc / 'cwd').symlink_to(script.parents[1], target_is_directory=True)
            self.assertEqual(other_prepare_pids(script, root), [500001])


class AggregationTests(unittest.TestCase):
    def test_protocol_b_report_requires_all_pairs_and_matches_list_labels(self):
        """Exercise final reporting only with invented predictions and completion files."""
        from contextlib import ExitStack
        from unittest.mock import patch
        import common
        import summarize
        from common import sha256, write_csv, write_json
        from error_analysis import analyze_errors
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'results').mkdir()
            (root / 'config').mkdir()
            (root / 'prepared').mkdir()
            records = []
            for group, label in [('a/OK/S1', 0), ('a/NG/Pit/S1', 1), ('a/NG/Pit/S2', 1)]:
                for camera in range(1, 6):
                    missing = group == 'a/NG/Pit/S2' and camera == 2
                    records.append(dict(id=group + '/image_C%d.png' % camera, group=group, category='a',
                        camera=camera, object_label=label, protocol_b_image_label=0 if missing else label,
                        is_ng_without_mask=missing, partition='test', visible_eligible=not missing,
                        visible_label=None if missing else label, pixel_eligible=True))
            listed = [{'id': r['id']} for r in records if r['is_ng_without_mask']]
            list_path = root / 'config/list.csv'
            write_csv(list_path, listed, ['id'])
            cfg = dict(training_seeds=[1, 2, 3], split_seed=1, total_steps=100000,
                bootstrap_replicates=10, protocol_b_null_mask_list=str(list_path), expected_ng_without_masks=1)
            write_json(root / 'config/experiment.json', cfg)
            write_json(root / 'prepared/COMPLETE.json', {'synthetic': True})
            write_json(root / 'prepared/summary.json', dict(protected_test_groups=1, protocol_b_retained_quality_flagged_views=0))
            for model in ('arcad', 'dinomaly2'):
                for seed in cfg['training_seeds']:
                    target = root / 'runs' / model / ('seed_%d' % seed)
                    ev = target / 'evaluation'
                    ev.mkdir(parents=True)
                    scored = [dict(r, score=.9 if r['object_label'] or r['camera'] == 1 else .1) for r in records]
                    scored, errors = analyze_errors(scored, {'a': {'threshold': .5}})
                    write_csv(ev / 'view_scores.csv', scored, list(scored[0]))
                    objects = [dict(group=group, label=label) for group, label in
                               sorted({(r['group'], r['object_label']) for r in records})]
                    write_csv(ev / 'object_scores.csv', objects, ['group', 'label'])
                    values = dict(category='a', **{metric: .9 if model == 'arcad' else .8 for metric in summarize.METRICS})
                    values.update(objects=3, normal_objects=1, defective_objects=2, views=15, visible_views=14,
                        ng_without_mask_views=1, ng_without_mask_background_pixels=65536, b_normal_views=6, b_anomaly_views=9,
                        visible_normal_views=5, visible_defective_views=9, pixel_b_normal_views=6,
                        pixel_b_anomaly_views=9, pixel_views=15, foreground_pixels=20, background_pixels=40, regions=2)
                    write_json(ev / 'category_metrics.json', [values])
                    write_json(ev / 'error_analysis.json', errors)
                    write_json(target / 'TRAIN_COMPLETE.json', dict(model=model, seed=seed, steps=100000,
                        model_sha256='synthetic_model', config_sha256=sha256(root / 'config/experiment.json')))
                    write_json(ev / 'COMPLETE.json', dict(model=model, seed=seed, scoring_protocol='B',
                        protocol_b_list_sha256=sha256(list_path), ng_without_mask_test_views=1,
                        prepared_sha256=sha256(root / 'prepared/COMPLETE.json'), checkpoint_sha256='synthetic_model',
                        artifacts={p.name: sha256(p) for p in ev.iterdir() if p.is_file()}))
            with ExitStack() as stack:
                stack.enter_context(patch.object(summarize, 'ROOT', root))
                stack.enter_context(patch.object(summarize, 'config', return_value=cfg))
                stack.enter_context(patch.object(summarize, 'check_prepared', return_value={}))
                stack.enter_context(patch.object(summarize, 'run_dir', side_effect=lambda model, seed: root / 'runs' / model / ('seed_%d' % seed)))
                stack.enter_context(patch.object(common, 'load_records', return_value=records))
                final = root / 'runs/dinomaly2/seed_3/evaluation/COMPLETE.json'
                saved = final.read_bytes()
                final.rename(final.with_suffix('.pending'))
                with self.assertRaises(FileNotFoundError):
                    summarize.main()
                self.assertFalse((root / 'results/COMPLETE.json').exists())
                final.write_bytes(saved)
                # A protocol-label error cannot be hidden by updating that CSV's checksum.
                ev = root / 'runs/arcad/seed_1/evaluation'
                score_path = ev / 'view_scores.csv'
                original = score_path.read_bytes()
                bad_rows = list(common.rows(score_path))
                next(r for r in bad_rows if r['is_ng_without_mask'] == 'True')['protocol_b_image_label'] = '1'
                write_csv(score_path, bad_rows, list(bad_rows[0]))
                complete = common.read_json(ev / 'COMPLETE.json')
                complete['artifacts']['view_scores.csv'] = sha256(score_path)
                write_json(ev / 'COMPLETE.json', complete)
                with self.assertRaisesRegex(RuntimeError, 'identities/labels'):
                    summarize.main()
                score_path.write_bytes(original)
                complete['artifacts']['view_scores.csv'] = sha256(score_path)
                write_json(ev / 'COMPLETE.json', complete)
                summarize.main()
                report = common.read_json(root / 'results/comparison.json')
                self.assertEqual(report['primary_metric'], 'I_B_AUROC')
                self.assertAlmostEqual(report['metrics'][0]['paired_mean_difference'], .1)
                self.assertEqual(report['ng_without_mask_test_views'], 1)
                self.assertIn('50.00%', (root / 'results/RESULTS.md').read_text())


if __name__ == '__main__':
    unittest.main()
