import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from checkpointing import learning_rate, save_milestone
from common import read_json, sha256
from evaluation_setup import resolve_checkpoint, selected_groups
from validation_manifest import propose


class InfrastructureTests(unittest.TestCase):
    def test_schedules_and_config_scope(self):
        base = read_json(ROOT / 'config/experiment_100k.json')
        short = read_json(ROOT / 'config/experiment_50k.json')
        self.assertEqual({k: v for k, v in base.items() if k not in ('total_steps', 'milestone_updates')},
                         {k: v for k, v in short.items() if k not in ('total_steps', 'milestone_updates')})
        self.assertEqual(base['milestone_updates'], [20000, 40000, 50000, 70000, 100000])
        self.assertEqual(short['milestone_updates'], [20000, 40000, 50000])
        self.assertAlmostEqual(learning_rate(99, 100000), .002)
        self.assertAlmostEqual(learning_rate(49999, 100000), .00110143, places=7)
        self.assertAlmostEqual(learning_rate(49999, 50000), .0002)
        self.assertAlmostEqual(learning_rate(99999, 100000), .0002)

    def test_immutable_checkpoint_and_metadata(self):
        import torch
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / 'milestone_020000.pth'
            state = dict(step=20000, model={'weight': torch.tensor([1., 2.])},
                         optimizer={'state': {}}, scheduler={'completed_steps': 20000},
                         python_rng=(1, 2, 3))
            self.assertTrue(save_milestone(path, state))
            digest = sha256(path)
            from model_adapter import torch_load
            self.assertEqual(torch_load(path, 'cpu')['step'], 20000)
            self.assertFalse(save_milestone(path, state))
            self.assertEqual(sha256(path), digest)
            with self.assertRaisesRegex(RuntimeError, 'differs'):
                save_milestone(path, dict(state, model={'weight': torch.tensor([9.])}))
            self.assertEqual(sha256(path), digest)
            with self.assertRaises(ValueError):
                save_milestone(Path(directory) / 'milestone_040000.pth', state)

    def test_arbitrary_checkpoint_identity_and_isolation(self):
        import torch
        cfg = read_json(ROOT / 'config/experiment_50k.json')
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / 'milestone_020000.pth'
            output = directory / 'evaluation'
            state = dict(model={'weight': torch.tensor([1.])}, optimizer={}, step=20000,
                         model_name='dinomaly2', seed=1, config_sha256='config',
                         prepared_sha256='prepared', normal_stream={}, python_rng=(),
                         numpy_rng=(), torch_rng=torch.tensor([1]), cuda_rng=[],
                         scheduler={'completed_steps': 20000, 'total_steps': 50000})
            torch.save(state, path)
            with patch('evaluation_setup.sha256', side_effect=lambda p: 'config' if 'experiment_' in str(p) else 'prepared' if 'COMPLETE' in str(p) else 'checkpoint'):
                got = resolve_checkpoint(path, 'dinomaly2', 1, cfg)
                self.assertEqual(got[2], 20000)
                self.assertEqual(got[0], path)
                with self.assertRaisesRegex(RuntimeError, 'model/seed'):
                    resolve_checkpoint(path, 'arcad', 1, cfg)
            self.assertFalse(output.exists())

    def test_manifest_selection_protects_original_test(self):
        records = []
        for category in range(159):
            for label in (0, 1):
                for number in range(2):
                    group = f'category_{category}/{label}/{number}'
                    for camera in range(1, 6):
                        records.append(dict(group=group, category=f'category_{category}',
                            partition='test', object_label=label, camera=camera,
                            protected_test_group=False, is_ng_without_mask=False,
                            exclude_reasons=[], source_split='test', pixel_eligible=True,
                            raw_mask='mask' if label else None,
                            mask_foreground_raw=1 if label else 0, id=f'{group}/{camera}'))
        proposal = propose(records, 'prepared', per_class=1)
        self.assertEqual((proposal['validation_objects'], proposal['final_test_objects']), (318, 318))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selection.json'
            path.write_text(json.dumps(proposal))
            with patch('evaluation_setup.sha256', return_value='prepared'):
                selected, _ = selected_groups(records, set(), 'validation', path)
                self.assertEqual(len(selected), 318)
                selected, _ = selected_groups(records, set(), 'test', path)
                self.assertEqual(len(selected), 318)


if __name__ == '__main__':
    unittest.main()
