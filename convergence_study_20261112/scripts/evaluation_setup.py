"""CPU-only checkpoint and optional validation-manifest checks."""
import collections
from pathlib import Path

from common import PREPARED, config_path, read_json, sha256


def prepare_output_dir(path, save_maps=False):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    (path / 'pixel_statistics').mkdir()
    if save_maps:
        (path / 'maps').mkdir()
    return path


def resolve_checkpoint(path, model, seed, cfg, default_run=None):
    from model_adapter import torch_load
    explicit = path is not None
    if explicit:
        path = Path(path).expanduser().resolve(strict=True)
        state = torch_load(path, 'cpu')
        required = {'model', 'optimizer', 'step', 'model_name', 'seed', 'config_sha256',
                    'prepared_sha256', 'normal_stream', 'python_rng', 'numpy_rng',
                    'torch_rng', 'cuda_rng'}
        if not isinstance(state, dict) or not required <= state.keys():
            raise RuntimeError('Explicit checkpoint must contain full resumable training state')
        if (state['model_name'], state['seed']) != (model, seed):
            raise RuntimeError('Checkpoint model/seed mismatch')
        if not 1 <= state['step'] <= cfg['total_steps']:
            raise RuntimeError('Checkpoint iteration outside configured schedule')
        if path.name.startswith('milestone_') and path.name != 'milestone_%06d.pth' % state['step']:
            raise RuntimeError('Checkpoint filename and completed iteration disagree')
        if state['config_sha256'] != sha256(config_path()):
            raise RuntimeError('Checkpoint training configuration mismatch')
        if state['prepared_sha256'] != sha256(PREPARED / 'COMPLETE.json'):
            raise RuntimeError('Checkpoint prepared-data fingerprint mismatch')
        if 'scheduler' in state and (state['scheduler']['completed_steps'] != state['step']
                or state['scheduler']['total_steps'] != cfg['total_steps']):
            raise RuntimeError('Checkpoint scheduler metadata mismatch')
        if model == 'arcad' and not {'learner', 'discriminator', 'support_stream'} <= state.keys():
            raise RuntimeError('ArcAD full training state incomplete')
        return path, state['model'], state['step'], sha256(path)
    if default_run is None:
        raise ValueError('default_run required for default benchmark evaluation')
    completion = read_json(default_run / 'TRAIN_COMPLETE.json')
    if (completion['model'], completion['seed'], completion['steps']) != (model, seed, cfg['total_steps']):
        raise RuntimeError('Completed training identity or budget mismatch')
    if completion['config_sha256'] != sha256(config_path()):
        raise RuntimeError('Completed training configuration mismatch')
    if completion['prepared_sha256'] != sha256(PREPARED / 'COMPLETE.json'):
        raise RuntimeError('Completed training prepared-data mismatch')
    path = default_run / 'model.pth'
    if completion['model_sha256'] != sha256(path):
        raise RuntimeError('Completed model checkpoint fingerprint mismatch')
    return path, torch_load(path, 'cpu'), completion['steps'], completion['model_sha256']


def selected_groups(records, listed_ids, split, manifest_path=None):
    groups = collections.defaultdict(list)
    for record in records:
        if record['partition'] == 'test':
            groups[record['group']].append(record)
    if manifest_path is None:
        if split != 'test':
            raise ValueError('Validation requires an explicit selection manifest')
        return groups, None
    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = read_json(manifest_path)
    if manifest.get('prepared_sha256') != sha256(PREPARED / 'COMPLETE.json'):
        raise RuntimeError('Validation manifest prepared-data fingerprint mismatch')
    validation = manifest.get('validation_groups')
    final_test = manifest.get('test_groups')
    if (not isinstance(validation, list) or not isinstance(final_test, list)
            or len(validation) != len(set(validation)) or len(final_test) != len(set(final_test))
            or set(validation) & set(final_test) or set(validation) | set(final_test) != set(groups)):
        raise RuntimeError('Manifest must partition all original test objects exactly once')
    protected = {g for g, rs in groups.items() if any(r['protected_test_group'] for r in rs)}
    if not protected <= set(final_test):
        raise RuntimeError('Protected missing-mask object moved out of final test')
    if {r['id'] for g in protected for r in groups[g] if r['is_ng_without_mask']} != listed_ids:
        raise RuntimeError('Protocol B protected views changed')
    chosen = set(validation if split == 'validation' else final_test)
    selected = {g: rs for g, rs in groups.items() if g in chosen}
    categories = {rs[0]['category'] for rs in groups.values()}
    if {rs[0]['category'] for rs in selected.values()} != categories:
        raise RuntimeError('Selected split loses category coverage')
    for category in categories:
        if {rs[0]['object_label'] for rs in selected.values() if rs[0]['category'] == category} != {0, 1}:
            raise RuntimeError('Selected split loses a class: ' + category)
    return selected, sha256(manifest_path)
