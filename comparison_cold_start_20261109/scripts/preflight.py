"""MANUAL stage: frozen inputs, dependencies, disk, and device checks. No forward pass."""
import importlib
import platform
import shutil
import subprocess
import sys

from common import ROOT, check_lock, config, now, protocol_b_ids, write_json

# The comparison uses its own PNG cache loader. natsort/tifffile belong to
# unused Dinomaly2 video/TIFF dataset classes and are not runtime requirements.
REQUIRED_MODULES = ('torch', 'torchvision', 'timm', 'numpy', 'scipy', 'sklearn',
                    'skimage', 'PIL', 'cv2', 'pandas', 'matplotlib')


def main():
    lock_hash = check_lock()
    cfg = config()
    protocol_b_ids(cfg)
    if cfg['scoring_protocol'] != 'B' or not 0 < cfg['false_alarm_target_fpr'] < 1:
        raise ValueError('Expected Protocol B and a valid source-OK false-alarm calibration target')
    if not 0 < cfg['normal_fraction'] < 1 or cfg['support_objects_per_category'] != 1:
        raise ValueError('This frozen protocol requires a normal fraction in (0,1) and one support object')
    if cfg['image_size'] % 14 or cfg['map_size'] < 1 or cfg['total_steps'] <= 100:
        raise ValueError('Invalid geometry or step budget')
    usage = shutil.disk_usage(ROOT)
    if usage.free < cfg['minimum_free_gib'] * 1024 ** 3:
        raise RuntimeError('Need at least %d GiB free for cache/checkpoints' % cfg['minimum_free_gib'])
    versions = {}
    for module in REQUIRED_MODULES:
        try:
            imported = importlib.import_module(module)
        except ImportError as error:
            raise RuntimeError('Required comparison dependency failed in %s: %s (%s)' %
                               (sys.executable, module, error)) from error
        versions[module] = getattr(imported, '__version__', 'unknown')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in selected interpreter')
    inventory = {'created_at': now(), 'python': sys.executable, 'python_version': platform.python_version(),
        'packages': versions, 'cuda_runtime': torch.version.cuda, 'free_disk_gib': usage.free / 1024 ** 3,
        'gpus': [dict(index=i, name=torch.cuda.get_device_name(i),
                       memory_gib=torch.cuda.get_device_properties(i).total_memory / 1024 ** 3)
                 for i in range(torch.cuda.device_count())], 'input_lock_sha256': lock_hash}
    # Import each repository in its own interpreter to avoid same-name module collisions.
    for model in ('arcad', 'dinomaly2'):
        code = ('import sys, json; sys.path.insert(0, sys.argv[1]); '
                'from common import config; from model_adapter import validate_repository_imports; '
                'print(json.dumps(validate_repository_imports(sys.argv[2], config())))')
        subprocess.run([sys.executable, '-c', code, str(ROOT / 'scripts'), model], check=True, cwd=ROOT)
    stamp = now().replace(':', '').replace('+', '_')
    write_json(ROOT / 'logs' / ('preflight_' + stamp + '.json'), inventory)
    print('Preflight passed. No dataset preparation, model forward, training, or evaluation was run.')


if __name__ == '__main__':
    main()
