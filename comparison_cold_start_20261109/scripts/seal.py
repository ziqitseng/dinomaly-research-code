"""Setup utility: snapshot audit evidence and hash frozen implementation/input files.

No image decoding, split generation, model loading, or training. Refuses to reseal.
"""
import shutil
from pathlib import Path

from common import ROOT, config, now, sha256, write_json

AUDIT_FILES = ['REPORT.md', 'structure_summary.json', 'summary.json', 'supplemental_summary.json',
               'manifest_inventory.csv', 'file_checks.jsonl', 'blank_or_nearly_uniform_images.csv',
               'dimension_mismatches.csv', 'duplicate_images.csv', 'pixel_issues.csv',
               'run_impact_20261107/REPORT.md', 'run_impact_20261107/summary.json']


def main():
    target = ROOT / 'provenance/input_lock.json'
    if target.exists():
        raise RuntimeError('Already sealed. Preserve this experiment; fork a new folder to change protocol.')
    cfg = config()
    audit = Path(cfg['audit_root'])
    snapshot = ROOT / 'provenance/audit_snapshot'
    snapshot.mkdir(exist_ok=True)
    files = []
    for relative in AUDIT_FILES:
        path = audit / relative
        files.append({'path': str(path), 'sha256': sha256(path), 'role': 'external_audit'})
        # Keep large per-file inventories referenced and hashed, rather than copying 225 MB.
        if relative not in ('manifest_inventory.csv', 'file_checks.jsonl'):
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    for key in ('arcad_repo', 'dinomaly2_repo'):
        repo = Path(cfg[key])
        # Freeze every source file used directly or indirectly, excluding prior results.
        for folder in ('models', 'dinov1', 'dinov2', 'dinov3', 'beit', 'tips', 'optimizers', 'adeval'):
            if (repo / folder).is_dir():
                for path in sorted((repo / folder).rglob('*.py')):
                    files.append({'path': str(path), 'sha256': sha256(path), 'role': key})
        for path in sorted(repo.glob('*.py')):
            files.append({'path': str(path), 'sha256': sha256(path), 'role': key})
    path = Path(cfg['backbone_file'])
    files.append({'path': str(path), 'sha256': sha256(path), 'role': 'shared_pretrained_backbone'})
    # Freeze local implementation, protocol, and evidence. Later results are deliberately outside the lock.
    paths = list((ROOT / 'scripts').glob('*.py')) + list((ROOT / 'tests').glob('*.py'))
    paths += [p for p in (ROOT / 'config').iterdir() if p.is_file()] + list(ROOT.glob('*.md')) + [ROOT / 'run.sh']
    paths += [p for p in snapshot.rglob('*') if p.is_file()]
    # The synthetic regression test compares against the exact pre-optimization code.
    reference = ROOT / 'provenance/revisions/before_fast_prepare'
    if reference.exists():
        paths += [reference / 'scripts/prepare.py', reference / 'input_lock.json']
    for path in sorted(paths):
        files.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path), 'role': 'experiment_setup'})
    write_json(target, {'created_at': now(), 'files': files,
        'scope': 'code, protocol, config, shared backbone, and audit metadata; source payload verified at prepare'})
    print('Sealed %d files; no experiment stages executed.' % len(files))


if __name__ == '__main__':
    main()
