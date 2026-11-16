"""Evaluate published weights, train CostFilter for 17 epochs, then compare.

Runs the repository's configurable entry point without replacing its model,
losses, training recipe, anomaly synthesis, or evaluation implementation.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'costfilter-ad/Costfilter_Dinomaly'
SCRIPT = MODEL / 'costfilter_dinomaly_mvtec_uni_my_train_test.py'
OFFICIAL = MODEL / 'checkpoint_paths/mvtecad0.0001/costfilter_dinaomaly_mvtecad.pth'
METRICS = ['I_AUROC', 'I_AP', 'I_F1max', 'P_AUROC', 'P_AP', 'P_F1max', 'AUPRO']
CATEGORIES = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
              'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=Path, default=ROOT.parent / 'dataset/MVTec-AD/mvtec_anomaly_detection')
    parser.add_argument('--dtd-path', type=Path, default=ROOT / 'datasets/dtd/images')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'costfilter_setup/runs/mvtec_reproduction_20261113')
    parser.add_argument('--resume', action='store_true', help='Keep completed stages and retry a failed stage')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_file = output / 'status.json'
    if status_file.exists() and not args.resume:
        raise FileExistsError('Use a fresh output directory; this run already has a status file.')
    official_digest = sha256(OFFICIAL)
    counts = {}
    for category in CATEGORIES:
        directory = args.data_path / category
        train = list((directory / 'train/good').glob('*.png'))
        test = list((directory / 'test').glob('*/*.png'))
        if not train or not test:
            raise RuntimeError('Missing MVTec category: ' + category)
        counts[category] = {'train': len(train), 'test': len(test)}
    status = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'complete': False,
              'dataset': str(args.data_path.resolve()), 'dtd_images': str(args.dtd_path.resolve()),
              'counts': counts, 'official_sha256_before': official_digest, 'stages': []}
    if status_file.exists():
        status = json.loads(status_file.read_text())
        if status['dataset'] != str(args.data_path.resolve()) or status['official_sha256_before'] != official_digest:
            raise RuntimeError('Dataset or official checkpoint changed; start a fresh run.')
        status.pop('error', None)
        status.setdefault('resumed_at_utc', []).append(datetime.now(timezone.utc).isoformat())
    def save_status():
        temporary = status_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(status_file)
    save_status()
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                       MPLCONFIGDIR=str(ROOT / '.conda/tmp/matplotlib'),
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    common = [sys.executable, '-B', str(SCRIPT), '--data_path', str(args.data_path.resolve()),
              '--device', 'cuda:0', '--batch_size', '12', '--seed', '1',
              '--num_workers', '4', '--cpu_threads', '4']
    def run_stage(name, options):
        directory = output / name
        completed = [entry for entry in status['stages'] if entry['name'] == name and entry['complete']]
        if completed:
            print('Keeping completed stage: ' + name, flush=True)
            return directory
        if directory.exists():
            attempt = 1
            while (output / ('%s_failed_attempt_%d' % (name, attempt))).exists():
                attempt += 1
            failed_directory = output / ('%s_failed_attempt_%d' % (name, attempt))
            directory.rename(failed_directory)
            for entry in status['stages']:
                if entry['name'] == name and not entry['complete']:
                    entry['preserved_log'] = str(failed_directory / 'console.log')
        directory.mkdir()
        command = common + ['--output_dir', str(directory)] + options
        record = {'name': name, 'command': command, 'complete': False,
                  'started_at_utc': datetime.now(timezone.utc).isoformat()}
        status['stages'].append(record)
        save_status()
        print('Starting ' + name, flush=True)
        start = time.monotonic()
        with (directory / 'console.log').open('w') as log:
            result = subprocess.run(command, cwd=MODEL, env=environment, stdout=log, stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode, seconds=time.monotonic() - start,
                      complete=result.returncode == 0)
        save_status()
        if result.returncode:
            raise RuntimeError('%s failed; see %s' % (name, directory / 'console.log'))
        print('Completed %s in %.1f seconds' % (name, record['seconds']), flush=True)
        return directory
    try:
        official_dir = run_stage('official', ['--costfilter_checkpoint', str(OFFICIAL)])
        textures = list(args.dtd_path.glob('*/*.jpg'))
        if len(textures) != 5640:
            raise RuntimeError('Expected all 5640 official DTD images; found %d' % len(textures))
        run_stage('smoke', ['--train', '--dtd_path', str(args.dtd_path.resolve()), '--epochs', '1', '--max_steps', '2'])
        trained_dir = run_stage('training', ['--train', '--dtd_path', str(args.dtd_path.resolve()), '--epochs', '17'])
        trained_checkpoint = trained_dir / 'epoch_16_costfilter_dinaomaly_my_mvtecad.pth'
        local_dir = run_stage('local', ['--costfilter_checkpoint', str(trained_checkpoint)])
        official = json.loads((official_dir / 'metrics.json').read_text())
        local = json.loads((local_dir / 'metrics.json').read_text())
        assert official['complete'] and local['complete']
        assert set(official['categories']) == set(local['categories']) == set(CATEGORIES)
        rows = []
        for category in CATEGORIES + ['mean']:
            a = official['mean'] if category == 'mean' else official['categories'][category]
            b = local['mean'] if category == 'mean' else local['categories'][category]
            for metric in METRICS:
                rows.append({'category': category, 'metric': metric,
                             'official_percent': 100 * a[metric], 'local_percent': 100 * b[metric],
                             'delta_percentage_points': 100 * (b[metric] - a[metric])})
        with (output / 'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = ['# MVTec-AD CostFilter training reproduction', '',
                  'Both evaluations use the same frozen Dinomaly checkpoint, all 15 categories,',
                  '448-pixel resize / 392-pixel crop, batch size 12, seed 1, and upstream metrics.',
                  'The local model is the final epoch of the released 17-epoch recipe; it was',
                  'not selected by its test score. Only normal MVTec training images and DTD',
                  'textures are used to train the CostFilter network.', '',
                  '| Metric | Official (%) | Local (%) | Local minus official (pp) |',
                  '| --- | ---: | ---: | ---: |']
        for row in rows:
            if row['category'] == 'mean':
                report.append('| {metric} | {official_percent:.4f} | {local_percent:.4f} | {delta_percentage_points:+.4f} |'.format(**row))
        reference = (MODEL / 'checkpoint_paths/mvtecad0.0001/log_dinomaly_my_mvtec.txt').read_text()
        match = re.search(r'Mean: I-Auroc:([\d.]+), I-AP:([\d.]+), I-F1:([\d.]+), P-AUROC:([\d.]+), P-AP:([\d.]+), P-F1:([\d.]+), P-AUPRO:([\d.]+)', reference)
        if match:
            report += ['', 'Official checkpoint versus the authors\' bundled reference log:', '',
                       '| Metric | Bundled reference (%) | This PC (%) | Difference (pp) |',
                       '| --- | ---: | ---: | ---: |']
            for metric, value in zip(METRICS, map(float, match.groups())):
                measured = official['mean'][metric]
                report.append('| %s | %.4f | %.4f | %+.4f |' % (metric, value * 100, measured * 100, (measured - value) * 100))
        report += ['', 'See `comparison.csv` for every category and metric; `training/training.json`',
                   'contains per-epoch losses and runtime. `status.json` records the exact commands.', '']
        (output / 'REPORT.md').write_text('\n'.join(report))
        status.update(official_sha256_after=sha256(OFFICIAL),
                      local_checkpoint_sha256=sha256(trained_checkpoint),
                      completed_at_utc=datetime.now(timezone.utc).isoformat())
        assert status['official_sha256_after'] == official_digest
        status['complete'] = True
        save_status()
        print('Comparison complete: ' + str(output / 'REPORT.md'), flush=True)
    except Exception as error:
        status['error'] = str(error)
        save_status()
        raise


if __name__ == '__main__':
    main()
