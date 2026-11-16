"""Wait for full Real-IAD training, then plot losses and evaluate all views."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
RUNNER = WORKSPACE / 'costfilter-ad/Costfilter_Dinomaly/costfilter_dinomaly_realiad_uni_train_test.py'
PLOTTER = WORKSPACE / 'costfilter_setup/plot_realiad_loss.py'
DATA = Path('/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/dataset/Real-IAD')


def now():
    return datetime.now(timezone.utc).isoformat()


def save_status(path, status):
    status['updated_at_utc'] = now()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(status, indent=2) + '\n')
    temporary.replace(path)


def training_alive(pid):
    try:
        command = Path('/proc/%d/cmdline' % pid).read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    return (b'costfilter_dinomaly_realiad_uni_train_test.py' in command and
            b'--train' in command)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--expected-epochs', type=int, default=17)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    status_path = root / 'workflow_status.json'
    status = {'training_pid': args.training_pid, 'expected_epochs': args.expected_epochs,
              'training': 'running', 'plot': 'pending', 'evaluation': 'pending',
              'started_at_utc': now()}
    save_status(status_path, status)
    try:
        while training_alive(args.training_pid):
            time.sleep(60)
        training_dir = root / 'training'
        record_path = training_dir / 'training.json'
        if not record_path.is_file():
            raise RuntimeError('Training process exited without training.json')
        records = json.loads(record_path.read_text())['epochs']
        if len(records) != args.expected_epochs:
            raise RuntimeError('Training stopped after %d of %d epochs' %
                               (len(records), args.expected_epochs))
        final_checkpoint = Path(records[-1]['checkpoint'])
        if not final_checkpoint.is_file() or final_checkpoint.stat().st_size < 1_000_000:
            raise RuntimeError('Final CostFilter checkpoint is missing or incomplete')
        status.update(training='complete', training_steps=records[-1]['step'],
                      final_checkpoint=str(final_checkpoint),
                      final_checkpoint_sha256=sha256(final_checkpoint))
        save_status(status_path, status)

        status['plot'] = 'running'
        save_status(status_path, status)
        subprocess.run([sys.executable, '-B', str(PLOTTER), str(training_dir)],
                       cwd=str(WORKSPACE), check=True)
        status['plot'] = 'complete'
        save_status(status_path, status)

        evaluation_dir = root / 'evaluation'
        evaluation_dir.mkdir(exist_ok=True)
        status['evaluation'] = 'running'
        save_status(status_path, status)
        command = [sys.executable, '-B', str(RUNNER),
                   '--data-path', str(DATA),
                   '--costfilter-checkpoint', str(final_checkpoint),
                   '--output-dir', str(evaluation_dir),
                   '--eval-batch-size', '8', '--num-workers', '4',
                   '--pixel-backend', 'adeval', '--resume-eval']
        status['evaluation_command'] = command
        save_status(status_path, status)
        print('Running:', ' '.join(command), flush=True)
        subprocess.run(command, cwd=str(WORKSPACE), check=True)
        metrics = json.loads((evaluation_dir / 'metrics.json').read_text())
        if not metrics['complete'] or len(metrics['categories']) != 30:
            raise RuntimeError('Full 30-category evaluation did not complete')
        if sum(row['views'] for row in metrics['categories'].values()) != 114585:
            raise RuntimeError('Evaluation view count is not 114,585')
        if sum(row['objects'] for row in metrics['categories'].values()) != 22917:
            raise RuntimeError('Evaluation object count is not 22,917')
        status['evaluation'] = 'complete'
        status['mean_metrics'] = metrics['mean']
        save_status(status_path, status)
        print('Full Real-IAD workflow complete', flush=True)
    except Exception as error:
        status['error'] = repr(error)
        if status['training'] != 'complete':
            status['training'] = 'failed'
        elif status['plot'] != 'complete':
            status['plot'] = 'failed'
        else:
            status['evaluation'] = 'failed'
        save_status(status_path, status)
        raise


if __name__ == '__main__':
    main()
