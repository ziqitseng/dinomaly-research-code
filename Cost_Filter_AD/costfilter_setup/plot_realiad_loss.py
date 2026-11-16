"""Plot per-step and per-epoch losses from a Real-IAD training run."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--window', type=int, default=200,
                        help='Moving-average window in optimizer steps')
    args = parser.parse_args()
    with (args.run_dir / 'step_losses.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    epochs = json.loads((args.run_dir / 'training.json').read_text())['epochs']
    if not rows or not epochs:
        raise ValueError('No completed Real-IAD training losses found')
    steps = np.asarray([int(row['step']) for row in rows])
    total = np.asarray([float(row['total_loss']) for row in rows])
    seg = np.asarray([float(row['segmentation_loss']) for row in rows])
    cls = np.asarray([float(row['classification_loss']) for row in rows])
    window = min(max(args.window, 1), len(total))
    smooth = np.convolve(total, np.ones(window) / window, mode='valid')

    fig, (ax_step, ax_epoch) = plt.subplots(2, 1, figsize=(10, 8),
                                            constrained_layout=True)
    ax_step.plot(steps, total, color='#9ab6d3', alpha=0.35, linewidth=0.6,
                 label='Total, each step')
    ax_step.plot(steps[window - 1:], smooth, color='#15528a', linewidth=1.7,
                 label='Total, %d-step mean' % window)
    ax_step.plot(steps, seg, color='#d97706', alpha=0.35, linewidth=0.5,
                 label='Segmentation')
    ax_step.plot(steps, 0.5 * cls, color='#7c3aed', alpha=0.35,
                 linewidth=0.5, label='0.5 × classification')
    ax_step.set(xlabel='Optimizer step', ylabel='Loss', title='Real-IAD CostFilter training')
    ax_step.grid(alpha=0.2)
    ax_step.legend(fontsize=8)

    ax_epoch.plot([entry['epoch'] + 1 for entry in epochs],
                  [entry['mean_loss'] for entry in epochs], 'o-',
                  color='#15528a')
    ax_epoch.set(xlabel='Epoch', ylabel='Mean total loss',
                 title='Completed epoch means')
    ax_epoch.grid(alpha=0.2)
    output = args.run_dir / 'training_loss.png'
    fig.savefig(output, dpi=160)
    print(output)


if __name__ == '__main__':
    main()
