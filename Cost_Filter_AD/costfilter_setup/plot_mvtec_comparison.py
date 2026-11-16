"""Export comparison and training-loss figures after the reproduction finishes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path, nargs='?', default=Path(__file__).resolve().parent / 'runs/mvtec_reproduction_20261113')
args = parser.parse_args()
run = args.run.resolve()
status = json.loads((run / 'status.json').read_text())
assert status['complete'], 'Wait for both full evaluations to finish.'
official = json.loads((run / 'official/metrics.json').read_text())
local = json.loads((run / 'local/metrics.json').read_text())
training = json.loads((run / 'training/training.json').read_text())
categories = list(official['categories'])
assert set(categories) == set(local['categories'])
x = np.arange(len(categories))
fig, axis = plt.subplots(figsize=(13, 4.7), constrained_layout=True)
axis.bar(x - .2, [100 * official['categories'][c]['P_AP'] for c in categories], width=.4,
         color='#287D9B', label='Published CostFilter weights')
axis.bar(x + .2, [100 * local['categories'][c]['P_AP'] for c in categories], width=.4,
         color='#E4872B', label='Local training, final epoch 17')
axis.set(xticks=x, xticklabels=[c.replace('_', ' ') for c in categories], ylim=(0, 100),
         ylabel='Pixel average precision (%)', title='MVTec-AD: published and locally trained CostFilter checkpoints')
axis.tick_params(axis='x', labelrotation=40)
axis.set_axisbelow(True)
axis.grid(axis='y', alpha=.2)
axis.legend(loc='lower left', framealpha=.95)
fig.savefig(run / 'pixel_ap_comparison.png', dpi=180)
plt.close(fig)

records = training['epochs']
assert len(records) == 17
fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
axis.plot([r['epoch'] + 1 for r in records], [r['mean_loss'] for r in records],
          marker='o', color='#287D9B', linewidth=1.8)
axis.set(xlabel='Completed training epoch', ylabel='Mean training loss',
         title='CostFilter training on MVTec-AD + DTD', xticks=[1, 5, 9, 13, 17])
axis.grid(alpha=.2)
fig.savefig(run / 'training_loss.png', dpi=180)
plt.close(fig)

report = run / 'REPORT.md'
with report.open('a') as stream:
    stream.write('\n![Per-category pixel AP comparison](pixel_ap_comparison.png)\n\n')
    stream.write('![Training loss](training_loss.png)\n')
print(run / 'pixel_ap_comparison.png')
print(run / 'training_loss.png')
