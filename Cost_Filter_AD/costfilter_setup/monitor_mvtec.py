"""Print the latest durable state of the MVTec reproduction run."""
import json
import sys
from pathlib import Path

run = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'runs/mvtec_reproduction_20261113'
status = json.loads((run / 'status.json').read_text())
stage = status['stages'][-1]['name']
print('Stage:', stage, '| complete:', status['complete'], '| error:', status.get('error'))
for name in ['official', 'smoke', 'training', 'local']:
    path = run / name / ('training.json' if name in ['smoke', 'training'] else 'metrics.json')
    if not path.exists():
        continue
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError:
        continue  # A category/epoch result is currently being written.
    if 'epochs' in record:
        latest = record['epochs'][-1]
        print('%s: %d epochs; latest loss %.6f, %.1f s, lr %g' %
              (name, len(record['epochs']), latest['mean_loss'], latest['seconds'], latest['lr']))
    else:
        print('%s: %d/15 categories; complete: %s' % (name, len(record['categories']), record['complete']))
        if record.get('mean'):
            print('  Mean (%):', {k: round(100*v, 4) for k, v in record['mean'].items()})
path = run / stage / 'console.log'
if path.exists():
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 5000))
        tail = stream.read().decode('utf-8', errors='replace').replace('\r', '\n').splitlines()
    print('\n'.join(tail[-2:]))
