"""Real-IAD sample grouping following ADEval's sample-level convention."""
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from utils import f1_score_max


SAMPLE_KEY = re.compile(
    r'([a-zA-Z][a-zA-Z0-9_]*_[0-9]{4}_[A-Z][A-Z_]*[A-Z])_C[0-9]_'
)


def sample_id(path):
    match = SAMPLE_KEY.search(str(path))
    if match is None:
        raise ValueError('No Real-IAD sample ID in path: %s' % path)
    return match.group(1)


def sample_scores(paths, scores, labels, expected_views=5, key_fn=sample_id):
    """Use max view score/label per object, as in ADEval's Real-IAD example."""
    if not (len(paths) == len(scores) == len(labels)):
        raise ValueError('View paths, scores, and labels have different lengths')
    groups = defaultdict(list)
    for index, (path, score, label) in enumerate(zip(paths, scores, labels)):
        groups[key_fn(path)].append((index, float(score), int(label)))
    rows = []
    for key in sorted(groups):
        views = groups[key]
        if expected_views is not None and len(views) != expected_views:
            raise ValueError('Real-IAD sample %s has %d views, expected %d' %
                             (key, len(views), expected_views))
        rows.append({'sample_id': key, 'view_count': len(views),
                     'score': max(view[1] for view in views),
                     'label': max(view[2] for view in views),
                     'indices': [view[0] for view in views]})
    return rows


def classification_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) == 0 or len(np.unique(labels)) != 2:
        raise ValueError('AUROC requires normal and anomalous examples')
    return {'AUROC': float(roc_auc_score(labels, scores)),
            'AP': float(average_precision_score(labels, scores)),
            'F1max': float(f1_score_max(labels, scores))}
