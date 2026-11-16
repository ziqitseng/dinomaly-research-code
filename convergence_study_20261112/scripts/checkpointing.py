"""Immutable full-state snapshots; latest.pth remains replaceable."""
import os
import tempfile
from pathlib import Path


def _equal(a, b):
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def save_milestone(path, state):
    """Link a complete checkpoint into place only if absent; verify any prior copy."""
    import torch
    from model_adapter import torch_load
    path = Path(path)
    if state['step'] <= 0 or path.name != 'milestone_%06d.pth' % state['step']:
        raise ValueError('Milestone name must match completed optimizer steps')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = torch_load(path, 'cpu')
        if not _equal(old, state):
            raise RuntimeError('Existing milestone differs: ' + str(path))
        return False
    fd, temporary = tempfile.mkstemp(prefix='.milestone_', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)  # Fails atomically if another writer created it.
        except FileExistsError:
            old = torch_load(path, 'cpu')
            if not _equal(old, state):
                raise RuntimeError('Existing milestone differs: ' + str(path))
            return False
        return True
    finally:
        os.unlink(temporary)


def learning_rate(completed_before_step, total_steps, warmup=100):
    import math
    if total_steps <= warmup or not 0 <= completed_before_step < total_steps:
        raise ValueError('Invalid schedule position')
    if completed_before_step < warmup:
        return 2e-3 * (completed_before_step + 1) / warmup
    phase = (completed_before_step - warmup) / (total_steps - warmup - 1)
    return 2e-4 + 0.5 * (2e-3 - 2e-4) * (1 + math.cos(math.pi * phase))
