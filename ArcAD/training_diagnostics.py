"""Training-only numerical checks and atomic diagnostic snapshots (2026-11-06)."""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class TrainingDiagnostics:
    def __init__(self, directory, args, logger):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.batch = {}
        self.attention_modules = {}
        config = dict(vars(args))
        root = Path(__file__).resolve().parent
        config['source_sha256'] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ('arcad_realiad_uni.py', 'models/vision_transformer.py', 'training_diagnostics.py')
        }
        (self.directory / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')

    def set_batch(self, step, epoch, batch_index, images, masks, labels, paths):
        self.batch = dict(step=step, epoch=epoch, batch_index=batch_index,
                          images=images.detach(), masks=masks.detach(),
                          labels=labels.detach(), paths=list(paths))

    def check(self, tensors, stage):
        items = [(name, value.detach()) for name, value in tensors.items() if value is not None]
        if not items:
            return
        flags = torch.stack([torch.isfinite(value).all() for _, value in items])
        if flags.all().item():
            return
        bad = [name for (name, _), finite in zip(items, flags.cpu().tolist()) if not finite]
        info = {key: self.batch[key] for key in ('step', 'epoch', 'batch_index', 'paths')}
        info.update(stage=stage, nonfinite_tensors=bad)
        info['tensor_details'] = {
            name: dict(shape=list(value.shape), nan_count=int(torch.isnan(value).sum()),
                       inf_count=int(torch.isinf(value).sum()))
            for name, value in items if name in bad
        }
        # Retain per-layer denominator evidence for the last attempted forward.
        info['attention_last_forward'] = {
            name: {key: str(value.item()) for key, value in module.last_diagnostics.items()}
            for name, module in self.attention_modules.items()
        }
        (self.directory / 'nonfinite_failure.json').write_text(json.dumps(info, indent=2) + '\n')
        batch = {name: value.cpu() if torch.is_tensor(value) else value
                 for name, value in self.batch.items()}
        atomic_torch_save(batch, self.directory / 'nonfinite_batch.pth')
        message = f'Non-finite values at step {self.batch["step"]}, stage={stage}: {bad}'
        self.logger(message + '; batch saved to nonfinite_batch.pth')
        raise FloatingPointError(message)

    def snapshot(self, completed_steps, model, learner, discriminator, optimizer, scheduler):
        # Stores diagnostic state. No claim of exact replay with multiworker loaders.
        state = dict(completed_steps=completed_steps, model=model.state_dict(),
                     learner=learner.state_dict(), discriminator=discriminator.state_dict(),
                     learner_initialized=learner.is_initialized,
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                     epoch=self.batch.get('epoch'), batch_index=self.batch.get('batch_index'),
                     last_batch_paths=self.batch.get('paths'))
        atomic_torch_save(state, self.directory / 'training_state_latest.pth')
        self.logger(f'Saved training_state_latest.pth at step {completed_steps}')
