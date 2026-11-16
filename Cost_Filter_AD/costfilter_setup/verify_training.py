"""Bounded training feasibility check using the upstream MVTec training body.
Uses random weights and synthetic inputs; does not train a useful checkpoint.
Only data/checkpoint I/O, epoch count, and batch size are replaced in memory.
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import ast
import gc
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import torch

sys.dont_write_bytecode = True
repo = Path(__file__).resolve().parents[1] / 'costfilter-ad/Costfilter_Dinomaly'
sys.path.insert(0, str(repo))
import costfilter_dinomaly_mvtec_uni_my_train_test as upstream
from dinov2.models.vision_transformer import vit_base

def random_encoder(_):
    return vit_base(patch_size=14, img_size=518, block_chunks=0,
                    init_values=1e-8, num_register_tokens=4,
                    interpolate_antialias=False, interpolate_offset=0.1)

class NoOpWriter:
    def __init__(self, **kwargs):
        pass
    def add_scalar(self, *args, **kwargs):
        pass
    def close(self):
        pass

class SyntheticBatches:
    def __init__(self, batch):
        self.batch = batch
    def __iter__(self):
        image = torch.randn(self.batch, 3, 392, 392)
        mask = torch.zeros(self.batch, 1, 392, 392)
        mask[:, :, 100:200, 140:240] = 1
        labels = torch.arange(self.batch) % 15
        for step in range(3):
            torch.cuda.synchronize()
            self.start = time.perf_counter()
            self.step = step + 1
            yield image, mask, torch.zeros(self.batch), tuple('carpet' for _ in range(self.batch)), labels

def run(batch):
    tree = ast.parse((repo / 'costfilter_dinomaly_mvtec_uni_my_train_test.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'encoder_name')
    # Keep the real model construction, optimizer, cost volume, all losses,
    # backward pass, optimizer update, scheduler and gradient clipping.
    fn.body = ast.parse('setup_seed(1)\nbatch_size = %d' % batch).body + fn.body[start:]
    filtered = []
    for node in fn.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == 'state_dict':
                continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == 'load_state_dict':
                continue
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'epoch':
            node.iter = ast.parse('range(1)', mode='eval').body
            # Omit checkpoint writes and their associated messages only.
            end = next(i for i, stmt in enumerate(node.body) if isinstance(stmt, ast.Assign)
                       and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == 'checkpoint_path')
            node.body = node.body[:end]
            inner = next(stmt for stmt in node.body if isinstance(stmt, ast.For))
            position = next(i for i, stmt in enumerate(inner.body) if isinstance(stmt, ast.Expr)
                            and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute)
                            and stmt.value.func.attr == 'step')
            inner.body[position + 1:position + 1] = ast.parse('report_step(model_unet, optimizer, loss)').body
        filtered.append(node)
    fn.body = filtered
    batches = SyntheticBatches(batch)
    def report(model, optimizer, loss):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - batches.start
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)
        assert optimizer.state and torch.isfinite(loss)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        print(dict(batch=batch, step=batches.step, loss=round(loss.item(),5),
                   seconds=round(elapsed,3), peak_allocated_GiB=round(torch.cuda.max_memory_allocated()/2**30,3),
                   peak_reserved_GiB=round(torch.cuda.max_memory_reserved()/2**30,3)),flush=True)
    namespace = vars(upstream).copy()
    namespace.update(device='cuda:0', print_fn=print, train_data_list=[None], train_data=range(batch * 3),
                     train_dataloader=batches, SummaryWriter=NoOpWriter, report_step=report,
                     args=SimpleNamespace(output_dir=str(repo.parent.parent / 'costfilter_setup'), max_steps=None),
                     vit_encoder=SimpleNamespace(load=random_encoder))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),
                 str(repo / 'costfilter_dinomaly_mvtec_uni_my_train_test.py'), 'exec'),namespace)
    torch.cuda.reset_peak_memory_stats()
    namespace['train'](['carpet'], [])
    # Break the exec namespace's function/global cycle before the next size.
    namespace.clear()
    gc.collect()
    torch.cuda.empty_cache()

torch.set_num_threads(4)
print('GPU:',torch.cuda.get_device_name(),'Torch:',torch.__version__,flush=True)
for batch in [8,12]:
    run(batch)
