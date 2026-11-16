"""Small mathematical/RNG regression against the pinned original ArcAD method.

No model construction, real data, training, or evaluation. On CUDA, the original
reference alone is run with deterministic checks disabled because its gather
backward is unsupported. The replacement is always checked with strict mode on.
"""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

from common import ROOT, check_lock, config, now, write_json


def original_augmentation():
    import torch
    import torch.nn.functional as F
    path = Path(config()['arcad_repo']) / 'arcad_realiad_uni.py'
    tree = ast.parse(path.read_text())
    learner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'BottleneckPrototypeLearner')
    method = next(node for node in learner.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'augment_features')
    namespace = dict(torch=torch, F=F)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['augment_features']


def check_selection(device='cpu'):
    import torch
    import torch.nn.functional as F
    from deterministic_ops import augment_features
    original = original_augmentation()
    original_flag = torch.are_deterministic_algorithms_enabled()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.startswith('cuda') else None
    results = []
    try:
        for dtype in (torch.float32, torch.float64):
            for count, width in ((1, 13), (4, 64)):
                anchors = torch.linspace(-.8, .8, count * width, dtype=dtype, device=device).reshape(count, width)
                prototypes = F.normalize(torch.linspace(-1, 1, 9 * width, dtype=dtype,
                    device=device).reshape(9, width), dim=1)
                learner = SimpleNamespace(prototypes=prototypes, noise_std=.015)
                weights = torch.linspace(.2, 1., count * width, dtype=dtype, device=device).reshape(count, width)

                def run(function, strict):
                    torch.use_deterministic_algorithms(strict)
                    torch.manual_seed(12345)
                    if device.startswith('cuda'):
                        torch.cuda.manual_seed_all(12345)
                    x = anchors.clone().requires_grad_(True)
                    value = function(learner, x)
                    (value * weights).sum().backward()
                    rng = torch.cuda.get_rng_state(device) if device.startswith('cuda') else torch.get_rng_state()
                    return value.detach(), x.grad.detach(), rng

                reference = run(original, strict=False)
                updated = run(augment_features, strict=True)
                repeat = run(augment_features, strict=True)
                tolerance = 1e-6 if dtype == torch.float32 else 1e-12
                torch.testing.assert_close(updated[0], reference[0], rtol=tolerance, atol=tolerance)
                torch.testing.assert_close(updated[1], reference[1], rtol=tolerance, atol=tolerance)
                assert torch.equal(updated[2], reference[2]), 'Candidate search changed RNG consumption'
                assert torch.equal(updated[0], repeat[0]) and torch.equal(updated[1], repeat[1]), 'Strict repeat differs'
                results.append(dict(dtype=str(dtype), anchors=count, width=width,
                    output_max_error=float((updated[0] - reference[0]).abs().max()),
                    gradient_max_error=float((updated[1] - reference[1]).abs().max()),
                    same_rng_state=True, strict_repeat_exact=True))
    finally:
        torch.use_deterministic_algorithms(original_flag)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return dict(device=device, cases=results, strict_replacement_passed=True,
                reference_only_allows_nondeterminism=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    lock_hash = check_lock()
    report = dict(check_selection(args.device), timestamp=now(), input_lock_sha256=lock_hash)
    stamp = now().replace(':', '').replace('+', '_')
    write_json(ROOT / 'logs' / ('deterministic_selection_' + stamp + '.json'), report)
    print(report)
