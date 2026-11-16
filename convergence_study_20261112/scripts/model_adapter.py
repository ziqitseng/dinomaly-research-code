"""Construct original architectures in separate processes; never import both repos."""
import inspect
import sys
from pathlib import Path


def encoder_load_arguments(encoder_loader, cfg):
    """ArcAD accepts only name; Dinomaly2 additionally accepts WEIGHTS_DIR."""
    signature = inspect.signature(encoder_loader.load)
    kwargs = {}
    if 'WEIGHTS_DIR' in signature.parameters:
        kwargs['WEIGHTS_DIR'] = str(Path(cfg['backbone_file']).parent)
    signature.bind('dinov2reg_vit_base_14', **kwargs)
    return kwargs


def validate_repository_imports(name, cfg):
    """Check the actual import path and loader signature without constructing a model."""
    encoder_loader = repository_imports(name, cfg)
    from models.vision_transformer import Block, LinearAttention2
    from optimizers import StableAdamW
    from utils import global_cosine_hm_percent
    if name == 'arcad':
        from models.uad import ViTill
        from arcad_realiad_uni import BottleneckPrototypeLearner, Discriminator, BinaryFocalLoss
    else:
        from models.uad import Dinomaly
    return dict(model=name, loader_signature=str(inspect.signature(encoder_loader.load)),
                loader_kwargs=encoder_load_arguments(encoder_loader, cfg))


def repository_imports(name, cfg):
    repo = Path(cfg[name + '_repo'])
    sys.path.insert(0, str(repo))
    # No hidden download or write into either original repository.
    import models.vit_encoder as encoder_loader

    def local_only(url, *args, **kwargs):
        if not str(url).endswith('dinov2_vitb14_reg4_pretrain.pth'):
            raise RuntimeError('Unapproved backbone request: ' + str(url))
        return cfg['backbone_file']

    encoder_loader.download_cached_file = local_only
    return encoder_loader


def build(name, cfg, device):
    import torch
    from functools import partial

    encoder_loader = repository_imports(name, cfg)
    from models.vision_transformer import Block, LinearAttention2
    # Both loaders resolve weights through local_only; respect their different call signatures.
    encoder = encoder_loader.load('dinov2reg_vit_base_14', **encoder_load_arguments(encoder_loader, cfg))
    encoder.requires_grad_(False)
    encoder.eval()
    width = 768
    decoder = torch.nn.ModuleList([Block(dim=width, num_heads=12, mlp_ratio=4., qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-8),
        attn=partial(LinearAttention2, eps=cfg['attention_eps'])) for _ in range(8)])
    shared = dict(encoder=encoder, decoder=decoder, target_layers=list(range(2, 10)),
                  remove_class_token=False, fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
                  fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]])
    if name == 'arcad':
        from models.vision_transformer import bMlp
        from models.uad import ViTill
        from arcad_realiad_uni import BottleneckPrototypeLearner, Discriminator, BinaryFocalLoss
        from deterministic_ops import augment_features
        class DeterministicPrototypeLearner(BottleneckPrototypeLearner):
            def augment_features(self, features):
                return augment_features(self, features)
        bottleneck = torch.nn.ModuleList([bMlp(width, width * 4, width, drop=0.4)])
        model = ViTill(bottleneck=bottleneck, mask_neighbor_size=0, **shared)
        learner = DeterministicPrototypeLearner(width, num_prototypes=cfg['prototype_count'],
            num_special_tokens=5, noise_std=0.015, cluster_mode='sinkhorn').to(device)
        discriminator = Discriminator(width).to(device)
        trainable = torch.nn.ModuleList([bottleneck, decoder, discriminator])
        for module in trainable.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
        extra = dict(learner=learner, discriminator=discriminator,
                     focal=BinaryFocalLoss(alpha=0.75, gamma=2.0).to(device))
    else:
        from models.uad import Dinomaly
        bottleneck = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(width, 256), torch.nn.Dropout(0.4)),
            torch.nn.Sequential(torch.nn.Linear(256, width * 4), torch.nn.GELU(), torch.nn.Dropout(0.4),
                torch.nn.Linear(width * 4, width), torch.nn.Dropout(0.4))])
        model = Dinomaly(bottleneck=bottleneck, context_aware_recenter=True, **shared)
        model.init_weights()
        trainable = torch.nn.ModuleList([bottleneck, decoder])
        extra = {}
    return model.to(device), trainable, extra


def torch_load(path, device='cpu'):
    import torch
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Older local PyTorch versions.
        return torch.load(path, map_location=device)
