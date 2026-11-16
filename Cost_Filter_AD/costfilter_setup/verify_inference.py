"""Run the repository's Dinomaly + CostFilter architecture with random weights.
No checkpoints, downloads, package changes, or anomaly accuracy validation.
"""
import gc
import sys
import time
from pathlib import Path
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F

sys.dont_write_bytecode = True
repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'costfilter-ad/Costfilter_Dinomaly'
sys.path.insert(0, str(repo))
from dinov2.models.vision_transformer import vit_base
from models.vision_transformer import Block, bMlp, LinearAttention2
from unet3d_att_dino_channel_test_48_min import DiscriminativeSubNetwork_3d_att_dino_channel
from models.uad import ViTill
from utils import cal_anomaly_maps
torch.set_num_threads(4)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
encoder = vit_base(patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
                   num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1)
bottleneck = nn.ModuleList([bMlp(768, 3072, 768, drop=0.2)])
decoder = nn.ModuleList([Block(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=LinearAttention2) for _ in range(8)])
model = ViTill(encoder, bottleneck, decoder, target_layers=[2,3,4,5,6,7,8,9],
              fuse_layer_encoder=[[0,1,2,3],[4,5,6,7]],
              fuse_layer_decoder=[[0,1,2,3],[4,5,6,7]]).cuda().eval()
filter_model = DiscriminativeSubNetwork_3d_att_dino_channel(768,2,48).cuda().eval()
print('GPU:', torch.cuda.get_device_name(), 'Torch:', torch.__version__, flush=True)
print('Python:', sys.version.split()[0], 'CUDA runtime:', torch.version.cuda, flush=True)
print('Total parameters:', sum(p.numel() for m in [model,filter_model] for p in m.parameters()), flush=True)

@torch.no_grad()
def forward(batch):
    img = torch.randn(batch, 3, 392, 392, device='cuda')
    en, de = model(img)
    volumes = []
    for rec_feat, org_feat in zip(en, de):
        pi = org_feat.reshape(batch,768,784).permute(0,2,1)
        pr = rec_feat.reshape(batch,768,784).permute(0,2,1)
        pi = pi / (torch.norm(pi, p=2, dim=-1, keepdim=True) + 1e-8)
        pr = pr / (torch.norm(pr, p=2, dim=-1, keepdim=True) + 1e-8)
        costs = 1 - torch.bmm(pi, pr.permute(0,2,1))
        _, indices = torch.topk(costs,768,dim=-1,largest=False,sorted=False)
        selected = torch.gather(costs,-1,torch.sort(indices,dim=-1)[0])
        volumes.append(selected.view(batch,28,28,768).unsqueeze(1))
    volume = torch.stack(volumes, dim=0).squeeze(2)
    volume = nn.UpsamplingBilinear2d(scale_factor=64/28)(volume.permute(0,1,4,2,3).reshape(-1,768,28,28))
    volume = volume.view(2,batch,768,64,64).permute(1,0,2,3,4).permute(0,2,1,3,4)
    pred, _ = cal_anomaly_maps(en, de, out_size=392)
    features = [e.view(batch,768,784).permute(0,2,1) for e in en]
    guidance = F.interpolate(pred,size=(64,64),mode='bilinear',align_corners=True).repeat(1,2,1,1)
    output, classes = filter_model(volume.float(),features,guidance.float())
    filtered = F.interpolate(torch.softmax(output,dim=1),size=392,mode='bilinear',align_corners=True)[:,1:2]
    result = filtered * .05 + pred * .95
    assert result.shape == (batch,1,392,392)
    assert torch.isfinite(result).all() and torch.isfinite(classes).all()
    return tuple(result.shape)

for batch in [1,12]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        shape = forward(batch)
        torch.cuda.synchronize()
        print(dict(batch=batch,shape=shape,seconds=round(time.perf_counter()-start,3),
                   peak_allocated_GiB=round(torch.cuda.max_memory_allocated()/2**30,3),
                   peak_reserved_GiB=round(torch.cuda.max_memory_reserved()/2**30,3)),flush=True)
    except torch.cuda.OutOfMemoryError as error:
        print('OUT OF MEMORY batch',batch,str(error),flush=True)
        raise
