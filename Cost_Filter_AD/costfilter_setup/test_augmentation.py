"""Exercise every upstream augmenter on real uint8 texture/structure inputs."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'costfilter-ad/Costfilter_Dinomaly'))
from dataset import MVTecDataset_train, get_data_transforms

data_root = ROOT.parent / 'dataset/MVTec-AD/mvtec_anomaly_detection'
transform, gt_transform = get_data_transforms(448, 392)
dataset = MVTecDataset_train(str(data_root / 'carpet/train'), transform, gt_transform,
                            'train', 'carpet', 0, str(ROOT / 'datasets/dtd/images'))
image = np.array(Image.open(dataset.img_paths[0]).convert('RGB').resize((448, 448)))
for branch, seed in [('texture', 1), ('structure', 0)]:
    for augmenter in dataset.augmenters:
        # These seeds choose the requested branch on its first uniform draw.
        np.random.seed(seed)
        output = dataset.anomaly_source(image, dataset.anomaly_source_paths, augmenter)
        assert output.shape == image.shape
        assert output.dtype == np.float32
        assert np.isfinite(output).all()
        print('%s / %s: passed' % (branch, type(augmenter).__name__))
print('All 20 real-image augmentation branch checks passed.')
