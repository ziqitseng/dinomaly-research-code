# One-image ArcAD smoke test

From the ArcAD repository root, run:

```bash
conda activate arcad
python models/smoke_test/run_one_image.py --device cuda:0
```

The script loads one anomalous Real-IAD test image by default, applies ArcAD's
448-to-392 preprocessing, runs it through the DINOv2 + ViTill reconstruction
path, and computes the same cosine anomaly map used by evaluation. The default
data root is `../dataset/Real-IAD`; override it with `--data-path` if needed.

The encoder uses the pretrained DINOv2-reg weights. The ViTill bottleneck and
decoder are initialized randomly because this repository has no trained ArcAD
checkpoint yet, so the printed score only confirms the pipeline executes; it
is not an anomaly-detection result.
