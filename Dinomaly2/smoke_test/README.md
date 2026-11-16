# Dinomaly2 one-image training smoke test

This smoke test sends exactly one normal image through the Real-IAD training
preprocessing, Dinomaly2 encoder/bottleneck/decoder, training loss, backward
pass, and one `StableAdamW` update. It writes a lightweight checkpoint of the
trainable bottleneck and decoder plus a JSON run summary under `output/`.

From the workspace root:

```bash
conda activate ./dinomaly2
cd Dinomaly2
python smoke_test/run_one_image.py
```

By default, the script uses the first normal training image in the Real-IAD
`audiojack` category. It downloads the official DINOv2 ViT-B/14-with-registers
backbone weights on the first run if they are not already cached. To use a
specific image instead, pass `--image /path/to/image.png`.

This verifies that one training step runs; it does not train a useful anomaly
detector or evaluate anomaly metrics.
