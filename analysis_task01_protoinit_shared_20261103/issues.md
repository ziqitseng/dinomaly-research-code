# Issues and limitations

1. The two positive `laser_diode` views in `NG/OilStain/S0005` (cameras C1 and C4) have all-zero source PNG masks. In addition, ArcAD's native 448 resize / 392 crop / 256-map transform produces empty masks for 37 other positive `laser_diode` views with small annotations. The exact labels and maps were retained; see `output_validation.json` for paths. This affects ArcAD pixel metrics and is part of the native preprocessing protocol.
2. ArcAD and DiNomaly2 use different image sizes and center-crop transforms before producing 256×256 maps. Their pixel scores are reported with the same metric formulas, but pixel deltas are left blank because those map/mask coordinate frames do not represent identical crops.
3. DiNomaly2 inference maps were generated in the prior DiNomaly2-only evaluation. Their sample membership and per-view score consistency were checked against the fresh manifest. The ArcAD side was newly inferred from the prototype-initialized checkpoint.
4. DiNomaly2 native-test inference was not repeated in this update. Its reference I-AUROC values are shown separately and are not shared-split scores.
5. All inference ran on CPU in the current environment. The ArcAD checkpoint remained unchanged by SHA-256 and modification-time checks.

No additional categories were evaluated.
