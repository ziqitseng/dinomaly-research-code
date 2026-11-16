#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/dinomaly2/bin/python"
"$PYTHON" "$ROOT/analysis_task04A_crop_ablation_20261103/work/prepare.py"
"$PYTHON" "$ROOT/analysis_task04A_crop_ablation_20261103/work/validate_geometry.py"
"$PYTHON" "$ROOT/analysis_task04A_crop_ablation_20261103/work/inference.py" --smoke
"$PYTHON" "$ROOT/analysis_task04A_crop_ablation_20261103/work/inference.py" --run
"$PYTHON" "$ROOT/analysis_task04A_crop_ablation_20261103/work/analyze.py"
"$PYTHON" "$ROOT/analysis_task04A_crop_ablation_20261103/work/size_strata.py"
"$PYTHON" "$ROOT/analysis_task04A_crop_ablation_20261103/work/finalize.py"
