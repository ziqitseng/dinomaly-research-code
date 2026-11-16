#!/usr/bin/env bash
set -euo pipefail
PY=/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/dinomaly2/bin/python
"$PY" analysis_task04A_crop_ablation_20261102_01/work/preflight.py
"$PY" analysis_task04A_crop_ablation_20261102_01/work/run_arcad_branches.py
MPLCONFIGDIR=/tmp/mpl-task04a "$PY" analysis_task04A_crop_ablation_20261102_01/work/analyze_task04a.py
MPLCONFIGDIR=/tmp/mpl-task04a "$PY" analysis_task04A_crop_ablation_20261102_01/work/finalize_task04a.py
