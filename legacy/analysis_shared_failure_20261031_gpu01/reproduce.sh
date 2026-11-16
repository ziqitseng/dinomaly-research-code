#!/usr/bin/env bash
set -euo pipefail
PYTHON=/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/dinomaly2/bin/python
"$PYTHON" analysis_shared_failure_20261031_gpu01/work/prepare_analysis.py
"$PYTHON" analysis_shared_failure_20261031_gpu01/work/run_exports.py
MPLCONFIGDIR=/tmp/mpl-analysis "$PYTHON" analysis_shared_failure_20261031_gpu01/work/run_analysis.py
MPLCONFIGDIR=/tmp/mpl-analysis "$PYTHON" analysis_shared_failure_20261031_gpu01/work/complete_analysis.py
"$PYTHON" analysis_shared_failure_20261031_gpu01/work/finalize_outputs.py
