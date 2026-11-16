#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/dinomaly2/bin/python"
"$PYTHON" "$ROOT/analysis_task03_verification_20261103/work/phase0_inventory.py"
"$PYTHON" "$ROOT/analysis_task03_verification_20261103/work/phase1_audit_and_align.py"
"$PYTHON" "$ROOT/analysis_task03_verification_20261103/work/phase2_fusion_analysis.py"
