#!/usr/bin/env bash
set -euo pipefail

workspace_root="/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly"
template_dir="$(cd "$(dirname "$0")" && pwd)"
python_bin="$workspace_root/dinomaly2/bin/python"
base="$workspace_root/analysis_task01_protoinit_shared_$(date +%Y%m%d_%H%M%S)"
target="$base"
suffix=1
while ! mkdir "$target" 2>/dev/null; do
  target="${base}_${suffix}"
  suffix=$((suffix + 1))
done
mkdir "$target/work"
cp "$template_dir/work/"*.py "$target/work/"
cp "$template_dir/README.md" "$template_dir/issues.md" "$template_dir/reproduce.sh" "$target/"

TASK01_OUTPUT_DIR="$target" "$python_bin" "$target/work/build_shared_manifest.py"
TASK01_OUTPUT_DIR="$target" "$python_bin" "$target/work/evaluate_arcad_protoinit.py"
TASK01_OUTPUT_DIR="$target" "$python_bin" "$target/work/validate_reusable_d2_maps.py"
TASK01_OUTPUT_DIR="$target" "$python_bin" "$target/work/finalize_comparison.py"
printf 'Saved fresh run to %s\n' "$target"
