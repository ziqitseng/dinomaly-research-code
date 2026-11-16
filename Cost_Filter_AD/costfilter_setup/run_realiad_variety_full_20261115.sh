#!/usr/bin/env bash
set -euo pipefail

workspace=/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/Cost_Filter_AD
data=/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_raw
run="$workspace/costfilter_setup/runs/realiad_variety_full_20261115_transfer"
python_bin="$workspace/.conda/envs/cosfilterad/bin/python"
runner="$workspace/costfilter-ad/Costfilter_Dinomaly/costfilter_dinomaly_variety_uni_train_test.py"
teacher="$workspace/costfilter-ad/Costfilter_Dinomaly/saved_results_realiad/realiad_vitb_it100k.pth"
teacher_description='Real-IAD Dinomaly-B cross-dataset transfer'

mkdir -p "$run/training" "$run/evaluation" /tmp/costfilter_matplotlib
cd "$workspace"
export MPLCONFIGDIR=/tmp/costfilter_matplotlib
export PYTHONUNBUFFERED=1

on_exit() {
    exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        printf 'failed exit=%s time=%s\n' "$exit_code" "$(date -Is)" > "$run/status.txt"
    fi
}
trap on_exit EXIT

printf 'training started time=%s\n' "$(date -Is)" > "$run/status.txt"
"$python_bin" -B "$runner" \
    --train --data-path "$data" --dtd-path "$workspace/datasets/dtd/images" \
    --dinomaly-checkpoint "$teacher" --teacher-description "$teacher_description" \
    --output-dir "$run/training" --epochs 17 --batch-size 12 --num-workers 4

checkpoint=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["epochs"][-1]["checkpoint"])' "$run/training/training.json")
printf 'inference started time=%s checkpoint=%s\n' "$(date -Is)" "$checkpoint" > "$run/status.txt"
"$python_bin" -B "$runner" \
    --data-path "$data" --dinomaly-checkpoint "$teacher" \
    --teacher-description "$teacher_description" \
    --costfilter-checkpoint "$checkpoint" --output-dir "$run/evaluation" \
    --pixel-backend adeval --eval-batch-size 8 --num-workers 4

printf 'complete time=%s checkpoint=%s\n' "$(date -Is)" "$checkpoint" > "$run/status.txt"
