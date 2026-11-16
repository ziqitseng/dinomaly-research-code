#!/usr/bin/env bash
set -euo pipefail

workspace=/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/Cost_Filter_AD
data=/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_raw
run="$workspace/costfilter_setup/runs/dinomaly_variety_50k_20261116"
python_bin="$workspace/.conda/envs/dinomaly_variety/bin/python"
trainer="$workspace/costfilter-ad/Costfilter_Dinomaly/dinomaly_variety_train.py"

mkdir -p "$run" /tmp/dinomaly_variety_mpl
cd "$workspace"
export MPLCONFIGDIR=/tmp/dinomaly_variety_mpl
export PYTHONUNBUFFERED=1

on_exit() {
    exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        printf 'failed exit=%s time=%s\n' "$exit_code" "$(date -Is)" > "$run/status.txt"
    fi
}
trap on_exit EXIT

printf 'training started time=%s\n' "$(date -Is)" > "$run/status.txt"
latest_checkpoint=$("$python_bin" -c 'import glob,re,sys; p=glob.glob(sys.argv[1]+"/trainer_step_*.pth"); print(max(p,key=lambda x:int(re.search(r"trainer_step_(\d+)\.pth$",x).group(1))) if p else "")' "$run")
resume_args=()
if [ -n "$latest_checkpoint" ]; then
    printf 'Resuming from %s\n' "$latest_checkpoint"
    resume_args=(--resume-checkpoint "$latest_checkpoint")
fi
"$python_bin" -B "$trainer" \
    --data-path "$data" --output-dir "$run" \
    --total-steps 50000 --batch-size 16 --num-workers 4 \
    --save-every 5000 --log-every 100 "${resume_args[@]}"
printf 'complete time=%s checkpoint=%s\n' "$(date -Is)" "$run/model.pth" > "$run/status.txt"
