#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_DIR"
export COLD_START_BASE_ROOT="${COLD_START_BASE_ROOT:-$TASK_DIR/../comparison_cold_start_20261109}"
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg XFORMERS_DISABLED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="$TASK_DIR/.runtime/matplotlib" TORCH_HOME="$TASK_DIR/.runtime/torch" XDG_CACHE_HOME="$TASK_DIR/.runtime/cache"
TASK_PYTHON="${PYTHON_BIN:-/home/avlab/anaconda3/envs/arcad/bin/python}"
case "${1:-plan}" in
  plan)
    cat <<'PLAN'
Convergence study. No run starts by default.
  bash run.sh inspect-validation
  bash run.sh test
  bash run.sh train 100k arcad 1 --device cuda:0
  bash run.sh train 100k dinomaly2 1 --device cuda:0
  bash run.sh train 50k arcad 1 --device cuda:0
  bash run.sh train 50k dinomaly2 1 --device cuda:0
  bash run.sh evaluate 100k arcad 1 --checkpoint runs/100k/arcad/seed_1/milestone_050000.pth --output-dir evaluations/A/arcad/seed_1/step_050000 --split validation --manifest manifests/validation.json
See IMPLEMENTATION_REPORT.md before running. Training and evaluation are manual GPU stages.
PLAN
    exit 0 ;;
  train|evaluate)
    TASK_STAGE="$1"; shift
    export STUDY_SCHEDULE="${1:?Specify 100k or 50k schedule}"; shift
    case "$STUDY_SCHEDULE" in 100k|50k) ;; *) echo 'Schedule must be 100k or 50k' >&2; exit 2;; esac
    TASK_COMMAND=("$TASK_PYTHON" "scripts/${TASK_STAGE}.py" "$@") ;;
  inspect-validation)
    TASK_STAGE=inspect_validation; shift
    TASK_COMMAND=("$TASK_PYTHON" scripts/validation_manifest.py "$@") ;;
  test)
    TASK_STAGE=test; shift
    TASK_COMMAND=("$TASK_PYTHON" -m unittest discover -s tests -v "$@") ;;
  *) echo "Unknown stage: $1" >&2; exit 2 ;;
esac
mkdir -p logs
TASK_LOG="logs/${TASK_STAGE}_$(date -u +%Y%m%dT%H%M%S%NZ).log"
printf 'Stage: %s\nInterpreter: %s\n' "$TASK_STAGE" "$TASK_PYTHON" > "$TASK_LOG"
set +e
"${TASK_COMMAND[@]}" 2>&1 | tee -a "$TASK_LOG"
TASK_PIPE_STATUS=("${PIPESTATUS[@]}")
TASK_STATUS=${TASK_PIPE_STATUS[0]}
if [ "$TASK_STATUS" -eq 0 ]; then TASK_STATUS=${TASK_PIPE_STATUS[1]}; fi
set -e
printf 'Exit status: %d\n' "$TASK_STATUS" | tee -a "$TASK_LOG"
exit "$TASK_STATUS"
