#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_DIR"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XFORMERS_DISABLED=1
export MPLCONFIGDIR="$TASK_DIR/.runtime/matplotlib"
export TORCH_HOME="$TASK_DIR/.runtime/torch"
export XDG_CACHE_HOME="$TASK_DIR/.runtime/cache"
TASK_PYTHON="${PYTHON_BIN:-/home/avlab/anaconda3/envs/arcad/bin/python}"
case "${1:-plan}" in
  plan)
    cat <<'PLAN'
Protocol B setup only. No run is started by default.
Read README.md, PROTOCOL_B_REVISION.md, PROTOCOL.md, DATASET_POLICY.md, and MANUAL_RUNBOOK.md.
The 3,235-entry list is indexed. Real EMPTY masks are deferred to manual prepare.
Manual stages:
  bash run.sh preflight
  bash run.sh test
  bash run.sh prepare --resume --workers 4
  bash run.sh smoke arcad
  bash run.sh smoke dinomaly2
  bash run.sh train arcad 1 [--device cuda:0] [--resume]
  bash run.sh train dinomaly2 1 [--device cuda:0] [--resume]
  bash run.sh evaluate arcad 1 [--device cuda:0]
  bash run.sh evaluate dinomaly2 1 [--device cuda:0]
Repeat train/evaluate for training seeds 2 and 3, then:
  bash run.sh summarize
Experiment outputs stay in this folder; synthetic tests use temporary directories.
Set PYTHON_BIN to a compatible existing interpreter if necessary.
Preparation: stop any old job first; --resume keeps its cache. See FAST_PREPARE.md.
PLAN
    exit 0
    ;;
  preflight|prepare|summarize)
    TASK_STAGE="$1"
    shift
    TASK_COMMAND=("$TASK_PYTHON" "scripts/${TASK_STAGE}.py" "$@")
    ;;
  train|evaluate|smoke)
    TASK_STAGE="$1"
    shift
    TASK_COMMAND=("$TASK_PYTHON" "scripts/${TASK_STAGE}.py" "$@")
    ;;
  test)
    TASK_STAGE=test
    TASK_COMMAND=("$TASK_PYTHON" -m unittest discover -s tests -v)
    ;;
  *)
    printf 'Unknown stage: %s\n' "$1" >&2
    exit 2
    ;;
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
