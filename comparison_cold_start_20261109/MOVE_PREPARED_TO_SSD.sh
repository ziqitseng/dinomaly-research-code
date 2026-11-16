#!/usr/bin/env bash
# Manual storage relocation only. Default prints the plan; --execute performs it.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_SSD_MOUNT=/media/avlab/afd90e24-420c-46a0-95e0-c017b48d0db0
TASK_SSD_BASE="$TASK_SSD_MOUNT/real_iad_variety_cold_start_cache_20261109"
TASK_DEST="$TASK_SSD_BASE/prepared"
TASK_SOURCE="$TASK_ROOT/prepared"
TASK_PYTHON="${PYTHON_BIN:-/home/avlab/anaconda3/envs/arcad/bin/python}"

if [ "${1:-plan}" != --execute ]; then
    printf 'Manual SSD relocation plan:\n  From: %s\n  To:   %s\n' "$TASK_SOURCE" "$TASK_DEST"
    printf 'Stop preparation first. --execute copies, verifies checksums, keeps the HDD folder as a backup, and installs a directory link.\n'
    printf 'No raw data, split, image pixels, model settings, or training budget changes.\n'
    printf 'Run: bash MOVE_PREPARED_TO_SSD.sh --execute\n'
    exit 0
fi

mountpoint -q "$TASK_SSD_MOUNT" || { printf 'SSD volume is not mounted: %s\n' "$TASK_SSD_MOUNT" >&2; exit 1; }
if [ -L "$TASK_SOURCE" ]; then
    if [ "$(readlink -f -- "$TASK_SOURCE")" = "$TASK_DEST" ] && [ -d "$TASK_DEST" ]; then
        printf 'prepared/ already points to this SSD. No copy performed.\n'
        exit 0
    fi
    printf 'prepared/ is already a link to another location. No changes made.\n' >&2
    exit 1
fi
[ -d "$TASK_SOURCE" ] || { printf 'No prepared/ folder exists to relocate.\n' >&2; exit 1; }

# This also detects the old preparation, which predates our output lock.
TASK_ACTIVE="$(PYTHONDONTWRITEBYTECODE=1 "$TASK_PYTHON" - "$TASK_ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'scripts'))
from common import check_lock
from prepare_cache import other_prepare_pids
check_lock()
active = other_prepare_pids(root / 'scripts/prepare.py')
if active:
    print(', '.join(map(str, active)))
PY
)"
[ -z "$TASK_ACTIVE" ] || { printf 'Stop preparation first (PID %s), then retry.\n' "$TASK_ACTIVE" >&2; exit 1; }
exec 9>>"$TASK_SOURCE/.prepare.lock"
flock -n 9 || { printf 'Another preparation writer holds the output lock. No copy started.\n' >&2; exit 1; }

PYTHONDONTWRITEBYTECODE=1 "$TASK_PYTHON" - "$TASK_SSD_MOUNT" <<'PY'
import shutil, sys
free = shutil.disk_usage(sys.argv[1]).free
if free < 200 * 1024**3:
    raise SystemExit('Need at least 200 GiB free on the SSD for the complete cache')
print('SSD free space: %.1f GiB' % (free / 1024**3))
PY

# A marker protects against accidentally copying into an unrelated existing folder.
if [ -e "$TASK_SSD_BASE" ]; then
    [ -f "$TASK_SSD_BASE/.source_experiment" ] && [ "$(cat "$TASK_SSD_BASE/.source_experiment")" = "$TASK_ROOT" ] || {
        printf 'SSD destination belongs to an unknown setup. No changes made.\n' >&2; exit 1;
    }
else
    mkdir -p -- "$TASK_SSD_BASE"
    printf '%s\n' "$TASK_ROOT" > "$TASK_SSD_BASE/.source_experiment"
fi
mkdir -p -- "$TASK_DEST"
printf 'Copying the partial cache and resume journal to NVMe...\n'
rsync -a --info=progress2 -- "$TASK_SOURCE/" "$TASK_DEST/"
printf '\nVerifying copied file contents; this adds a one-time HDD read...\n'
# --delete is DRY-RUN ONLY: unexpected extra destination files are reported, never deleted.
TASK_DIFF="$(rsync -acn --delete --itemize-changes -- "$TASK_SOURCE/" "$TASK_DEST/")"
if [ -n "$TASK_DIFF" ]; then
    printf 'Copy verification found differences. The HDD folder is unchanged:\n%s\n' "$TASK_DIFF" >&2
    exit 1
fi

TASK_BACKUP="$TASK_ROOT/prepared_hdd_backup_$(date -u +%Y%m%dT%H%M%SZ)"
[ ! -e "$TASK_BACKUP" ] || { printf 'Backup path already exists; retry after one second.\n' >&2; exit 1; }
mv -- "$TASK_SOURCE" "$TASK_BACKUP"
if ! ln -s -- "$TASK_DEST" "$TASK_SOURCE"; then
    # Never merge a backup into a concurrently created directory.
    if [ ! -e "$TASK_SOURCE" ] && [ ! -L "$TASK_SOURCE" ]; then
        mv -- "$TASK_BACKUP" "$TASK_SOURCE"
    fi
    printf 'Could not create the link. Original data remains at %s (or restored prepared/).\n' "$TASK_BACKUP" >&2
    exit 1
fi
printf '\nSSD relocation complete.\n  Active cache: %s\n  HDD backup:  %s\n' "$TASK_DEST" "$TASK_BACKUP"
printf 'Next, from the experiment folder: bash run.sh prepare --resume --workers 4\n'
