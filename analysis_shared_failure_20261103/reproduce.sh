#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
dinomaly2/bin/python analysis_shared_failure_20261103/run_analysis.py
dinomaly2/bin/python analysis_shared_failure_20261103/work/finalize_deliverables.py
