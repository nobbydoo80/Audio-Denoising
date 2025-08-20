#!/usr/bin/env bash
set -euo pipefail

# Simple runner for core pipeline on POSIX systems
REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}

CFG=${1:-"$REPO_ROOT/configs/full_no_ai.yaml"}

$PY "$REPO_ROOT/src/pipeline_v2.py" --config "$CFG" --phases all "$@"
