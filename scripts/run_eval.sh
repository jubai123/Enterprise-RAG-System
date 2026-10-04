#!/usr/bin/env bash
# One-click evaluation entry point for the GitHub N39 development set.
#
# Bootstraps a Python venv with the GPU dependency set (torch 2.5.1+cu121)
# if needed, then hands off to scripts/run_eval.py.
#
# Snapshot workflow (reuse the existing index instead of rebuilding):
#   # on the box with the populated Qdrant index:
#   ./scripts/run_eval.sh --create-snapshot snapshots/enterprise_rag_bench.snapshot
#
#   # on the target box (e.g. ModelScope GPU), upload the .snapshot file and:
#   ./scripts/run_eval.sh --start-qdrant ./qdrant \
#       --snapshot snapshots/enterprise_rag_bench.snapshot
#   (restores the collection when the index is missing/mismatched; falls back
#    to a rebuild if the snapshot is missing or has a different point count)
#
# Usage:
#   ./scripts/run_eval.sh                 # run the full pipeline
#   ./scripts/run_eval.sh --help          # show orchestrator options
#   ./scripts/run_eval.sh --setup         # (re)install deps, then run
#   ./scripts/run_eval.sh --limit 3       # smoke-test with 3 questions
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* || "$(uname -s)" == CYGWIN* ]]; then
    VPY="$ROOT/.venv/Scripts/python.exe"
    BASE_PY="${PYTHON:-python}"
else
    VPY="$ROOT/.venv/bin/python"
    BASE_PY="${PYTHON:-python3}"
fi

FORCE_SETUP=0
PASSTHROUGH=()
for arg in "$@"; do
    if [[ "$arg" == "--setup" ]]; then
        FORCE_SETUP=1
    else
        PASSTHROUGH+=("$arg")
    fi
done

NEED_SETUP=0
if [[ ! -x "$VPY" ]]; then
    echo "[run_eval] creating virtualenv at $ROOT/.venv"
    "$BASE_PY" -m venv "$ROOT/.venv"
    NEED_SETUP=1
fi
if [[ "$FORCE_SETUP" -eq 1 ]]; then
    NEED_SETUP=1
fi

if [[ "$NEED_SETUP" -eq 1 ]]; then
    echo "[run_eval] installing GPU dependencies (requirements-rest.txt + torch==2.5.1+cu121)"
    "$VPY" -m pip install --upgrade pip
    "$VPY" -m pip install -r "$ROOT/requirements-rest.txt"
    "$VPY" -m pip install torch==2.5.1+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121
    echo "[run_eval] dependency install complete"
fi

exec "$VPY" -m scripts.run_eval "${PASSTHROUGH[@]}"
