#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "Usage: bash scripts/install.sh [--python /path/to/python3.9]"
    echo "Installs runtime dependencies in this skill's .venv; model files are downloaded separately."
    exit 0
fi
if [ "$#" -eq 2 ] && [ "$1" = "--python" ]; then
    PY="$2"
elif [ "$#" -eq 0 ]; then
    PY=$(command -v python3.9 || command -v python || true)
else
    echo "Usage: bash scripts/install.sh [--python /path/to/python3.9]" >&2
    exit 2
fi
if [ -z "$PY" ]; then echo "Install Python 3.9 first." >&2; exit 2; fi
"$PY" -c 'import platform,sys; assert sys.version_info[:2]==(3,9), "Python 3.9 required"; assert platform.system()=="Linux" and platform.machine()=="x86_64", "Linux x86_64 required"'
if [ ! -x "$ROOT/.venv/bin/python" ]; then "$PY" -m venv "$ROOT/.venv"; fi
"$ROOT/.venv/bin/python" -c 'import sys; assert sys.version_info[:2]==(3,9), "Existing .venv must use Python 3.9"'
mkdir -p "$ROOT/models"
cd "$ROOT"
"$ROOT/.venv/bin/python" -m pip install -r requirements.txt

"$ROOT/.venv/bin/python" -m pip check
echo "Environment ready. Download the five model files listed in README.md."
echo "Check downloaded fold_* files: python scripts/verify_weights.py"
echo "Predict: bash scripts/run.sh --patient /absolute/patient.json --out report.md --json-out result.json"
