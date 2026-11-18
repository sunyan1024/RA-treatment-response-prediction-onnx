#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PY="${RA_PREDICT_PYTHON:-$ROOT/.venv/bin/python}"
if [ ! -x "$PY" ]; then
    echo "Python environment missing. Run scripts/install.sh --python /path/to/python3.9 in this skill." >&2
    exit 2
fi
exec "$PY" -B "$ROOT/scripts/launch.py" "$@"
