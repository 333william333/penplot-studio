#!/usr/bin/env bash
# Start PenPlot Studio.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "Setting up the Python environment (first run only)…"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
    VIRTUAL_ENV=.venv uv pip install -r requirements.txt
  else
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
  fi
fi

exec .venv/bin/python -m penplot "$@"
