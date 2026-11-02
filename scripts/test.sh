#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python || ! -d frontend/node_modules ]]; then
  echo "Dependencies are missing. Run: bash scripts/setup.sh" >&2
  exit 1
fi

.venv/bin/python -m pytest -q backend/tests
npm --prefix frontend run build
