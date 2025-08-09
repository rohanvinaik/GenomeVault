#!/usr/bin/env bash
set -euo pipefail

PKGS="${1:-.}"

echo "== Black --check =="
black --check $PKGS

echo "== Ruff (no fix) =="
ruff $PKGS

echo "== isort (via ruff) check =="
ruff $PKGS --select I

echo "== mypy (light) =="
mypy $PKGS

echo "== pylint (report only) =="
pylint $PKGS || true
