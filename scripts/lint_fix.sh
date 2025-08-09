#!/usr/bin/env bash
set -euo pipefail

PKGS="${1:-.}"

echo "== Black format =="
black $PKGS

echo "== Ruff autofix (safe rules only) =="
ruff $PKGS --fix

echo "== Ruff isort pass =="
ruff $PKGS --select I --fix

echo "== Ruff remaining =="
ruff $PKGS || true

echo "== mypy (informational) =="
mypy $PKGS || true

echo "== pylint (informational) =="
pylint $PKGS || true
