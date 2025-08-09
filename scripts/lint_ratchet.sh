#!/usr/bin/env bash
set -euo pipefail

AREAS=(
  "genomevault/api"
  "genomevault/local_processing"
  "genomevault/hdc"
  "genomevault/pir"
  "genomevault/zk_proofs"
)

for A in "${AREAS[@]}"; do
  echo "=== Area: $A ==="
  ./scripts/lint_fix.sh "$A"
  git add -A
  git commit -m "lint: $A (black+ruff+isort, no functional changes)" || true
done
