#!/usr/bin/env bash
set -euo pipefail

echo "Running MVP smoke tests..."
python -m pytest tests/smoke -v --tb=short

echo ""
echo "Running basic linters..."
python -m ruff check genomevault/hdc genomevault/federated genomevault/local_processing --select=E,F
python -m mypy genomevault/hdc genomevault/federated genomevault/local_processing --ignore-missing-imports

echo ""
echo "MVP implementation validated!"
