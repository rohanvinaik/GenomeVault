#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-SBOM.json}"

# Ensure the CycloneDX CLI is available (provided by cyclonedx-bom package)
if ! command -v cyclonedx-py >/dev/null 2>&1; then
  python -m pip install -q --upgrade pip
  python -m pip install -q "cyclonedx-bom>=4.1,<7"
fi

# Decide input source based on what your repo uses
if [ -f "poetry.lock" ]; then
  echo "[SBOM] Using Poetry lockfile"
  cyclonedx-py poetry --output-format json --output-file "$OUT"

elif [ -f "Pipfile.lock" ]; then
  echo "[SBOM] Using Pipenv lockfile"
  cyclonedx-py pipenv --output-format json --output-file "$OUT"

elif [ -f "requirements.txt" ] || ls requirements*.txt >/dev/null 2>&1; then
  echo "[SBOM] Using requirements.txt"
  cyclonedx-py requirements --output-format json --output-file "$OUT"

else
  echo "[SBOM] No supported dependency file found (poetry.lock / Pipfile.lock / requirements*.txt)"
  exit 2
fi

echo "[SBOM] Wrote $OUT"
