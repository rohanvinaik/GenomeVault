#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-SBOM.json}"
python -m pip install --upgrade pip >/dev/null
python -m pip install cyclonedx-bom==4.0.1 >/dev/null
cyclonedx-py --format json --output "$OUT"
echo "Wrote $OUT"
