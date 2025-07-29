#!/usr/bin/env bash
set -euo pipefail
docker compose up -d --build api
echo "API at http://localhost:8000"
