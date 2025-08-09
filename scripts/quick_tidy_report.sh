#!/usr/bin/env bash
set -euo pipefail

OUT=".tidy"
mkdir -p "$OUT"

echo "== 1. Counting tracked files =="
git ls-files | wc -l

echo "== 2. Large files > 1MB =="
find . -type f -size +1M -not -path "./.git/*" -not -path "./venv/*" -not -path "./.venv/*" 2>/dev/null | head -20

echo "== 3. Python file count =="
git ls-files "*.py" | wc -l

echo "== 4. Test file count =="
git ls-files "tests/*.py" | wc -l

echo "== 5. Checking for common junk files =="
for pattern in "*.pyc" "__pycache__" ".DS_Store" "*.log" "*.tmp" "*.bak"; do
  COUNT=$(git ls-files "*$pattern*" 2>/dev/null | wc -l)
  [ "$COUNT" -gt 0 ] && echo "  Found $COUNT files matching $pattern"
done

echo "== 6. Directory structure summary =="
tree -d -L 2 -I "venv|.git|__pycache__|.mypy_cache|.ruff_cache" 2>/dev/null || find . -type d -maxdepth 2 -not -path "./.git*" -not -path "./venv*" | sort

echo "== 7. Recent commits =="
git log --oneline -5

echo "== DONE =="
