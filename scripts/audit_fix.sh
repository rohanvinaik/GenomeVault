#!/usr/bin/env bash
set -euo pipefail

OUTDIR=".audit"
mkdir -p "$OUTDIR"

echo "== 0) Info =="
git status -sb | tee "$OUTDIR/git_status.txt" || true
git branch -vv | tee "$OUTDIR/git_branch.txt" || true

echo "== 1) Syntax check (compileall) =="
git ls-files '*.py' -z | xargs -0 -I{} python -m py_compile {} \
  2> "$OUTDIR/syntax_errors.txt" || true

echo "== 2) Ruff report (no changes yet) =="
ruff check . > "$OUTDIR/ruff_before.txt" || true

echo "== 3) Autofix (Ruff + isort via Ruff) =="
git ls-files '*.py' -z | xargs -0 ruff check --fix
git ls-files '*.py' -z | xargs -0 ruff check --select I --fix

echo "== 4) Black format =="
git ls-files '*.py' -z | xargs -0 black --quiet

echo "== 5) Ruff after =="
ruff check . > "$OUTDIR/ruff_after.txt" || true

echo "== 6) Light mypy (non-blocking) =="
mypy . > "$OUTDIR/mypy.txt" || true

echo "== 7) Smoke tests (best effort) =="
if [ -d tests/smoke ]; then
  pytest -q tests/smoke > "$OUTDIR/pytest_smoke.txt" || true
else
  pytest -q -k "health or vectors or proofs" > "$OUTDIR/pytest_smoke.txt" || true
fi

echo "== 8) Summarize =="
{
  echo "# Audit Summary"
  echo
  echo "## Syntax Errors"
  if [ -s "$OUTDIR/syntax_errors.txt" ]; then
    cat "$OUTDIR/syntax_errors.txt"
  else
    echo "None"
  fi
  echo
  echo "## Ruff Issues Before"
  tail -n +1 "$OUTDIR/ruff_before.txt" | sed -n '1,120p'
  echo
  echo "## Ruff Issues After"
  tail -n +1 "$OUTDIR/ruff_after.txt" | sed -n '1,120p'
  echo
  echo "## Mypy (light)"
  sed -n '1,120p' "$OUTDIR/mypy.txt"
  echo
  echo "## Pytest Smoke"
  sed -n '1,120p' "$OUTDIR/pytest_smoke.txt"
} > "$OUTDIR/REPORT.md"

echo
echo "== Done =="
echo "Report: $OUTDIR/REPORT.md"
