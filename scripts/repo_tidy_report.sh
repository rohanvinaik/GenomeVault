#!/usr/bin/env bash
set -euo pipefail

OUT=".tidy"
mkdir -p "$OUT"

echo "== Track only versioned files list ==" | tee "$OUT/00_git_files.txt"
git ls-files > "$OUT/00_git_files.txt"

echo "== Big files (tracked) > 2MB ==" | tee "$OUT/01_big_files.txt"
git ls-files | while read -r f; do
  if [ -f "$f" ]; then
    SIZE=$(wc -c < "$f" 2>/dev/null || echo 0)
    [ "$SIZE" -gt 2000000 ] && echo "$SIZE  $f"
  fi
done | sort -nr > "$OUT/01_big_files.txt" || true

echo "== Obvious junk patterns ==" | tee "$OUT/02_junk_patterns.txt"
printf "%s\n" \
  "*.pyc" "__pycache__/" ".pytest_cache/" ".ruff_cache/" ".mypy_cache/" \
  ".ipynb_checkpoints/" "node_modules/" "dist/" "build/" "*.egg-info/" \
  ".DS_Store" "*.log" "*.tmp" "*.bak" ".env" ".env.*" ".coverage*" \
  > "$OUT/02_patterns.txt"
grep -Rni -f "$OUT/02_patterns.txt" --null -l . 2>/dev/null | sed 's|^\./||' | sort -u > "$OUT/02_junk_patterns.txt" || true

echo "== Jupyter notebooks not touched in 90 days ==" | tee "$OUT/03_old_notebooks.txt"
git ls-files "*.ipynb" | while read -r f; do
  TS=$(git log -1 --format=%ct -- "$f" 2>/dev/null || echo 0)
  NOW=$(date +%s); AGE=$(( (NOW-TS)/86400 ))
  [ "$AGE" -ge 90 ] && echo "$AGE d  $f"
done | sort -nr > "$OUT/03_old_notebooks.txt"

echo "== Tracked but ignored by .gitignore (waste) ==" | tee "$OUT/04_tracked_but_should_ignore.txt"
git check-ignore -v $(git ls-files) 2>/dev/null | awk '{print $3}' | sort -u > "$OUT/04_tracked_but_should_ignore.txt" || true

echo "== Python files not imported anywhere (dead candidates) ==" | tee "$OUT/05_dead_python_candidates.txt"
# naive import graph using ripgrep fallback to grep
if command -v rg >/dev/null; then
  RG="rg"
else
  RG="grep -R"
fi
git ls-files "*.py" | while read -r f; do
  MOD="${f%.py}"
  MOD="${MOD#*/}"   # rough: strip top dir for import match
  # Skip obvious entrypoints and tests
  case "$f" in
    tests/*|scripts/*|**/__init__.py) continue ;;
  esac
  # Look for 'import x' or 'from x import'
  if ! $RG -q -e "from ${MOD//\//\\.} " -e "import ${MOD//\//\\.}" . 2>/dev/null; then
    echo "$f"
  fi
done | sort -u > "$OUT/05_dead_python_candidates.txt" || true

echo "== Unused dependencies (pipdeptree vs imports) ==" | tee "$OUT/06_unused_deps.txt"
pipdeptree --warn silence > "$OUT/pipdeptree.txt" || true
# cheap heuristic: list top-level packages not mentioned in code
python - <<'PY' > ".tidy/06_unused_deps.txt" || true
import json, subprocess, sys, re, os
try:
    out = subprocess.check_output(["pipdeptree","--json-tree"], text=True)
    pkgs = {d["package"]["key"] for d in json.loads(out)}
except Exception:
    pkgs = set()
# scan imports
code = ""
for root, dirs, files in os.walk("."):
    if any(x in root for x in (".venv","venv",".git","node_modules",".mypy_cache",".ruff_cache","dist","build")):
        continue
    for fn in files:
        if fn.endswith(".py"):
            try:
                with open(os.path.join(root, fn), "r", encoding="utf-8", errors="ignore") as fh:
                    code += fh.read()+"\n"
            except Exception:
                pass
imps = set(re.findall(r'^\s*(?:from|import)\s+([a-zA-Z0-9_]+)', code, flags=re.M))
print("\n".join(sorted(pkgs - imps)))
PY

echo "== DONE =="
ls -l "$OUT"
