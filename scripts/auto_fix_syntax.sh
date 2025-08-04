#!/usr/bin/env bash
# scripts/auto_fix_syntax.sh
# -----------------------------------------------------------------------------
# One-shot “rescue” script that fixes the three patterns blocking Black/Ruff.
# Works on macOS (BSD sed) and Linux (GNU sed) – autodetects which flag to use.
# -----------------------------------------------------------------------------

set -euo pipefail
#shopt -s globstar

###############################################################################
# Helper: choose sed -i syntax
###############################################################################
if sed --help 2>&1 | grep -q -- '--version'; then
  # GNU sed
  SED_INPLACE=(-i)
else
  # BSD sed (macOS)
  SED_INPLACE=(-i '')
fi

###############################################################################
echo "➜ 1. Locate files Black cannot parse ..."
###############################################################################
mapfile -t bad_files < <(
  black --check --diff . 2>&1 \
    | grep -oE 'cannot format .+?\.py' \
    | awk '{print $3}' \
    | sort -u
)

if [[ ${#bad_files[@]} -eq 0 ]]; then
  echo "All files already Black-clean 🎉"
  exit 0
fi
printf 'Found %d broken files\n' "${#bad_files[@]}"

###############################################################################
echo "➜ 2. Apply heuristic fixes ..."
###############################################################################
for f in "${bad_files[@]}"; do
  # 2-a  Fix extra= misuse
  sed "${SED_INPLACE[@]}" -E \
    -e 's/extra=%s"([A-Za-z0-9_]+)"\s*:/extra={"\1": /g' \
    -e 's/extra="([A-Za-z0-9_]+)"\s*:/extra={"\1": /g' \
    -e 's/extra=\{([^}]*)\)\s*$/extra={\1}/' \
    "$f"

  # 2-b  Drop standalone  extra=%s  tokens
  sed "${SED_INPLACE[@]}" -E \
    -e 's/,\s*extra=%s//g' \
    -e '/^\s*extra=%s\s*$/d' \
    "$f"

  # 2-c  Remove %s inside strings / f-strings, balance single } in them
  python - <<PY "$f"
import re, pathlib, sys, textwrap
p = pathlib.Path(sys.argv[1])
txt = p.read_text()
txt = re.sub(r'%s([}"])?', r'\1', txt)               # drop %s remnants
txt = re.sub(r'f"([^"]*?)[}]([^"]*?)"', r'f"\1\2"', txt)  # remove stray }
p.write_text(txt)
PY
done

###############################################################################
echo "➜ 3. Run Black, isort, Ruff (auto-fix) ..."
###############################################################################
python -m black .          --quiet
python -m isort .          --quiet
ruff check . --fix         --quiet

###############################################################################
echo "➜ 4. Verify tools are green ..."
###############################################################################
black --check . --quiet
ruff check .   --quiet
# optional mypy/pytest here if desired

###############################################################################
echo "➜ 5. Stage only changed files and commit ..."
###############################################################################
git add "${bad_files[@]}"
git diff --cached --quiet || git commit -m "chore: auto-fix syntax (extra=, %s, braces) and reformat"

echo "✓ ALL GREEN – Black/Ruff pass and fixes committed"
