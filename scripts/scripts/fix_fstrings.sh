#!/usr/bin/env bash
# scripts/fix_fstrings.sh
set -euo pipefail

# 1. Collect the files Black still can’t parse
readarray -t bad_files < <(
  black --check --diff . 2>&1 \
    | grep -oE 'cannot format .*\.py' \
    | awk '{print $3}' | sort -u
)

[[ ${#bad_files[@]} -eq 0 ]] && { echo "Black already clean 👍"; exit 0; }

echo "Fixing ${#bad_files[@]} files…"

# 2. Heuristic repairs
for f in "${bad_files[@]}"; do
  perl -0777 -pi -e '
    # Add missing } before closing quote
    s/f"([^"\n]*\{[^}"\n]*)(\n?")/f"$1}"$2/g;

    # Add }" for lines like  raise ValueError(f"...{var")
    s/f"([^"\n]*\{[^}"\n]*)"/f"$1}"/g;

    # If a line ends with  {xxx"\n)   → add })
    s/(\{\w+)"\n\s*\)/$1}")/g;

    # Balance !s
    s/\{([^}!\n]+)!s(?=["}])/\{$1!s}/g;
  ' "$f"
done

# 3. Format once more
python -m black .    --quiet
python -m isort .    --quiet
ruff check . --fix   --quiet

# 4. Stage only changed files and commit if hooks pass
git add "${bad_files[@]}"
pre-commit run --all-files
git commit -m "fix: auto-balance remaining f-strings and calls; formatter clean"
git push origin clean-slate
echo "✅ All hooks green & pushed"
