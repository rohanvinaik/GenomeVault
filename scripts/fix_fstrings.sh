#!/usr/bin/env bash
set -euo pipefail

# 1. collect files Black still cannot parse
readarray -t bad_files < <(
  black --check --diff . 2>&1 |
  grep -oE 'cannot format .*\.py' |
  awk '{print $3}' |
  sort -u
)

[[ ${#bad_files[@]} -eq 0 ]] && { echo "Black already clean 🎉"; exit 0; }

echo "Fixing ${#bad_files[@]} files …"

for f in "${bad_files[@]}"; do
  # A) mend extra= dict-braces + drop orphan %s tokens
  sed -i '' -E \
    -e 's/extra=\{([^}]*)\)\s*$/extra={\1}/' \
    -e 's/,\s*extra=%s//g' \
    -e '/^\s*extra=%s\s*$/d' \
    "$f"

  # B) remove %s inside any string / f-string, balance single }
  perl -0777 -pi -e '
    s/%s(?=["}])//g;                           # drop %s remnants
    s/f"([^"]*?\{[^}"\n]*)"/f"$1}"/g;         # add missing }
  ' "$f"
done

echo "Running formatters …"
black .   --quiet
isort .   --quiet
ruff check . --fix --quiet

git add "${bad_files[@]}"
pre-commit run --all-files
git commit -m "fix: auto-balance remaining f-strings & extra= syntax"
echo "✓ commit created – push with:  git push origin clean-slate"
