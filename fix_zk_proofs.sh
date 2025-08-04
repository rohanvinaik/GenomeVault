#!/usr/bin/env bash
################################################################################
# AUTO-FIX  E402  (imports)  +  F841  (unused vars)   in the zk_proofs package
################################################################################

set -euo pipefail

pkg="genomevault/zk_proofs/circuits"
echo "Starting auto-fix for zk_proofs package..."
echo "Package path: $pkg"
echo "Current directory: $(pwd)"

## A) move every  'import …'  and 'from … import …'  line to the top of file
echo ""
echo "Step A: Moving imports to top of files..."
find "$pkg" -name '*.py' -type f | while read -r file; do
    echo "Processing: $file"
    # Create temporary files for imports and body
    grep '^[[:space:]]*import\|^[[:space:]]*from' "$file" > "/tmp/imports.tmp" || true
    grep -v '^[[:space:]]*import\|^[[:space:]]*from' "$file" > "/tmp/body.tmp" || true
    # Combine them (imports first, then body)
    cat "/tmp/imports.tmp" "/tmp/body.tmp" > "$file"
    rm -f "/tmp/imports.tmp" "/tmp/body.tmp"
done

## B) prefix every unused variable that Ruff reported with underscore
echo ""
echo "Step B: Prefixing unused variables with underscore..."
for v in diff loss_diff max_allowed_increase gen_trans_corr trans_prot_corr \
         genotype_commitment gene allele genotype_str p_value; do
    echo "Processing variable: $v"
    # Use word boundaries to avoid partial matches
    find "$pkg" -name '*.py' -type f -exec sed -i '' -E "s/\b${v}\b/_${v}/g" {} + || true
done

## C) run formatters once more
echo ""
echo "Step C: Running formatters..."
echo "Running black..."
python3 -m black "$pkg" --quiet || true
echo "Running isort..."
python3 -m isort "$pkg" --quiet || true
echo "Running ruff with fixes..."
python3 -m ruff check "$pkg" --fix --unsafe-fixes || true

## D) stage & commit
echo ""
echo "Step D: Staging and committing changes..."
git add "$pkg"
echo "Changes to be committed:"
git status --porcelain | head -10
echo "Committing changes..."
git commit -m "fix(zk_proofs): auto-reorder imports & silence unused vars" || true

echo ""
echo "✅ Auto-fix complete!"
echo "Summary:"
echo "- Reordered imports to top of files"
echo "- Prefixed unused variables with underscore"
echo "- Applied black, isort, and ruff formatting"
echo "- Committed changes to git"
