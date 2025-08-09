#!/bin/bash

echo "🚀 SIMPLE PUSH TO GITHUB - CLEAN-SLATE BRANCH"
echo "============================================"

cd /Users/rohanvinaik/genomevault

# First, let's see what we have
echo "Files to be committed:"
ls -1 | grep -E "^(fix_|validate_|audit_|AUDIT|REAL_|QUICK_|preflight|generate_)" | grep -E "\.(py|sh|md|txt)$"

echo ""
echo "Adding files..."
git add fix_*.py validate_*.py *.check.py *comparison*.py *.sh AUDIT*.md REAL_*.md QUICK_*.txt PUSH_*.md MANUAL_*.md

echo ""
echo "Committing..."
git commit -m "Add audit fix scripts and docs - project is healthy (334 files, not 45k)"

echo ""
echo "Pushing to clean-slate..."
git push origin HEAD:clean-slate

echo ""
echo "✅ DONE!"
echo ""
echo "View at: https://github.com/rohanvinaik/GenomeVault/tree/clean-slate"
