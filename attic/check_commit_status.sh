#!/bin/bash

echo "🔍 Checking Git Status After Pre-commit Hook Issues"
echo "=================================================="

cd /Users/rohanvinaik/genomevault

echo "1. Current git status:"
git status

echo ""
echo "2. Last 5 commits:"
git log --oneline -5

echo ""
echo "3. Check if audit files are tracked:"
for file in fix_audit_issues.py validate_project_only.py audit_menu_final.sh; do
    if git ls-files --error-unmatch "$file" >/dev/null 2>&1; then
        echo "✓ $file is tracked in git"
    else
        echo "✗ $file is NOT tracked"
    fi
done

echo ""
echo "4. Local vs Remote comparison:"
echo "Local clean-slate:"
git log --oneline -1 clean-slate 2>/dev/null || echo "No local clean-slate branch"
echo ""
echo "Remote clean-slate:"
git log --oneline -1 origin/clean-slate 2>/dev/null || echo "No remote clean-slate branch"

echo ""
echo "5. Uncommitted changes:"
git diff --name-only | head -10

echo ""
echo "To bypass pre-commit hooks and force push:"
echo "./bypass_hooks_push.sh"
