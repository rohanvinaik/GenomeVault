#!/bin/bash

# Quick Git Status Check

cd /Users/rohanvinaik/genomevault

echo "=== Git Repository Status ==="
echo ""
echo "📍 Current branch: $(git branch --show-current)"
echo ""
echo "📋 Status summary:"
git status --short | head -20
echo ""

# Check if our audit files exist but aren't committed
echo "🔍 Checking for audit files:"
for file in fix_audit_issues.py validate_project_only.py audit_menu_final.sh REAL_AUDIT_STATUS.md; do
    if [ -f "$file" ]; then
        if git ls-files --error-unmatch "$file" >/dev/null 2>&1; then
            echo "✓ $file (tracked)"
        else
            echo "✗ $file (UNTRACKED - needs to be added)"
        fi
    else
        echo "? $file (not found)"
    fi
done

echo ""
echo "📊 Summary:"
echo "- Untracked files: $(git ls-files --others --exclude-standard | wc -l)"
echo "- Modified files: $(git diff --name-only | wc -l)"
echo "- Staged files: $(git diff --cached --name-only | wc -l)"
echo ""
echo "💡 If files are untracked, run: ./commit_and_push_now.sh"
