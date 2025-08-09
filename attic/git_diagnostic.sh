#!/bin/bash

echo "🔍 Git Diagnostic Report"
echo "======================="
echo ""

cd /Users/rohanvinaik/genomevault

echo "1. Current location: $(pwd)"
echo "2. Current branch: $(git branch --show-current)"
echo "3. Remote info:"
git remote -v
echo ""
echo "4. Recent commits:"
git log --oneline -3
echo ""
echo "5. Files that need to be added:"
echo "---"
ls -la | grep -E "(fix_|validate_|audit_|AUDIT|REAL_|QUICK_)" | grep -v ".sh~"
echo ""
echo "6. Current git status (first 10 lines):"
git status | head -10
echo ""
echo "🚀 To push everything, run: ./one_line_push.sh"
