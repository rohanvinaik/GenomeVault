#!/bin/bash

# Simple and direct push to GitHub

echo "🚀 GenomeVault - Direct Push to GitHub"
echo "======================================"

cd /Users/rohanvinaik/genomevault

# Show what we're about to do
echo "This script will:"
echo "1. Add all audit fix files"
echo "2. Commit them with a descriptive message"
echo "3. Push to the clean-slate branch"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Step 1: Adding files..."

    # Add all audit-related files (use -f to force add even if in .gitignore)
    git add fix_*.py
    git add validate_*.py
    git add preflight_check.py
    git add quick_fix_init_files.py
    git add generate_comparison_report.py
    git add *audit*.sh
    git add check_git_status.sh
    git add commit_and_push_now.sh
    git add AUDIT*.md
    git add REAL_AUDIT_STATUS.md
    git add QUICK_REFERENCE.txt
    git add PUSH_SUMMARY.md

    echo ""
    echo "Step 2: Showing what will be committed..."
    git status --short

    echo ""
    echo "Step 3: Committing..."
    git commit -m "Add audit fix scripts and documentation

These scripts help analyze and fix code quality issues:
- validate_project_only.py: Shows real metrics (excludes venv)
- fix_targeted_issues.py: Applies smart fixes
- audit_menu_final.sh: Interactive menu
- Documentation shows the code is actually in good shape"

    echo ""
    echo "Step 4: Handling branch..."

    # Check current branch
    current_branch=$(git branch --show-current)
    echo "Current branch: $current_branch"

    if [ "$current_branch" != "clean-slate" ]; then
        echo "Switching to clean-slate branch..."
        git checkout -b clean-slate 2>/dev/null || git checkout clean-slate
    fi

    echo ""
    echo "Step 5: Pushing to GitHub..."
    git push -u origin clean-slate

    echo ""
    echo "✅ Done! Check your GitHub repository:"
    echo "https://github.com/[your-username]/genomevault/tree/clean-slate"
else
    echo "Cancelled."
fi
