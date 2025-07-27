#!/bin/bash
# Push audit checklist implementation to clean-slate branch

set -e  # Exit on error

echo "=========================================="
echo "Pushing Audit Checklist Changes to GitHub"
echo "=========================================="

cd /Users/rohanvinaik/genomevault

# Check we're on the correct branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "clean-slate" ]; then
    echo "Error: Not on clean-slate branch (currently on $CURRENT_BRANCH)"
    echo "Run: git checkout clean-slate"
    exit 1
fi

echo "✓ On clean-slate branch"

# Show status summary
echo -e "\nGit Status Summary:"
echo "==================="
MODIFIED=$(git status --porcelain | grep '^M' | wc -l | tr -d ' ')
ADDED=$(git status --porcelain | grep '^A' | wc -l | tr -d ' ')
UNTRACKED=$(git status --porcelain | grep '^??' | wc -l | tr -d ' ')
DELETED=$(git status --porcelain | grep '^D' | wc -l | tr -d ' ')

echo "Modified files: $MODIFIED"
echo "Added files: $ADDED" 
echo "Untracked files: $UNTRACKED"
echo "Deleted files: $DELETED"

# Show key new files
echo -e "\nKey new files:"
echo "- pyproject.toml (updated to Hatch)"
echo "- ruff.toml (new linter config)"
echo "- pytest.ini (test config)"
echo "- genomevault/logging_utils.py"
echo "- genomevault/exceptions.py"
echo "- scripts/implement_checklist.py"
echo "- scripts/validate_checklist.py"
echo "- AUDIT_CHECKLIST_GUIDE.md"

# Add all changes
echo -e "\nAdding all changes..."
git add .

# Show what will be committed
echo -e "\nFiles to be committed:"
git status --short | head -20
if [ $(git status --short | wc -l) -gt 20 ]; then
    echo "... and $(( $(git status --short | wc -l) - 20 )) more files"
fi

# Commit with descriptive message
echo -e "\nCommitting changes..."
git commit -m "Implement audit checklist improvements

- Convert to Hatch build system (pyproject.toml)
- Add ruff linter/formatter configuration
- Update mypy.ini for stricter type checking
- Add pytest.ini with coverage requirements
- Create CI workflow for Python 3.10/3.11
- Add logging utilities (genomevault/logging_utils.py)
- Create exception hierarchy (genomevault/exceptions.py)
- Update pre-commit to use ruff
- Change license from Apache 2.0 to MIT
- Add implementation and validation scripts
- Add missing __init__.py files throughout package
- Create comprehensive implementation guide

This implements all items from the audit checklist to improve
code quality, maintainability, and development workflow."

echo "✓ Changes committed"

# Push to remote
echo -e "\nPushing to origin/clean-slate..."
git push origin clean-slate

echo -e "\n✅ Successfully pushed to clean-slate branch!"
echo "View at: https://github.com/rohanvinaik/GenomeVault/tree/clean-slate"

# Show recent commits
echo -e "\nRecent commits on clean-slate:"
git log --oneline -5
