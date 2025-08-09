#!/bin/bash
cd /Users/rohanvinaik/genomevault

echo "==================================="
echo "Git Push - 8-4 Audit Implementation"
echo "==================================="

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Check for uncommitted changes
if [[ -n $(git status --porcelain) ]]; then
    echo "⚠️  Uncommitted changes detected:"
    git status --short
    exit 1
fi

# Show what will be pushed
echo -e "\nCommits to be pushed:"
echo "--------------------"
git log origin/$CURRENT_BRANCH..HEAD --oneline

# Count commits
COMMIT_COUNT=$(git log origin/$CURRENT_BRANCH..HEAD --oneline | wc -l)
echo -e "\nTotal commits to push: $COMMIT_COUNT"

# Push to origin
echo -e "\nPushing to origin/$CURRENT_BRANCH..."
git push origin $CURRENT_BRANCH

if [ $? -eq 0 ]; then
    echo -e "\n✅ Push successful!"
    echo -e "\nPushed commits:"
    git log origin/$CURRENT_BRANCH..HEAD --oneline 2>/dev/null || echo "All commits pushed."
else
    echo -e "\n❌ Push failed!"
    exit 1
fi
