#!/bin/bash
cd /Users/rohanvinaik/genomevault

echo "==================================="
echo "Git Push Script"
echo "==================================="

# Check if we're on the right branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Check for uncommitted changes
if [[ -n $(git status --porcelain) ]]; then
    echo "⚠️  Warning: You have uncommitted changes:"
    git status --short
    echo ""
    read -p "Do you want to commit these changes first? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add -A
        read -p "Enter commit message: " commit_msg
        git commit -m "$commit_msg"
    fi
fi

# Show what will be pushed
echo -e "\nCommits to be pushed:"
echo "--------------------"
git log origin/$CURRENT_BRANCH..HEAD --oneline

# Confirm push
echo ""
read -p "Push to origin/$CURRENT_BRANCH? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Pushing to origin/$CURRENT_BRANCH..."
    git push origin $CURRENT_BRANCH

    if [ $? -eq 0 ]; then
        echo "✅ Push successful!"
        echo ""
        echo "Summary of pushed changes:"
        echo "-------------------------"
        git log origin/$CURRENT_BRANCH..HEAD --oneline
    else
        echo "❌ Push failed. Please check the error above."
    fi
else
    echo "Push cancelled."
fi
