#!/bin/bash

# GenomeVault GitHub Push Script
# This script will push the correct version to GitHub

set -e  # Exit on error

echo "🧬 GenomeVault GitHub Push Script"
echo "================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Navigate to the correct genomevault directory
cd /Users/rohanvinaik/genomevault

echo -e "${BLUE}📍 Working directory: $(pwd)${NC}"
echo ""

# 1. Show current repository state
echo -e "${YELLOW}📊 Current Repository State:${NC}"
echo -e "Branch: ${GREEN}$(git branch --show-current)${NC}"
echo -e "Remote: ${GREEN}$(git remote get-url origin 2>/dev/null || echo 'No remote set')${NC}"
echo ""

# 2. Check for uncommitted changes
echo -e "${YELLOW}📝 Checking for uncommitted changes...${NC}"
if [[ -n $(git status --porcelain) ]]; then
    echo -e "${RED}Found uncommitted changes!${NC}"
    echo ""
    echo "Summary of changes:"
    git status --short | head -20
    TOTAL_CHANGES=$(git status --short | wc -l | tr -d ' ')
    if [ $TOTAL_CHANGES -gt 20 ]; then
        echo "... and $(( $TOTAL_CHANGES - 20 )) more files"
    fi
    echo ""

    # Clean up unnecessary files
    echo -e "${YELLOW}🧹 Cleaning up unnecessary files...${NC}"
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "*.pyo" -delete 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name ".DS_Store" -delete 2>/dev/null || true
    find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name ".mypy_cache" -type d -exec rm -rf {} + 2>/dev/null || true

    # Stage all changes
    echo -e "${YELLOW}📦 Staging all changes...${NC}"
    git add -A

    # Create commit
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    git commit -m "chore: Clean up and prepare for push - $TIMESTAMP" || echo "No changes to commit after cleanup"
    echo -e "${GREEN}✓ Changes committed${NC}"
else
    echo -e "${GREEN}✓ No uncommitted changes${NC}"
fi

echo ""

# 3. Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${YELLOW}🌿 Current branch: ${GREEN}$CURRENT_BRANCH${NC}"

# 4. Fetch latest from remote
echo -e "\n${YELLOW}🔄 Fetching latest from remote...${NC}"
git fetch origin || echo "Could not fetch from origin"

# 5. Check if we need to pull
echo -e "\n${YELLOW}🔍 Checking remote status...${NC}"
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u} 2>/dev/null || echo "none")

if [ "$REMOTE" = "none" ]; then
    echo -e "${YELLOW}⚠️  No upstream branch set${NC}"
    NEED_UPSTREAM=true
elif [ "$LOCAL" = "$REMOTE" ]; then
    echo -e "${GREEN}✓ Branch is up to date with remote${NC}"
    NEED_UPSTREAM=false
elif git merge-base --is-ancestor $LOCAL $REMOTE; then
    echo -e "${YELLOW}⚠️  Remote has changes. Pulling...${NC}"
    git pull --rebase origin $CURRENT_BRANCH
    NEED_UPSTREAM=false
else
    echo -e "${GREEN}✓ Local is ahead of remote${NC}"
    NEED_UPSTREAM=false
fi

# 6. Push to GitHub
echo -e "\n${YELLOW}🚀 Pushing to GitHub...${NC}"

if [ "$NEED_UPSTREAM" = true ]; then
    echo -e "${BLUE}Setting upstream and pushing...${NC}"
    git push -u origin $CURRENT_BRANCH
else
    echo -e "${BLUE}Pushing to origin/$CURRENT_BRANCH...${NC}"
    git push origin $CURRENT_BRANCH
fi

# 7. Show result
echo ""
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Successfully pushed to GitHub!${NC}"
    echo ""
    echo -e "${BLUE}📎 Repository URL:${NC} https://github.com/rohanvinaik/GenomeVault"
    echo -e "${BLUE}🔍 View on GitHub:${NC} https://github.com/rohanvinaik/GenomeVault/tree/$CURRENT_BRANCH"
    echo ""

    # Show recent commits
    echo -e "${YELLOW}📜 Recent commits pushed:${NC}"
    git log --oneline -5
else
    echo -e "${RED}❌ Push failed!${NC}"
    echo "Please check the error messages above."
fi

echo ""
echo "Done! 🎉"
