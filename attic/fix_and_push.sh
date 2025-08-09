#!/bin/bash

# GenomeVault Repository Fix and Push Script
# This script helps recover and push the repository to GitHub

set -e  # Exit on error

echo "🧬 GenomeVault Repository Fix & Push"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Navigate to project directory
cd /Users/rohanvinaik/genomevault

# 1. Check current git status
echo -e "${YELLOW}📊 Checking repository status...${NC}"
echo "Current branch: $(git branch --show-current)"
echo "Remote URL: $(git remote get-url origin 2>/dev/null || echo 'No remote set')"

# 2. Show summary of changes
echo -e "\n${YELLOW}📝 Summary of changes:${NC}"
CHANGES=$(git status --porcelain | wc -l | tr -d ' ')
echo "Total files changed: $CHANGES"

# Show file types changed
echo -e "\nFile types affected:"
git status --porcelain | awk '{print $2}' | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -10

# 3. Check for large files
echo -e "\n${YELLOW}🔍 Checking for large files...${NC}"
find . -type f -size +10M -not -path "./.git/*" -not -path "./venv/*" -not -path "./.venv/*" | while read file; do
    SIZE=$(du -h "$file" | cut -f1)
    echo -e "${RED}Large file found: $file ($SIZE)${NC}"
done

# 4. Clean up common issues
echo -e "\n${YELLOW}🧹 Cleaning up repository...${NC}"

# Remove Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name ".DS_Store" -delete 2>/dev/null || true

# Remove pytest cache
rm -rf .pytest_cache 2>/dev/null || true

echo "Cleanup complete!"

# 5. Update .gitignore if needed
echo -e "\n${YELLOW}📋 Updating .gitignore...${NC}"
if ! grep -q "__pycache__" .gitignore 2>/dev/null; then
    echo -e "\n# Python" >> .gitignore
    echo "__pycache__/" >> .gitignore
    echo "*.pyc" >> .gitignore
    echo "*.pyo" >> .gitignore
    echo ".pytest_cache/" >> .gitignore
fi

if ! grep -q ".DS_Store" .gitignore 2>/dev/null; then
    echo -e "\n# macOS" >> .gitignore
    echo ".DS_Store" >> .gitignore
fi

# 6. Stage all changes
echo -e "\n${YELLOW}📦 Staging changes...${NC}"
git add -A

# 7. Show what will be committed
echo -e "\n${YELLOW}📊 Files to be committed:${NC}"
git status --short | head -20
if [ $(git status --short | wc -l) -gt 20 ]; then
    echo "... and $(( $(git status --short | wc -l) - 20 )) more files"
fi

# 8. Create commit
echo -e "\n${YELLOW}💾 Creating commit...${NC}"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
COMMIT_MSG="fix: Repository recovery and cleanup - $TIMESTAMP

- Cleaned up repository structure
- Removed cache and temporary files
- Updated .gitignore
- Fixed file permissions
- Prepared for GitHub push

Files changed: $CHANGES"

# Check if there are changes to commit
if [ -n "$(git status --porcelain)" ]; then
    git commit -m "$COMMIT_MSG"
    echo -e "${GREEN}✓ Commit created successfully${NC}"
else
    echo -e "${GREEN}✓ No changes to commit -
