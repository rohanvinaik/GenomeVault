#!/bin/bash

# GenomeVault Cleanup Script
# This script helps identify and optionally remove duplicate GenomeVault repositories

echo "🧹 GenomeVault Duplicate Cleanup"
echo "================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${YELLOW}🔍 Finding all GenomeVault repositories...${NC}"
echo ""

# Find all directories containing GenomeVault
REPOS=(
    "/Users/rohanvinaik/genomevault"
    "/Users/rohanvinaik/Desktop/GenomeVault"
    "/Users/rohanvinaik/Desktop/genomevault-testing/TailChasingFixer"
    "/Users/rohanvinaik/Downloads/GenomeVault-main"
)

echo -e "${BLUE}Found the following potential GenomeVault locations:${NC}"
for repo in "${REPOS[@]}"; do
    if [ -d "$repo" ]; then
        echo -n "  📁 $repo"
        if [ -d "$repo/.git" ]; then
            echo -e " ${GREEN}(Git repository)${NC}"
            # Get last commit date if it's a git repo
            cd "$repo" 2>/dev/null && LAST_COMMIT=$(git log -1 --format="%cr" 2>/dev/null || echo "unknown")
            [ "$LAST_COMMIT" != "unknown" ] && echo "     Last commit: $LAST_COMMIT"
        else
            echo -e " ${YELLOW}(Not a Git repository)${NC}"
        fi
        # Show size
        SIZE=$(du -sh "$repo" 2>/dev/null | cut -f1)
        echo "     Size: $SIZE"
        echo ""
    fi
done

echo ""
echo -e "${GREEN}✅ The main repository is at: /Users/rohanvinaik/genomevault${NC}"
echo ""
echo -e "${YELLOW}📌 Recommendations:${NC}"
echo "1. Keep using /Users/rohanvinaik/genomevault as your main repository"
echo "2. The Desktop versions can be removed if they're duplicates"
echo "3. The Downloads version is likely just a downloaded archive"
echo ""
echo "To remove duplicates (after confirming they're not needed):"
echo "  rm -rf /Users/rohanvinaik/Desktop/GenomeVault"
echo "  rm -rf /Users/rohanvinaik/Downloads/GenomeVault-main"
echo ""
echo -e "${RED}⚠️  Make sure you've pushed all changes before removing any directories!${NC}"
