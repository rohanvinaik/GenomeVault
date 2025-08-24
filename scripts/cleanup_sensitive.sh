#!/bin/bash
set -e

echo "🔒 Cleaning up sensitive files from repository"
echo "============================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Files to remove
SENSITIVE_FILES=(
    ".zk_transcript_key"
    "*.db"
    "*.sqlite"
    "*.sqlite3"
    "history.db"
    ".python_history"
    "*.key"
    "*.pem"
    "*.p12"
    ".env"
    ".env.local"
    "*_secret*"
    "*_private*"
)

# Remove sensitive files
echo -e "${YELLOW}Removing sensitive files...${NC}"
for pattern in "${SENSITIVE_FILES[@]}"; do
    # Remove from working directory
    find . -name "$pattern" -type f -exec rm -f {} \; 2>/dev/null || true
    
    # Remove from git cache if tracked
    git rm --cached $(find . -name "$pattern" -type f 2>/dev/null | head -20) 2>/dev/null || true
done

# Specifically remove known sensitive files
git rm --cached .zk_transcript_key 2>/dev/null || true
git rm --cached "*.db" 2>/dev/null || true
git rm --cached "history.db" 2>/dev/null || true

echo -e "${GREEN}✅ Sensitive files cleaned up${NC}"
echo ""
echo "Next steps:"
echo "  1. Review git history for any remaining sensitive data"
echo "  2. Rotate any exposed credentials immediately"
echo "  3. Update CI/CD with new key management"
echo ""
echo "To verify cleanup:"
echo "  git ls-files | grep -E '\.db$|\.key$|\.env'"