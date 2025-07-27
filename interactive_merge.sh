#!/bin/bash

# Interactive merge helper for GenomeVault
# Provides options for handling conflicts

set -e

echo "🧬 GenomeVault - Interactive Merge Helper"
echo "========================================"

cd /Users/rohanvinaik/genomevault

# Function to show file differences
show_diff() {
    local file=$1
    echo -e "\n📄 Differences in $file:"
    echo "----------------------------------------"
    git diff HEAD $file | head -50
    echo "----------------------------------------"
}

# Step 1: Backup current state
echo -e "\n💾 Creating backup of your current work..."
cp README.md README.md.backup
echo "✓ Backed up README.md to README.md.backup"

# Step 2: Check what's different
echo -e "\n🔍 Checking differences with remote..."
git fetch origin main
git diff --name-only HEAD origin/main

# Step 3: Show options
echo -e "\n🤔 How would you like to proceed?"
echo ""
echo "1) Keep YOUR version of README.md and add new audit files"
echo "2) Take REMOTE version of README.md and add your audit changes"
echo "3) Manually merge README.md line by line"
echo "4) See the differences first"
echo "5) Abort and keep everything as-is"

read -p "Enter choice (1-5): " choice

case $choice in
    1)
        echo -e "\n✓ Keeping your README.md version..."

        # Stash changes
        git stash save "Temp stash for merge"

        # Pull remote
        git pull origin main --strategy-option=ours

        # Apply stash
        git stash pop

        # Add all files
        git add -A
        ;;

    2)
        echo -e "\n✓ Taking remote README.md and adding your changes..."

        # Save audit files temporarily
        mkdir -p /tmp/genomevault_backup
        cp -r genomevault/pir/server/secure_pir_server.py /tmp/genomevault_backup/ 2>/dev/null || true
        cp -r genomevault/zk_proofs/srs_manager /tmp/genomevault_backup/ 2>/dev/null || true
        cp -r genomevault/hypervector/kan/calibration /tmp/genomevault_backup/ 2>/dev/null || true
        cp -r genomevault/blockchain/consent /tmp/genomevault_backup/ 2>/dev/null || true
        cp -r tests/security /tmp/genomevault_backup/ 2>/dev/null || true
        cp AUDIT_IMPLEMENTATION_SUMMARY.md /tmp/genomevault_backup/ 2>/dev/null || true

        # Reset to remote
        git reset --hard origin/main

        # Restore audit files
        cp /tmp/genomevault_backup/secure_pir_server.py genomevault/pir/server/ 2>/dev/null || true
        cp -r /tmp/genomevault_backup/srs_manager genomevault/zk_proofs/ 2>/dev/null || true
        cp -r /tmp/genomevault_backup/calibration genomevault/hypervector/kan/ 2>/dev/null || true
        cp -r /tmp/genomevault_backup/consent genomevault/blockchain/ 2>/dev/null || true
        cp -r /tmp/genomevault_backup/security tests/ 2>/dev/null || true
        cp /tmp/genomevault_backup/AUDIT_IMPLEMENTATION_SUMMARY.md . 2>/dev/null || true

        # Add all files
        git add -A
        ;;

    3)
        echo -e "\n🔧 Manual merge selected..."

        # Create a merge commit
        git stash
        git pull origin main --no-commit --no-ff
        git stash pop

        echo "Opening your default merge tool..."
        git mergetool README.md

        # Add resolved files
        git add README.md
        ;;

    4)
        show_diff "README.md"
        echo -e "\n❓ Please run the script again after reviewing."
        exit 0
        ;;

    5)
        echo -e "\n❌ Aborting merge. Your files remain unchanged."
        exit 0
        ;;

    *)
        echo "Invalid choice. Aborting."
        exit 1
        ;;
esac

# Step 4: Commit if needed
if ! git diff --cached --quiet; then
    echo -e "\n💾 Creating commit..."

    # Generate commit message
    COMMIT_MSG="feat: Tech audit implementations with upstream sync

Implemented high-priority security improvements:
- Timing-resistant PIR server (PIR-002)
- ZK SRS management (ZK-001)
- KAN-HD calibration suite (CAL-003)
- Consent ledger system (GOV-004)
- Security test suite

Merged with upstream changes to maintain compatibility."

    git commit -m "$COMMIT_MSG"
    echo "✅ Changes committed!"
else
    echo "✅ No changes to commit"
fi

# Step 5: Push options
echo -e "\n🚀 Ready to push?"
echo "1) Push to main branch"
echo "2) Create feature branch and push"
echo "3) Don't push yet"

read -p "Enter choice (1-3): " push_choice

case $push_choice in
    1)
        git push origin main
        echo "✅ Pushed to main!"
        ;;
    2)
        branch="feat/audit-impl-$(date +%Y%m%d)"
        git checkout -b $branch
        git push -u origin $branch
        echo "✅ Pushed to $branch!"
        echo "PR URL: https://github.com/rohanvinaik/GenomeVault/compare/$branch"
        ;;
    3)
        echo "✓ Changes committed locally. Push when ready with: git push origin main"
        ;;
esac

echo -e "\n✨ Done! Your audit implementations are ready."

# Cleanup
rm -f README.md.backup 2>/dev/null || true
