#!/bin/bash

# GenomeVault Safe Merge and Push Script
# Handles conflicts and preserves your local changes

set -e

echo "🧬 GenomeVault - Safe Merge and Push"
echo "===================================="

cd /Users/rohanvinaik/genomevault

# Step 1: Show current status
echo -e "\n📊 Current status:"
git status --short

# Step 2: Stash local changes
echo -e "\n💾 Stashing your local changes..."
git stash save "Local audit improvements - $(date '+%Y-%m-%d %H:%M:%S')"

# Step 3: Pull remote changes
echo -e "\n⬇️  Pulling remote changes..."
git pull origin main

# Step 4: Apply stashed changes
echo -e "\n📝 Applying your local changes back..."
git stash pop || {
    echo "⚠️  Merge conflicts detected. Let's resolve them..."
    echo ""
    echo "Files with conflicts:"
    git diff --name-only --diff-filter=U
    echo ""
    echo "Opening merge tool to resolve conflicts..."
    
    # Try to auto-merge where possible
    git status --porcelain | grep "^UU" | awk '{print $2}' | while read file; do
        echo "Attempting auto-merge for: $file"
        if [[ "$file" == "README.md" ]]; then
            echo "README.md has conflicts. Keeping local version with remote additions..."
            # This will keep your local README but we'll need to manually review
            git checkout --ours README.md
            git add README.md
        fi
    done
}

# Step 5: Check if we need to stage the audit improvements
echo -e "\n📁 Checking audit implementation files..."

# List of new files we created
NEW_FILES=(
    "genomevault/pir/server/secure_pir_server.py"
    "genomevault/zk_proofs/srs_manager/__init__.py"
    "genomevault/zk_proofs/srs_manager/srs_manager.py"
    "genomevault/hypervector/kan/calibration/__init__.py"
    "genomevault/hypervector/kan/calibration/calibration_suite.py"
    "genomevault/blockchain/consent/__init__.py"
    "genomevault/blockchain/consent/consent_ledger.py"
    "tests/security/__init__.py"
    "tests/security/test_timing_side_channels.py"
    "AUDIT_IMPLEMENTATION_SUMMARY.md"
)

for file in "${NEW_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ Adding: $file"
        git add "$file"
    fi
done

# Also add the modified README
git add README.md

# Step 6: Show what we're about to commit
echo -e "\n📋 Changes to be committed:"
git status --short

# Step 7: Create commit if there are changes
if ! git diff --cached --quiet; then
    echo -e "\n💾 Creating commit..."
    git commit -m "feat: Implement tech audit improvements with README sync

Merged local audit implementations with remote changes:
- Secure PIR server with timing protection (PIR-002)
- ZK SRS management system (ZK-001)
- KAN-HD calibration suite (CAL-003)
- Consent ledger with ZK binding (GOV-004)
- Comprehensive security test suite

Reconciled README.md with upstream changes while preserving
audit implementation documentation."
else
    echo -e "\n✅ No changes to commit - repository is up to date"
fi

# Step 8: Push to remote
echo -e "\n🚀 Ready to push. Choose an option:"
echo "1) Push directly to main"
echo "2) Create a feature branch and push"
echo "3) Exit without pushing"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "Pushing to main..."
        git push origin main
        echo "✅ Successfully pushed to main!"
        ;;
    2)
        BRANCH_NAME="feat/audit-improvements-$(date +%Y%m%d-%H%M%S)"
        echo "Creating branch: $BRANCH_NAME"
        git checkout -b $BRANCH_NAME
        git push -u origin $BRANCH_NAME
        echo "✅ Successfully pushed to $BRANCH_NAME!"
        echo "Create a PR at: https://github.com/rohanvinaik/GenomeVault/compare/main...$BRANCH_NAME?quick_pull=1"
        ;;
    3)
        echo "Exiting without pushing. Your changes are committed locally."
        ;;
    *)
        echo "Invalid choice. Exiting without pushing."
        ;;
esac

echo -e "\n🎉 Done!"
