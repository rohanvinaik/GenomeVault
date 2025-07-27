#!/bin/bash

# Quick push script with automatic branch creation
# This creates a feature branch and pushes the audit improvements

set -e

echo "🧬 GenomeVault Audit Improvements - Quick Push"
echo "=============================================="

cd /Users/rohanvinaik/genomevault

# Create and checkout feature branch
BRANCH_NAME="feat/tech-audit-improvements-$(date +%Y%m%d)"
echo -e "\n🌿 Creating feature branch: $BRANCH_NAME"
git checkout -b $BRANCH_NAME 2>/dev/null || git checkout $BRANCH_NAME

# Add all audit implementation files
echo -e "\n📁 Staging audit implementation files..."
git add genomevault/pir/server/secure_pir_server.py
git add genomevault/zk_proofs/srs_manager/
git add genomevault/hypervector/kan/calibration/
git add genomevault/blockchain/consent/
git add tests/security/
git add README.md
git add AUDIT_IMPLEMENTATION_SUMMARY.md

# Quick commit
echo -e "\n💾 Committing..."
git commit -m "feat: Implement tech audit security improvements (PIR-002, ZK-001, CAL-003, GOV-004)

- Timing-resistant PIR server
- ZK SRS management
- KAN-HD calibration suite
- Consent ledger system
- Security test suite

See AUDIT_IMPLEMENTATION_SUMMARY.md for details."

# Push to origin
echo -e "\n🚀 Pushing to GitHub..."
git push -u origin $BRANCH_NAME

echo -e "\n✅ Successfully pushed to branch: $BRANCH_NAME"
echo -e "\n📋 Next steps:"
echo "1. Go to https://github.com/rohanvinaik/GenomeVault"
echo "2. Create a Pull Request from '$BRANCH_NAME' to 'main'"
echo "3. Add reviewers and merge when ready"
echo ""
echo "Direct PR link:"
echo "https://github.com/rohanvinaik/GenomeVault/compare/main...$BRANCH_NAME?quick_pull=1"
