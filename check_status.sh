#!/bin/bash

# Quick status check for GenomeVault

cd /Users/rohanvinaik/genomevault

echo "🧬 GenomeVault Local Changes Status"
echo "==================================="

echo -e "\n📋 Modified files:"
git status --short

echo -e "\n📄 README.md changes preview:"
echo "----------------------------"
git diff README.md | head -30

echo -e "\n📁 New audit implementation files:"
echo "--------------------------------"
for file in genomevault/pir/server/secure_pir_server.py \
            genomevault/zk_proofs/srs_manager/srs_manager.py \
            genomevault/hypervector/kan/calibration/calibration_suite.py \
            genomevault/blockchain/consent/consent_ledger.py \
            tests/security/test_timing_side_channels.py \
            AUDIT_IMPLEMENTATION_SUMMARY.md; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (not found)"
    fi
done

echo -e "\n💡 Recommendations:"
echo "1. Run ./interactive_merge.sh for step-by-step merge"
echo "2. Or run ./safe_merge_push.sh for automatic handling"
