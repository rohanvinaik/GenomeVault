#!/bin/bash

# GenomeVault Tech Audit Implementation Push Script
# This script stages and commits the audit improvements to GitHub

set -e  # Exit on error

echo "🧬 GenomeVault Tech Audit Implementation Push"
echo "============================================"

# Ensure we're in the genomevault directory
cd /Users/rohanvinaik/genomevault

# Check git status
echo -e "\n📊 Current git status:"
git status --short

# Add all the new files we created
echo -e "\n📁 Adding new audit implementation files..."

# PIR Security Improvements
git add genomevault/pir/server/secure_pir_server.py

# ZK SRS Manager
git add genomevault/zk_proofs/srs_manager/
git add genomevault/zk_proofs/srs_manager/__init__.py
git add genomevault/zk_proofs/srs_manager/srs_manager.py

# KAN-HD Calibration Suite
git add genomevault/hypervector/kan/calibration/
git add genomevault/hypervector/kan/calibration/__init__.py
git add genomevault/hypervector/kan/calibration/calibration_suite.py

# Consent Ledger
git add genomevault/blockchain/consent/
git add genomevault/blockchain/consent/__init__.py
git add genomevault/blockchain/consent/consent_ledger.py

# Security Tests
git add tests/security/
git add tests/security/__init__.py
git add tests/security/test_timing_side_channels.py

# Documentation
git add README.md
git add AUDIT_IMPLEMENTATION_SUMMARY.md

# Check what we're about to commit
echo -e "\n📝 Files to be committed:"
git status --short

# Create commit message
COMMIT_MESSAGE="feat: Implement high-priority tech audit improvements

Based on comprehensive tech audit (2025-07-26), implemented:

1. PIR Timing Side-Channel Protection (PIR-002)
   - SecurePIRServer with padding, jitter, constant-time ops
   - QueryMixer for correlation attack prevention
   - Target: <1% timing variance

2. ZK Backend Security (ZK-001)
   - SRSManager for cryptographic lifecycle management
   - GnarkDockerBuilder for deterministic builds
   - Domain-separated transcripts

3. KAN-HD Calibration Suite (CAL-003)
   - Clinical error budget framework
   - Pareto frontier analysis
   - Use-case specific recommendations

4. Consent Ledger System (GOV-004)
   - Cryptographic consent binding
   - ZK proof integration
   - HIPAA/GDPR compliance

5. Security Test Suite
   - Adversarial timing attack tests
   - Statistical side-channel analysis
   - ML attack simulations

These improvements enhance security, enable clinical deployment,
and ensure regulatory compliance as identified in the audit.

Refs: #security #audit #clinical-validation"

# Commit the changes
echo -e "\n💾 Committing changes..."
git commit -m "$COMMIT_MESSAGE"

echo -e "\n✅ Changes committed successfully!"
echo -e "\n🚀 Ready to push to GitHub. Run:"
echo "   git push origin main"
echo ""
echo "Or if you're on a different branch:"
echo "   git push origin <your-branch-name>"
echo ""
echo "To create a new branch for these changes:"
echo "   git checkout -b feat/tech-audit-improvements"
echo "   git push -u origin feat/tech-audit-improvements"
