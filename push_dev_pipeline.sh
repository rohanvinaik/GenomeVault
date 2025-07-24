#!/bin/bash
# push_dev_pipeline_implementation.sh
# Script to commit and push the dev pipeline implementation

echo "🚀 Pushing Dev Pipeline Implementation to GitHub"
echo "=============================================="

# Navigate to the repository
cd /Users/rohanvinaik/genomevault

# Show current status
echo -e "\n📋 Current Git Status:"
git status --short

# Add all new files
echo -e "\n📦 Adding new files..."
git add -A

# Show what will be committed
echo -e "\n📝 Files to be committed:"
git status --short

# Create commit message
COMMIT_MSG="Implement cross-cutting dev pipeline infrastructure

- Add comprehensive dev pipeline audit (DEV_PIPELINE_AUDIT.md)
- Create test naming standardization script (scripts/fix_test_naming.sh)
- Add automated benchmark CI workflow (.github/workflows/benchmarks.yml)
- Document benchmarking system (benchmarks/README.md)
- Update test documentation with naming conventions
- Add performance report generator integration

This completes the implementation of the cross-cutting setup:
✅ Versioning & registry (VERSION.md, version.py)
✅ Metrics harness (bench.py with lane support)
✅ Threat model matrix (SECURITY.md, security_check.py)
✅ Test taxonomy (proper directory structure)
✅ Makefile targets (bench-*, threat-scan, coverage)

The pipeline now provides consistent development experience across
all three lanes (ZK, PIR, HDC) with automated benchmarking, security
validation, and standardized testing conventions."

# Commit changes
echo -e "\n💾 Committing changes..."
git commit -m "$COMMIT_MSG"

# Show the commit
echo -e "\n✅ Commit created:"
git log --oneline -1

# Push to remote
echo -e "\n🌐 Pushing to GitHub..."
git push origin main

echo -e "\n✨ Push complete! Check GitHub for the updates."
echo "Next steps:"
echo "1. Run ./scripts/fix_test_naming.sh to standardize remaining test names"
echo "2. Check GitHub Actions tab to see the new benchmarks workflow"
echo "3. Review the pull request template if you need to create a PR"
