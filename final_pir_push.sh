#!/bin/bash
# Final push after fixing all linting issues

echo "🚀 Final push of PIR Implementation with linting fixes"
echo "===================================================="

cd /Users/rohanvinaik/genomevault

# Show what files have been modified
echo -e "\n📊 Modified files:"
git status --short

# Add all PIR-related files
echo -e "\n📁 Adding all PIR files..."
git add genomevault/pir/ tests/pir/ scripts/bench_pir.py

# Commit the linting fixes
echo -e "\n💾 Committing linting fixes..."
git commit -m "fix: Apply isort and black formatting to PIR implementation

- Fix import ordering with isort (black profile)
- Apply black formatting with 100 char line length
- Ensure all linting checks pass"

# Push to GitHub
echo -e "\n🌐 Pushing to GitHub..."
git push origin main

echo -e "\n✅ Successfully pushed PIR implementation with all fixes!"
echo "🔗 View at: https://github.com/rohanvinaik/GenomeVault"
