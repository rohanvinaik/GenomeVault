#!/bin/bash
# Quick CI fix push

cd /Users/rohanvinaik/genomevault

echo "🔧 Quick CI Fix Push"

# Add all changes
git add -A

# Commit
git commit -m "Add tests and Python implementation for CI

- Added test_basic.py with comprehensive tests
- Added test_simple.py as fallback
- Fixed module imports
- All modules now have proper implementations"

# Push
git push origin main

echo "✅ Pushed! Check CI status at: https://github.com/rohanvinaik/GenomeVault/actions"
