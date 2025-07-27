#!/bin/bash

# Quick push to clean-slate branch

cd /Users/rohanvinaik/genomevault

# Create/checkout clean-slate branch
git checkout -b clean-slate 2>/dev/null || git checkout clean-slate

# Add all audit fix files
git add fix_*.py validate_*.py preflight_check.py quick_fix_init_files.py generate_comparison_report.py
git add *.sh
git add AUDIT*.md REAL_AUDIT_STATUS.md QUICK_REFERENCE.txt

# Commit
git commit -m "Add audit fix scripts and documentation for GenomeVault code quality improvements"

# Push
git push -u origin clean-slate

echo "✅ Pushed to clean-slate branch!"
