#!/bin/bash

# Quick commit and push script for GenomeVault
# Usage: ./quick_push.sh "commit message"

cd /Users/rohanvinaik/genomevault || exit 1

# Use provided message or default
COMMIT_MSG="${1:-Update GenomeVault implementation}"

# Add all changes
git add -A

# Commit
git commit -m "$COMMIT_MSG"

# Push
git push origin main

echo "✅ Changes pushed to GitHub"
