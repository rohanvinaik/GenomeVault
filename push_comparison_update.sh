#!/bin/bash

# Script to commit and push the updated README with comparison tables to GitHub

cd /Users/rohanvinaik/genomevault

# Check current branch
echo "Current branch:"
git branch --show-current

# Check status
echo -e "\nGit status:"
git status

# Stage the README changes
echo -e "\nStaging README.md..."
git add README.md

# Commit with a descriptive message
echo -e "\nCommitting changes..."
git commit -m "Add comprehensive comparison tables to README

- Add vertical comparison tables for easier side-by-side analysis
- Include storage & compression comparisons with traditional formats
- Add privacy-preserving analysis method comparisons
- Include genomic similarity search benchmarks
- Add zero-knowledge proof system comparisons
- Include clinical platform comparisons
- Add key advantages and limitations summary with visual layout
- Use checkmarks and emojis for better visual clarity"

# Push to main branch
echo -e "\nPushing to GitHub..."
git push origin main

echo -e "\nDone! README comparison tables have been added on GitHub."

# Show the commit
echo -e "\nLatest commit:"
git log --oneline -1
