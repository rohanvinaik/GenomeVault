#!/bin/bash

# Script to commit and push the updated README to GitHub

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
git commit -m "Update README with realistic performance benchmarks and grounded project description

- Add comprehensive performance benchmarks based on actual code
- Include detailed HDC, PIR, and ZK performance metrics
- Add system requirements and optimization tips
- Remove unsubstantiated claims and fake testimonials
- Clearly mark as research prototype with appropriate disclaimers"

# Push to main branch
echo -e "\nPushing to GitHub..."
git push origin main

echo -e "\nDone! README has been updated on GitHub."

# Show the commit
echo -e "\nLatest commit:"
git log --oneline -1
