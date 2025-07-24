#!/bin/bash
# simple_push_fix.sh
# Based on the post-mortem lessons - simple and direct

echo "Git Push Fix - Simple and Direct"
echo "================================"

cd /Users/rohanvinaik/genomevault

# Step 1: Show current state (for you to see, not me)
echo -e "\n1. Current status:"
git status

# Step 2: If there are uncommitted changes, add and commit them
echo -e "\n2. Adding any uncommitted changes:"
git add -A

# Step 3: Commit if there are changes
echo -e "\n3. Creating commit:"
git commit -m "Complete dev pipeline implementation with all fixes" || echo "Nothing to commit"

# Step 4: Show what we're about to push
echo -e "\n4. Commits ready to push:"
git log origin/main..HEAD --oneline

# Step 5: Actually push
echo -e "\n5. Pushing to origin/main:"
git push origin main

# Step 6: Final status
echo -e "\n6. Final status:"
git status
echo -e "\nDone. Check the output above for any errors."
