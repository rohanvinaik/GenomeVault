#!/bin/bash
# Quick script to fix Black issues and push to GitHub

echo "🔧 Fixing Black formatting issues for GenomeVault..."
echo

# Make sure we're in the genomevault directory
cd /Users/rohanvinaik/genomevault

# Run the automatic fix script
echo "Running automatic Black fix..."
python3 fix_black_auto.py

# Check the result
if [ $? -eq 0 ]; then
    echo
    echo "✅ Black issues fixed!"
    echo

    # Show what changed
    echo "📝 Changes made:"
    git status --short

    echo
    read -p "Do you want to commit and push these changes? (y/n) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add -A
        git commit -m "Fix Black formatting errors for CI"
        git push
        echo
        echo "✅ Changes pushed to GitHub!"
    else
        echo "Changes not committed. You can review and commit manually."
    fi
else
    echo
    echo "❌ Some issues could not be fixed automatically."
    echo "Please check the error messages above."
fi
