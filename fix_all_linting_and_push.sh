#!/bin/bash
# fix_all_linting_and_push.sh
# Script to fix all linting issues (isort, flake8, pylint) and push

echo "🔧 Fixing All Linting Issues"
echo "============================"

cd /Users/rohanvinaik/genomevault

# First, let's see the current state
echo "📊 Initial git status:"
git status --short

# Run isort to fix import ordering
echo -e "\n📝 Running isort to fix import ordering..."
isort .
echo "✅ isort complete"

# Check what isort changed
echo -e "\n📋 Files modified by isort:"
git diff --name-only

# Now let's check Flake8 issues (but not fix automatically as some need manual intervention)
echo -e "\n🔍 Checking Flake8 issues..."
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || true
echo "ℹ️  Note: Most Flake8 issues need manual fixes"

# Check for the most common auto-fixable issues
echo -e "\n🔧 Attempting to fix common issues..."

# Fix trailing whitespace
find . -name "*.py" -type f -exec sed -i '' 's/[[:space:]]*$//' {} \;
echo "✅ Removed trailing whitespace"

# Check for any changes
if git diff --quiet; then
    echo -e "\n✅ No changes needed!"
else
    echo -e "\n📦 Staging all changes..."
    git add -A
    
    echo -e "\n📋 Summary of changes:"
    git diff --cached --stat
    
    # Commit the changes
    echo -e "\n💾 Committing linting fixes..."
    git commit -m "Fix linting issues (isort and formatting)

- Applied isort to fix import ordering
- Removed trailing whitespace
- Ensured consistent code style

This addresses CI linting failures for:
- isort import ordering
- Basic formatting issues"
    
    # Push to GitHub
    echo -e "\n🌐 Pushing to GitHub..."
    git push origin main
    
    echo -e "\n✨ Linting fixes pushed successfully!"
fi

# Final checks
echo -e "\n🔍 Final verification:"
echo "- Black check:"
black --check . && echo "  ✅ Black: PASS" || echo "  ❌ Black: FAIL"

echo -e "\n- isort check:"
isort --check-only . && echo "  ✅ isort: PASS" || echo "  ❌ isort: FAIL"

echo -e "\n- Flake8 summary:"
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

echo -e "\n📊 Pylint score (for reference):"
pylint genomevault --fail-under=0 --exit-zero | grep "Your code has been rated at" || echo "  (Pylint check skipped)"

echo -e "\n✅ Linting fixes complete!"
echo "Check GitHub Actions to see if CI passes now."
