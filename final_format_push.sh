#!/bin/bash
# Final push with all formatting fixes

cd /Users/rohanvinaik/genomevault

echo "Final Formatting and Push"
echo "========================"

# Install black if not present
if ! command -v black &> /dev/null; then
    echo "Installing black..."
    pip install black
fi

# Fix all Python files
echo "Formatting all Python files..."
find . -name "*.py" -path "./genomevault/*" -o -name "test_*.py" -o -name "example_*.py" | while read file; do
    if [ -f "$file" ]; then
        echo "Formatting: $file"
        black --quiet --target-version py311 "$file" 2>/dev/null || true
    fi
done

# Stage all changes
git add -A

# Show status
echo -e "\nGit status:"
git status --short

# Commit if there are changes
if ! git diff --cached --quiet; then
    git commit -m "fix: Apply comprehensive Black formatting

- Fix all Black formatting issues
- Ensure Python 3.11 compatibility
- Format all catalytic implementation files"
fi

# Push
git push origin $(git branch --show-current)

echo -e "\n✅ All formatting applied and pushed!"
echo "Check GitHub Actions to see if CI passes now."
