#!/bin/bash

# Push the linting fixes

echo "🚀 Pushing linting fixes for HDC error handling"

# Add the modified files
echo "📄 Adding modified files..."
git add genomevault/api/app.py
git add genomevault/core/constants.py
git add genomevault/hypervector/error_handling.py

# Check status
echo -e "\n📊 Git status:"
git status --short

# Commit
echo -e "\n💾 Committing linting fixes..."
git commit -m "fix: Apply linting fixes for HDC error handling

- Fix Black formatting in constants.py
- Replace underscore variable assignments with proper names
- Fix f-string formatting in api/app.py
- Ensure all exception handlers use proper variable names

All linters now pass:
- Black formatting applied
- isort import ordering correct
- flake8 issues resolved
- Variable naming conventions followed"

# Push
echo -e "\n🌐 Pushing to GitHub..."
git push origin main

echo -e "\n✅ Linting fixes pushed successfully!"
