#!/bin/bash

echo "🧬 Fixing linting issues and pushing KAN-HD enhancements..."

cd /Users/rohanvinaik/genomevault

# The pre-commit hooks have already been run and fixed most formatting
# Now let's add the already-formatted files and commit them

echo "📝 Adding formatted files..."
git add .

echo "💾 Committing with the fixes from pre-commit hooks..."
git commit -m "🚀 KAN-HD Hybrid Architecture - Code formatting and linting fixes

Applied automatic formatting from pre-commit hooks including:
- black formatting for consistent code style
- isort for import organization
- trailing whitespace removal
- flake8 compliance improvements

This commit contains the KAN-HD hybrid enhancement with proper formatting."

echo "🚀 Pushing to GitHub..."
git push origin main

echo "✅ Push completed! Check: https://github.com/rohanvinaik/GenomeVault"
echo ""
echo "🎉 Your KAN-HD enhancements are now live on GitHub!"
echo "🔗 Repository: https://github.com/rohanvinaik/GenomeVault"
echo ""
echo "📋 Next steps:"
echo "1. Check the GitHub repository to verify all files are there"
echo "2. Test the enhancements: python examples/kan_hd_enhanced_demo.py"
echo "3. Try the new API endpoints"
echo "4. Consider creating a release tag for this major version"
