#!/bin/bash
# Prepare GenomeVault for GitHub push

echo "Preparing GenomeVault for GitHub push..."

cd /Users/rohanvinaik/genomevault

# 1. Run Black to format code
echo "1. Running Black formatter..."
black .

# 2. Run isort to sort imports
echo -e "\n2. Running isort..."
isort .

# 3. Check for stubbed files
echo -e "\n3. Checking for stubbed files..."
python scripts/check_no_stubs.py

# 4. Stage all changes
echo -e "\n4. Staging changes..."
git add -A

# 5. Show status
echo -e "\n5. Git status:"
git status

echo -e "\nReady to commit and push!"
echo "Next steps:"
echo "  git commit -m 'Fix: Implement core modules and add CI/CD pipeline'"
echo "  git push origin main"
