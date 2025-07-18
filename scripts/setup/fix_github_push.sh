#!/bin/bash

# Quick fix for GitHub push

echo "Let's fix your GitHub remote setup..."
echo ""

# Remove any existing origin
git remote remove origin 2>/dev/null

# Get GitHub username
read -p "Enter your GitHub username: " GITHUB_USERNAME

# Validate username is not empty
if [ -z "$GITHUB_USERNAME" ]; then
    echo "Error: GitHub username cannot be empty"
    exit 1
fi

# Set up the remote
echo ""
echo "Setting up remote for: https://github.com/${GITHUB_USERNAME}/GenomeVault.git"
git remote add origin "https://github.com/${GITHUB_USERNAME}/GenomeVault.git"

# Verify the remote
echo ""
echo "Remote configured as:"
git remote -v

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
echo "Note: You'll need to authenticate with either:"
echo "  1. Your GitHub password (if you have 2FA disabled)"
echo "  2. A Personal Access Token (if you have 2FA enabled)"
echo ""
echo "To create a Personal Access Token:"
echo "  1. Go to https://github.com/settings/tokens"
echo "  2. Click 'Generate new token (classic)'"
echo "  3. Give it 'repo' scope"
echo "  4. Use the token as your password"
echo ""

git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Success! GenomeVault has been pushed to GitHub!"
    echo "Repository: https://github.com/${GITHUB_USERNAME}/GenomeVault"
else
    echo ""
    echo "If push failed, try:"
    echo "  1. Make sure you created the repository on GitHub"
    echo "  2. Check your authentication (use a Personal Access Token if needed)"
    echo "  3. Verify the repository name is exactly 'GenomeVault'"
fi
