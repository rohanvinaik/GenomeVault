#!/bin/bash

# GenomeVault GitHub Setup Script

echo "================================================="
echo "    GenomeVault GitHub Repository Setup"
echo "================================================="
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "Git repository not initialized. Run 'git init' first."
    exit 1
fi

echo "This script will help you push GenomeVault to GitHub."
echo ""
echo "Prerequisites:"
echo "1. You need a GitHub account"
echo "2. You need to create a new repository named 'GenomeVault' on GitHub"
echo "   - Go to: https://github.com/new"
echo "   - Repository name: GenomeVault"
echo "   - Description: Privacy-preserving genomic data platform"
echo "   - Make it public or private as you prefer"
echo "   - DO NOT initialize with README, .gitignore, or license"
echo ""
read -p "Have you created the GitHub repository? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please create the repository first, then run this script again."
    exit 1
fi

# Get GitHub username
read -p "Enter your GitHub username: " GITHUB_USERNAME

# Construct repository URL
REPO_URL="https://github.com/${GITHUB_USERNAME}/GenomeVault.git"

echo ""
echo "Setting up remote repository..."
git remote add origin $REPO_URL

echo ""
echo "Your repository will be pushed to: $REPO_URL"
read -p "Is this correct? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting. You can manually set the remote with:"
    echo "  git remote set-url origin <your-repo-url>"
    exit 1
fi

# Create main branch if it doesn't exist
git branch -M main

echo ""
echo "Pushing to GitHub..."
echo "You may be prompted for your GitHub credentials."
echo ""

# Push to GitHub
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Success! GenomeVault has been pushed to GitHub!"
    echo ""
    echo "Your repository: https://github.com/${GITHUB_USERNAME}/GenomeVault"
    echo ""
    echo "Next steps:"
    echo "1. Add a description and topics to your repository"
    echo "2. Set up branch protection rules for 'main'"
    echo "3. Configure GitHub Actions secrets if needed"
    echo "4. Consider adding badges to your README"
    echo ""
    echo "Suggested topics for your repo:"
    echo "  genomics, privacy, zero-knowledge-proofs, blockchain,"
    echo "  bioinformatics, hyperdimensional-computing, healthcare,"
    echo "  cryptography, federated-learning, precision-medicine"
else
    echo ""
    echo "❌ Push failed. Common issues:"
    echo "1. Authentication failed - set up a GitHub personal access token"
    echo "2. Repository doesn't exist - create it on GitHub first"
    echo "3. Repository name mismatch - check the URL"
    echo ""
    echo "For authentication, you may need to:"
    echo "1. Create a personal access token at https://github.com/settings/tokens"
    echo "2. Use the token as your password when prompted"
fi
