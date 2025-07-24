#!/bin/bash
# push_to_github.sh
# Simple script to push changes to GitHub

echo "🚀 Pushing to GitHub"
echo "==================="

cd /Users/rohanvinaik/genomevault

# Show current branch
echo "📍 Current branch:"
git branch --show-current

# Show remote info
echo -e "\n🌐 Remote repository:"
git remote -v | grep push

# Check if we have commits to push
echo -e "\n📊 Commits to push:"
git log origin/main..HEAD --oneline

if [ $? -eq 0 ]; then
    echo -e "\n📤 Pushing to GitHub..."
    git push origin main
    
    if [ $? -eq 0 ]; then
        echo -e "\n✅ Push successful!"
        echo "Check your GitHub repository for the updates."
    else
        echo -e "\n❌ Push failed. Possible issues:"
        echo "1. Authentication: You may need to enter your GitHub credentials"
        echo "2. If using HTTPS, you need a Personal Access Token (PAT)"
        echo "3. If using SSH, ensure your SSH key is added to GitHub"
        echo ""
        echo "To create a PAT:"
        echo "1. Go to GitHub.com → Settings → Developer settings → Personal access tokens"
        echo "2. Generate new token with 'repo' scope"
        echo "3. Use the token as your password when prompted"
    fi
else
    echo -e "\n✅ Already up to date - nothing to push!"
fi

# Show recent commits
echo -e "\n📝 Recent commits:"
git log --oneline -5
