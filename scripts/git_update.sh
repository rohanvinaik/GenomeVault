#!/bin/bash

# GenomeVault Git Update Script
# Safely commits and pushes changes to GitHub

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}[STATUS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Navigate to the genomevault directory
cd /Users/rohanvinaik/genomevault || { print_error "Failed to navigate to genomevault directory"; exit 1; }

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a git repository!"
    exit 1
fi

# Fetch latest changes from remote
print_status "Fetching latest changes from remote..."
git fetch origin

# Check for uncommitted changes
if [[ -n $(git status -s) ]]; then
    print_status "Uncommitted changes detected"
    
    # Show status
    git status -s
    
    # Ask for commit message
    echo -n "Enter commit message (or 'skip' to skip commit): "
    read commit_msg
    
    if [ "$commit_msg" != "skip" ]; then
        # Add all changes
        print_status "Adding all changes..."
        git add -A
        
        # Commit changes
        print_status "Committing changes..."
        git commit -m "$commit_msg"
    else
        print_warning "Skipping commit"
    fi
else
    print_status "No uncommitted changes"
fi

# Check if we're ahead of remote
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u} 2>/dev/null)
BASE=$(git merge-base @ @{u} 2>/dev/null)

if [ -z "$REMOTE" ]; then
    print_error "No upstream branch set"
    print_status "Setting upstream to origin/main..."
    git push -u origin main
elif [ $LOCAL = $REMOTE ]; then
    print_status "Branch is up to date with remote"
elif [ $LOCAL = $BASE ]; then
    print_warning "Branch is behind remote. Pull required."
    echo -n "Pull changes? (y/n): "
    read pull_confirm
    if [ "$pull_confirm" = "y" ]; then
        git pull origin main
    fi
elif [ $REMOTE = $BASE ]; then
    print_status "Branch is ahead of remote. Pushing changes..."
    git push origin main
else
    print_warning "Branch has diverged from remote"
    echo "You may need to merge or rebase. Manual intervention required."
    exit 1
fi

# Final status
print_status "Git update complete!"
git log --oneline -5
