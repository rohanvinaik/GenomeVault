#!/bin/bash
# Interactive PIR Implementation Git Push

echo "🚀 Preparing to push PIR Implementation to GitHub"
echo "================================================"

# Change to project directory
cd /Users/rohanvinaik/genomevault

# Show current branch
echo -e "\n📍 Current branch:"
git branch --show-current

# Show git status
echo -e "\n📊 Git status:"
git status

# Add PIR files
echo -e "\n📁 Adding PIR implementation files..."
git add genomevault/pir/
git add tests/pir/
git add scripts/bench_pir.py
git add schemas/pir_query.json
git add schemas/pir_response.json
git add PIR_IMPLEMENTATION_SUMMARY.md
git add check_pir_quality.sh

# Show what will be committed
echo -e "\n📋 Changes to be committed:"
git diff --cached --stat

# Ask for confirmation
echo -e "\n❓ Do you want to proceed with the commit and push? (y/n)"
read -r response

if [[ "$response" == "y" || "$response" == "Y" ]]; then
    # Commit
    echo -e "\n💾 Committing changes..."
    git commit -m "feat: Implement Information-Theoretic PIR Protocol

- Add 2-server IT-PIR with XOR-based scheme
- Implement enhanced PIR server with optimizations
- Add PIR coordinator for server management
- Create high-level query builder interface
- Add comprehensive test suite and benchmarks
- Implement security features
- Add JSON schemas and integration demo

See PIR_IMPLEMENTATION_SUMMARY.md for details."

    # Push
    echo -e "\n🌐 Pushing to GitHub..."
    git push origin main
    
    echo -e "\n✅ Successfully pushed to GitHub!"
    echo "🔗 View at: https://github.com/rohanvinaik/GenomeVault"
else
    echo -e "\n❌ Push cancelled."
fi
