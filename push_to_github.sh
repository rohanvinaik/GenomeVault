#!/bin/bash

# GenomeVault GitHub Push Script
# This script stages all changes and pushes to GitHub

echo "=== GenomeVault GitHub Push ==="
echo

# Navigate to project directory
cd /Users/rohanvinaik/genomevault

# Check current status
echo "Current git status:"
git status
echo

# Add all new files
echo "Adding all files..."
git add .

# Show what will be committed
echo "Files to be committed:"
git status --short
echo

# Commit with a descriptive message
echo "Creating commit..."
git commit -m "Implement GenomeVault 3.0 core architecture

- Add comprehensive configuration system with dual-axis node model
- Implement local processing with differential storage
- Create hypervector encoding with 10,000x compression
- Add zero-knowledge proof system with PLONK circuits
- Implement information-theoretic PIR with optimal server selection
- Create blockchain node with Tendermint BFT consensus
- Add FastAPI-based network API
- Include example usage demonstrating complete workflow

Key features:
- Compression tiers: Mini (25KB), Clinical (300KB), Full HDC (100-200KB)
- Privacy guarantees: PIR P_fail = (1-q)^k, DP ε=1.0
- Dual-axis voting: w = c + s (hardware + trust)
- HIPAA fast-track for healthcare providers
- Post-quantum ready cryptography"

# Push to GitHub
echo
echo "Pushing to GitHub..."
git push origin main

echo
echo "=== Push Complete ==="
echo "Your code is now on GitHub!"
echo
echo "View your repository at: https://github.com/Roh-codeur/genomevault"
