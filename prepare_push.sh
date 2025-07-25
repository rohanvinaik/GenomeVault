#!/bin/bash

echo "=== Preparing to push SNP dial implementation ==="

# Check git status
echo "Checking git status..."
git status

# Add new files
echo -e "\nAdding new files..."
git add genomevault/hypervector/positional.py
git add genomevault/api/routers/query_tuned.py
git add test_snp_dial.py
git add SNP_DIAL_README.md
git add examples/webdial/ACCURACY_DIAL_README.md

# Add modified files
echo -e "\nAdding modified files..."
git add genomevault/hypervector/encoding/genomic.py
git add genomevault/pir/client/batched_query_builder.py
git add genomevault/api/app.py
git add examples/webdial/index.html
git add README.md

# Show what will be committed
echo -e "\nFiles to be committed:"
git status --short

# Create commit
echo -e "\nCreating commit..."
git commit -m "feat: Add SNP dial for single-nucleotide accuracy

- Implement sparse positional encoding for up to 10M SNP positions
- Add panel granularity settings (Off/Common/Clinical/Custom)
- Support hierarchical zoom queries (3 levels)
- Update accuracy dial demo to realistic 90-99.99% range
- Add comprehensive test script and documentation

Key components:
- hypervector/positional.py: Memory-efficient position vectors
- Enhanced genomic encoder with panel support
- New API endpoints for panel and zoom queries
- Batch PIR query support for zoom levels
- Updated interactive demo with meaningful accuracy ranges"

# Show the commit
echo -e "\nCommit created:"
git log -1 --oneline

echo -e "\n=== Ready to push! ==="
echo "To push to remote, run:"
echo "  git push origin main"
echo ""
echo "Or if working on a branch:"
echo "  git push origin <branch-name>"
