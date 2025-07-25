#!/bin/bash

echo "=== Adding SNP dial implementation files ==="

# Check current status
echo "Current git status:"
git status --porcelain

# Add all the new and modified files
echo -e "\nAdding files..."

# Add new files
git add genomevault/hypervector/positional.py
git add genomevault/api/routers/query_tuned.py
git add test_snp_dial.py
git add SNP_DIAL_README.md
git add examples/webdial/ACCURACY_DIAL_README.md
git add SNP_DIAL_IMPLEMENTATION_SUMMARY.md

# Add modified files
git add genomevault/hypervector/encoding/genomic.py
git add genomevault/pir/client/batched_query_builder.py
git add genomevault/api/app.py
git add examples/webdial/index.html
git add README.md

# Also add the helper scripts
git add lint_snp_dial.sh
git add prepare_push.sh

# Show what's staged
echo -e "\nStaged files:"
git status --short

# Commit
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

echo -e "\nDone! Now you can push with: git push origin main"
