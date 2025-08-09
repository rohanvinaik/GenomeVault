#!/bin/bash
cd /Users/rohanvinaik/genomevault

echo "Task 4: Type-Safety Baseline"
echo "============================"

# Commit the changes
echo "Committing Task 4 changes..."
git add -A
git commit -m "feat(types): introduce mypy baseline

- Added mypy.ini configuration
- Set python_version = 3.11
- Configured namespace_packages = True
- Set strict = False for baseline
- Targeted files: genomevault, zk_proofs, pir
- Added ignore rules for unfinished areas"

echo "Task 4 completed!"
