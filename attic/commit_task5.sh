#!/bin/bash
cd /Users/rohanvinaik/genomevault

echo "Task 5: Unit Tests"
echo "=================="

# Commit the changes
echo "Committing Task 5 changes..."
git add -A
git commit -m "test: make pytest suite pass

- Created tests/smoke/test_api_startup.py for API health checks
- Created tests/unit/test_voting_power.py for voting power parity test
- Fixed import paths in test_refactored_circuits.py
- Added smoke test directory structure"

echo "Task 5 completed!"
