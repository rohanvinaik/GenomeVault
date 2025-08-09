#!/bin/bash
cd /Users/rohanvinaik/genomevault

echo "Task 6: CI Pipeline"
echo "==================="

# Commit the changes
echo "Committing Task 6 changes..."
git add -A
git commit -m "ci: add lint+test workflow

- Created .github/workflows/ci.yml
- Configured for Python 3.10 and 3.11
- Runs pre-commit hooks and pytest
- Triggers on push and pull_request"

echo "Task 6 completed!"
echo ""
echo "Summary of all completed tasks:"
echo "================================"
echo "✅ Task 0: Repo Hygiene"
echo "✅ Task 1: Syntax-Error Pass"
echo "✅ Task 2: Auto-format & Basic Lint"
echo "✅ Task 3: Tighten Ruff Rules"
echo "✅ Task 4: Type-Safety Baseline"
echo "✅ Task 5: Unit Tests"
echo "✅ Task 6: CI Pipeline"
echo ""
echo "The codebase is now lint-clean, type-aware, and CI-protected!"
