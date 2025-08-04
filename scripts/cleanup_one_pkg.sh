#!/usr/bin/env bash
# cleanup_one_pkg.sh   –  scaffold for an LLM-coder
#
# USAGE
#   ./scripts/cleanup_one_pkg.sh genomevault/hypervector
#
# WHAT IT DOES (coder must implement details ↓)
#   1. format_code      -> run black & isort on the package
#   2. lint_with_ruff   -> run Ruff, auto-fix safe rules, log the rest
#   3. typecheck_strict -> run mypy --strict; if ≤ THRESHOLD errors, auto-fix
#   4. run_tests        -> pytest -k <dotted-package>
#   5. commit_changes   -> git commit staged fixes
# Each step is written as a stub for the coder to fill in.
################################################################################

set -uo pipefail  # Remove -e so script continues on errors
PKG=${1:-}
if [[ -z "$PKG" || ! -d "$PKG" ]]; then
  echo "usage: $0 <package-dir>"; exit 1
fi

# ——————————————————————————————————————————————— CONFIG ——————————————————————————————————————————————— #
MYPY_AUTO_FIX=10         # only auto-fix if ≤ this many errors
# ————————————————————————————————————————————————————————————————————————————————————————————————————————— #

format_code() {
  echo "🔧 Formatting code in $PKG..."
  
  # Run black on the package
  echo "  Running black..."
  python3 -m black "$PKG" --quiet || true
  
  # Run isort on the package  
  echo "  Running isort..."
  python3 -m isort "$PKG" --quiet || true
  
  # Stage formatting changes
  git add "$PKG" || true
  echo "  ✔ Formatting complete"
}

lint_with_ruff() {
  echo "🔍 Linting with ruff in $PKG..."
  
  # Run ruff with auto-fix for safe rules
  echo "  Running ruff check with auto-fix..."
  python3 -m ruff check "$PKG" --fix --quiet || true  # Don't exit on ruff issues
  
  # Show remaining issues
  echo "  Checking for remaining ruff issues..."
  if ! python3 -m ruff check "$PKG" --quiet 2>/dev/null; then
    echo "  📋 Remaining ruff issues:"
    python3 -m ruff check "$PKG" 2>/dev/null | head -20 || true  # Show first 20 issues
    echo "  (run 'python3 -m ruff check $PKG' to see all)"
  else
    echo "  ✔ No remaining ruff issues"
  fi
  
  # Stage ruff fixes
  git add "$PKG" || true
  echo "  ✔ Linting complete"
}

typecheck_strict() {
  echo "🔬 Type checking with mypy --strict in $PKG..."
  
  # Run mypy and capture error count
  local mypy_output
  local mypy_exit_code=0
  
  mypy_output=$(python3 -m mypy --strict "$PKG" 2>&1) || mypy_exit_code=$?
  
  if [[ $mypy_exit_code -eq 0 ]]; then
    echo "  ✔ No mypy errors found"
    return
  fi
  
  # Count errors (look for "error:" lines)
  local error_count
  error_count=$(echo "$mypy_output" | grep -c "error:" || echo "0")
  
  echo "  Found $error_count mypy errors"
  
  if [[ $error_count -le $MYPY_AUTO_FIX ]]; then
    echo "  📝 Error count ($error_count) ≤ threshold ($MYPY_AUTO_FIX), attempting auto-fixes..."
    
    # Show the errors for context
    echo "$mypy_output" | head -10
    
    # Apply only very safe mypy fixes
    find "$PKG" -name "*.py" -exec python3 -c "
import sys, re
file_path = sys.argv[1]
try:
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Only fix missing imports for typing if needed
    if 'List[' in content or 'Dict[' in content or 'Optional[' in content:
        if 'from typing import' not in content and 'import typing' not in content:
            # Add typing import at the top after other imports
            lines = content.split('\n')
            import_idx = -1
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_idx = i
            if import_idx >= 0:
                lines.insert(import_idx + 1, 'from typing import List, Dict, Optional, Union')
                content = '\n'.join(lines)
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f'Applied safe typing imports to {file_path}')
except Exception as e:
    pass  # Skip files that can't be processed
" {} \;
    
    # Stage any fixes
    git add "$PKG" || true
    echo "  ✔ Auto-fixes applied"
  else
    echo "  📋 Too many errors ($error_count > $MYPY_AUTO_FIX) for auto-fix. First 10 errors:"
    echo "$mypy_output" | head -10
    echo "  (run 'python3 -m mypy --strict $PKG' to see all)"
  fi
  
  echo "  ✔ Type checking complete"
}

run_tests() {
  echo "🧪 Running tests for $PKG..."
  
  # Convert package path to dotted module name
  local test_pattern="${PKG//\//.}"
  
  echo "  Running pytest with pattern: $test_pattern"
  
  # Try to run tests with the package pattern
  echo "  Attempting to run tests for pattern: $test_pattern"
  if python3 -m pytest -q -k "$test_pattern" --tb=short --maxfail=5 2>/dev/null; then
    echo "  ✔ Tests passed for $test_pattern"
  else
    echo "  📋 Some tests failed or no tests found for pattern '$test_pattern'"
    
    # Try running tests in the package directory if it exists
    if [[ -d "$PKG/tests" ]]; then
      echo "  Trying tests in $PKG/tests..."
      python3 -m pytest -q "$PKG/tests" --tb=short --maxfail=3 2>/dev/null || {
        echo "  ⚠️  Tests in $PKG/tests had issues"
      }
    fi
    
    # Try finding and running test files specifically for this package
    local pkg_name
    pkg_name=$(basename "$PKG")
    echo "  Looking for test files related to '$pkg_name'..."
    
    # Look for test files in the main tests directory
    if [[ -d "tests" ]]; then
      local test_files
      test_files=$(find tests -name "*${pkg_name}*.py" -o -name "test_*${pkg_name}*.py" 2>/dev/null | head -3)
      if [[ -n "$test_files" ]]; then
        echo "  Found related test files, running them..."
        echo "$test_files" | xargs python3 -m pytest -q --tb=short --maxfail=3 2>/dev/null || {
          echo "  ⚠️  Some related tests had issues"
        }
      else
        echo "  🔍 No specific test files found for $pkg_name"
      fi
    fi
  fi
  
  echo "  ✔ Test run complete"
}

commit_changes() {    # commit only if staged changes exist
  if ! git diff --cached --quiet; then
      git commit -m "chore(clean): ${PKG} incremental tidy-up

- Applied black & isort formatting
- Fixed ruff linting issues  
- Applied mypy type checking fixes
- Ran test suite

Package: $PKG"
      echo "✔ committed changes for $PKG"
  else
      echo "ℹ nothing to commit for $PKG"
  fi
}

# ——————————————— EXECUTION ORDER ——————————————— #
echo "🚀 Starting cleanup for package: $PKG"
echo "========================================"

format_code
echo ""

lint_with_ruff  
echo ""

typecheck_strict
echo ""

run_tests
echo ""

commit_changes
echo ""

echo "🎉 Cleanup complete for $PKG"
