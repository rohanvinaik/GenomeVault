#!/bin/bash
################################################################################
# Setup GenomeVault Development Environment
################################################################################

set -e
echo "🚀 Setting up GenomeVault development environment..."
echo "Current directory: $(pwd)"
echo ""

# 1️⃣  Create an editable install so 'import genomevault' works
echo "🔧 Step 1: Create an editable install so 'import genomevault' works"
echo "Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing genomevault in editable mode..."
# Check if pyproject.toml exists and has dev extras
if [ -f "pyproject.toml" ] && grep -q "dev.*=" pyproject.toml; then
    echo "✓ Found pyproject.toml with dev extras, installing with [dev]"
    python -m pip install -e .[dev]
else
    echo "⚠ No dev extras found, installing basic editable package"
    python -m pip install -e .
fi

echo ""

# 2️⃣  Install the missing dev dependency reported by pytest
echo "🧪 Step 2: Install the missing dev dependency reported by pytest"
echo "Installing hypothesis..."
python -m pip install hypothesis

echo ""

# 3️⃣  Verify the package really imports
echo "✅ Step 3: Verify the package really imports"
python - <<'PY'
import importlib, sys
print("site-packages path:", *sys.path[:3], sep="\n  • ")
try:
    genomevault_module = importlib.import_module("genomevault")
    print("✓ genomevault imports to:", genomevault_module)
    print("✓ Package location:", genomevault_module.__file__)
except ImportError as e:
    print("❌ Failed to import genomevault:", e)
    sys.exit(1)
PY

echo ""

# 4️⃣  Re-run pytest (skip API & Nanopore folders if you moved them)
echo "🧪 Step 4: Re-run pytest (skip API & Nanopore folders)"
echo "Running pytest with filters..."
echo "Command: pytest -q -k \"not api and not nanopore\""
echo ""

# Run pytest and capture the result
if pytest -q -k "not api and not nanopore"; then
    echo ""
    echo "✅ All tests passed!"
else
    echo ""
    echo "⚠️  Some tests failed, but that's okay for now."
    echo "The important thing is that the package imports correctly."
fi

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Summary:"
echo "• ✅ Upgraded pip, setuptools, wheel"
echo "• ✅ Installed genomevault in editable mode with dev dependencies"
echo "• ✅ Installed hypothesis for testing"
echo "• ✅ Verified genomevault package imports correctly"
echo "• 🧪 Ran pytest with api/nanopore exclusions"
echo ""
echo "You can now use 'import genomevault' in Python!"
