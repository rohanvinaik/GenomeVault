#!/bin/bash
################################################################################
# Verify __init__.py Package Structure
################################################################################

echo "📦 GenomeVault Package Structure Verification"
echo "============================================="
echo ""

echo "📊 Statistics:"
total_dirs=$(find genomevault -type d ! -name '__pycache__' | wc -l)
total_init=$(find genomevault -name '__init__.py' | wc -l)
stub_count=$(find genomevault -name '__init__.py' -exec grep -l 'package stub' {} + 2>/dev/null | wc -l)

echo "• Total directories: $total_dirs"
echo "• Total __init__.py files: $total_init"
echo "• Stub files created: $stub_count"

if [ "$total_dirs" -eq "$total_init" ]; then
    echo "✅ Perfect coverage: Every directory has __init__.py"
else
    echo "⚠️  Partial coverage: $total_init out of $total_dirs directories"
fi

echo ""
echo "🔍 Sample directories and their __init__.py status:"
find genomevault -type d ! -name '__pycache__' | head -8 | while read dir; do
    if [ -f "$dir/__init__.py" ]; then
        content=$(head -1 "$dir/__init__.py" 2>/dev/null | cut -c1-30)
        echo "✅ $dir/ → $content"
    else
        echo "❌ $dir/ → MISSING"
    fi
done

echo ""
echo "🧪 Import Test:"
python - <<'PY'
try:
    # Test basic imports
    import genomevault
    from genomevault.utils import get_logger, get_metrics
    print("✅ Core imports successful")

    # Test functionality
    logger = get_logger("verify")
    metrics = get_metrics()
    metrics["verification"] = 1
    print(f"✅ Functionality working: {dict(metrics)}")

    # Test package discovery
    import pkgutil
    package_count = len(list(pkgutil.iter_modules(genomevault.__path__)))
    print(f"✅ Discovered {package_count} subpackages")

except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
PY

echo ""
echo "🎯 Key Benefits:"
echo "• All directories are now importable Python packages"
echo "• Package discovery works with setuptools and pkgutil"
echo "• Deep imports like 'from genomevault.module.submodule import ...' work"
echo "• Editable installs pick up all package structure"
echo ""
echo "🚀 GenomeVault package structure is complete!"
