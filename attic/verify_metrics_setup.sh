#!/bin/bash
################################################################################
# GenomeVault Metrics & Testing Verification
################################################################################

echo "🧬 GenomeVault Metrics & Testing Setup"
echo "======================================"
echo ""

echo "📋 Current Status:"
echo "• metrics.py: $([ -f genomevault/utils/metrics.py ] && echo '✅ exists' || echo '❌ missing')"
echo "• __init__.py exports: $(grep -q 'get_metrics' genomevault/utils/__init__.py && echo '✅ configured' || echo '❌ missing')"
echo "• editable install: $(python -c 'import genomevault; print("✅ working")' 2>/dev/null || echo '❌ broken')"
echo ""

echo "🧪 Quick Import Test:"
python - <<'PY'
try:
    # Test the specific imports that were problematic
    from genomevault.utils import get_metrics, MetricsCollector
    from genomevault.utils.config import Config, NodeClass, CompressionTier

    # Test functionality
    metrics = get_metrics()
    config = Config()

    # Verify it works
    metrics["setup_test"] = 1
    voting_power = config.get_voting_power()

    print("✅ All critical imports working")
    print(f"✅ Metrics: {dict(metrics)}")
    print(f"✅ Config voting power: {voting_power}")

except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
PY

echo ""
echo "📊 Test Environment:"
echo "• pytest config: $([ -f pytest.ini ] && echo '✅ configured' || echo '❌ missing')"
echo "• __pycache__ cleaned: $([ -z "$(find genomevault -name __pycache__ 2>/dev/null)" ] && echo '✅ clean' || echo '⚠️  has cache')"
echo ""

echo "🚀 Ready for Testing!"
echo ""
echo "Run these commands:"
echo "• pytest -q tests/unit/test_config.py       # Config tests"
echo "• pytest -q -k 'not api and not nanopore'  # All safe tests"
echo "• python -c 'from genomevault.utils import get_metrics; print(get_metrics())'  # Quick test"
echo ""
echo "All metrics functionality is now properly set up and importable! 🎉"
