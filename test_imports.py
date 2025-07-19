#!/usr/bin/env python3
"""
Test script to verify Pydantic imports are working correctly
"""

import sys


def test_imports():
    """Test that all critical imports work"""
    print("🧪 Testing GenomeVault imports...")

    try:
        # Test pydantic imports
        print("  📦 Testing pydantic imports...")
        from pydantic import Field, field_validator
        from pydantic_settings import BaseSettings

        print("  ✅ Pydantic imports successful")

        # Test config import
        print("  📦 Testing config module...")
        from core.config import Config, get_config

        print("  ✅ Config module imported successfully")

        # Test creating a config instance
        print("  📦 Testing config instantiation...")
        config = get_config()
        print(
            f"  ✅ Config created: node_type={config.node_type}, voting_power={config.total_voting_power}"
        )

        # Test other critical imports
        print("  📦 Testing other module imports...")
        from hypervector_transform import HypervectorEncoder
        from local_processing import SequencingEngine, TranscriptomicsProcessor
        from zk_proofs import VariantProver

        print("  ✅ All modules imported successfully!")

        return True

    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False


def main():
    """Main test function"""
    print("=" * 50)
    print("GenomeVault Import Test")
    print("=" * 50)

    success = test_imports()

    if success:
        print("\n✅ All tests passed! Your environment is ready.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed. Please run: ./fix_dependencies.sh")
        sys.exit(1)


if __name__ == "__main__":
    main()
