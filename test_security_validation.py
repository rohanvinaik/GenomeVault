#!/usr/bin/env python3
"""Test security validation for ZK proof backend."""

import os
import sys
import warnings


def test_development_mode():
    """Test that mock backend works in development mode."""
    print("Testing development mode...")

    # Set development environment
    os.environ["GENOMEVAULT_ENV"] = "development"

    # Import after setting environment
    from genomevault.zk_proofs.prover import Prover

    try:
        # Should work fine in development
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prover = Prover(use_circom=False)

            # Check that warning was issued
            warning_messages = [str(warning.message) for warning in w]
            has_mock_warning = any("MOCK proof backend" in msg for msg in warning_messages)

            if has_mock_warning:
                print("✅ Development mode: Mock backend allowed with warning")
            else:
                print("⚠️  Development mode: Warning not issued for mock backend")

        # Try to generate a proof
        # Don't provide variant_hash - let it be generated from variant_data
        proof = prover.prove_variant(
            public_input={},
            private_input={"variant_data": {"chr": "1", "pos": 123, "ref": "A", "alt": "G"}},
        )
        print(f"✅ Development mode: Proof generated successfully (ID: {proof.proof_id[:8]}...)")

    except Exception as e:
        print(f"❌ Development mode failed: {e}")
        return False

    return True


def test_production_mode():
    """Test that mock backend is blocked in production mode."""
    print("\nTesting production mode...")

    # Set production environment
    os.environ["GENOMEVAULT_ENV"] = "production"

    # Clear any cached modules
    if "genomevault.zk_proofs.prover" in sys.modules:
        del sys.modules["genomevault.zk_proofs.prover"]
    if "genomevault.config.security" in sys.modules:
        del sys.modules["genomevault.config.security"]

    # Import after setting environment
    from genomevault.config.security import SecurityConfig

    # First check that we're in production mode
    if SecurityConfig.check_production_mode():
        print("✅ Production mode detected")
    else:
        print("❌ Production mode not detected")
        return False

    try:
        # Try to validate mock backend - should fail
        SecurityConfig.validate_proof_backend("mock")
        print("❌ Production mode: Mock backend was allowed (should have been blocked)")
        return False
    except RuntimeError as e:
        if "Mock proof backend cannot be used in production" in str(e):
            print("✅ Production mode: Mock backend correctly blocked")
        else:
            print(f"❌ Production mode: Wrong error message: {e}")
            return False

    # Try to validate real backend - should work
    try:
        SecurityConfig.validate_proof_backend("circom")
        print("✅ Production mode: Real backend (circom) allowed")
    except Exception as e:
        print(f"❌ Production mode: Real backend blocked: {e}")
        return False

    return True


def test_api_key_validation():
    """Test API key validation in production."""
    print("\nTesting API key validation...")

    # Set production environment
    os.environ["GENOMEVAULT_ENV"] = "production"

    # Clear JWT secret if set
    if "JWT_SECRET_KEY" in os.environ:
        del os.environ["JWT_SECRET_KEY"]

    from genomevault.config.security import SecurityConfig

    # Test missing JWT secret
    try:
        SecurityConfig.validate_api_keys()
        print("❌ API key validation: Missing JWT secret was allowed")
        return False
    except RuntimeError as e:
        if "JWT_SECRET_KEY must be set" in str(e):
            print("✅ API key validation: Missing JWT secret correctly blocked")
        else:
            print(f"❌ API key validation: Wrong error: {e}")
            return False

    # Test insecure JWT secret
    os.environ["JWT_SECRET_KEY"] = "your-secret-key-here"
    try:
        SecurityConfig.validate_api_keys()
        print("❌ API key validation: Insecure JWT secret was allowed")
        return False
    except RuntimeError as e:
        if "JWT_SECRET_KEY must be set to a secure value" in str(e):
            print("✅ API key validation: Insecure JWT secret correctly blocked")
        else:
            print(f"❌ API key validation: Wrong error: {e}")
            return False

    # Test secure JWT secret
    os.environ["JWT_SECRET_KEY"] = "super-secure-random-key-abc123xyz789"
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SecurityConfig.validate_api_keys()
            print("✅ API key validation: Secure JWT secret accepted")

            # Check for debug mode warning if DEBUG is set
            os.environ["DEBUG"] = "true"
            SecurityConfig.validate_api_keys()
            warning_messages = [str(warning.message) for warning in w]
            if any("DEBUG mode is enabled in production" in msg for msg in warning_messages):
                print("✅ API key validation: Debug mode warning issued")
    except Exception as e:
        print(f"❌ API key validation failed: {e}")
        return False

    return True


def main():
    """Run all security validation tests."""
    print("=" * 60)
    print("🔒 TESTING SECURITY VALIDATION")
    print("=" * 60)

    all_passed = True

    # Test 1: Development mode
    if not test_development_mode():
        all_passed = False

    # Test 2: Production mode
    if not test_production_mode():
        all_passed = False

    # Test 3: API key validation
    if not test_api_key_validation():
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL SECURITY VALIDATION TESTS PASSED")
    else:
        print("❌ SOME SECURITY VALIDATION TESTS FAILED")
    print("=" * 60)

    # Reset environment
    os.environ["GENOMEVAULT_ENV"] = "development"
    if "JWT_SECRET_KEY" in os.environ:
        del os.environ["JWT_SECRET_KEY"]
    if "DEBUG" in os.environ:
        del os.environ["DEBUG"]

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
