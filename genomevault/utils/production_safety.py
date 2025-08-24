"""Production safety checks to prevent silent failures."""

import os
import logging
from typing import Optional, Any, Dict
from functools import wraps
import traceback

logger = logging.getLogger(__name__)


class ProductionSafetyError(Exception):
    """Raised when production safety checks fail."""

    pass


def is_production() -> bool:
    """Check if running in production environment."""
    env = os.environ.get("GENOMEVAULT_ENV", "development")
    return env.lower() in ["production", "prod"]


def is_staging() -> bool:
    """Check if running in staging environment."""
    env = os.environ.get("GENOMEVAULT_ENV", "development")
    return env.lower() in ["staging", "stage"]


def require_real_backend(func):
    """Decorator to ensure real backend in production."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check if instance has real backend availability
        has_real = getattr(self, "has_real_backend", None)
        is_production_ready = getattr(self, "is_production_ready", None)
        circom_backend = getattr(self, "circom_backend", None)

        # Determine if real backend is available
        real_backend_available = False
        if has_real and callable(has_real):
            real_backend_available = has_real()
        elif is_production_ready is not None:
            real_backend_available = is_production_ready
        elif circom_backend is not None:
            real_backend_available = circom_backend is not None

        # Fail in production if no real backend
        if is_production() and not real_backend_available:
            error_msg = (
                f"{func.__name__} requires real cryptographic backend in production. "
                f"Mock/simulation backend detected - this creates a SECURITY VULNERABILITY. "
                f"Ensure Circom/SnarkJS are properly installed and configured."
            )
            logger.error(error_msg)
            raise ProductionSafetyError(error_msg)

        # Warn in staging
        if is_staging() and not real_backend_available:
            logger.warning(
                f"⚠️  STAGING WARNING: {func.__name__} using MOCK backend - "
                f"ensure real backend is configured before production deployment!"
            )

        # Log warning if using mock in development
        if not real_backend_available and not is_production():
            logger.warning(
                f"🔧 DEVELOPMENT: {func.__name__} using MOCK backend - "
                f"NOT suitable for production! Install Circom/SnarkJS for real proofs."
            )
            logger.info(
                "💡 To enable real backend: cd zk_circuits && npm install circomlib snarkjs && "
                "run ./scripts/install_complete_circomlib.sh"
            )

        return func(self, *args, **kwargs)

    return wrapper


def validate_not_mock(proof: Any, context: str = "proof validation") -> bool:
    """
    Validate that proof is not a mock proof.

    Args:
        proof: Proof object or dictionary to validate
        context: Context description for better error messages

    Returns:
        True if proof appears to be real, False if mock detected

    Raises:
        ProductionSafetyError: If mock proof detected in production
    """
    # Mock indicators in proof data
    mock_indicators = [
        "mock_signature",
        "_is_mock",
        "mock_proof",
        "test_proof",
        "simulation_mode",
        "dev_mode",
        "mock_",
        "test_",
    ]

    # Check proof object attributes
    if hasattr(proof, "__dict__"):
        proof_data = proof.__dict__
    elif isinstance(proof, dict):
        proof_data = proof
    else:
        # If we can't inspect the proof, assume it might be real
        return True

    # Convert to string for comprehensive checking
    proof_str = str(proof_data).lower()

    # Check for mock indicators
    for indicator in mock_indicators:
        if indicator in proof_data or indicator in proof_str:
            error_msg = f"Mock proof detected in {context}: contains '{indicator}'"

            if is_production():
                logger.error(f"🚨 PRODUCTION SECURITY VIOLATION: {error_msg}")
                logger.error(f"   Context: {context}")
                logger.error(f"   Environment: {os.environ.get('GENOMEVAULT_ENV', 'development')}")
                raise ProductionSafetyError(
                    f"Mock proof detected in production environment: contains '{indicator}'. "
                    f"Context: {context}. "
                    f"This is a critical security violation. Only real cryptographic proofs "
                    f"are allowed in production."
                )
            elif is_staging():
                logger.warning(f"⚠️  STAGING WARNING: {error_msg}")
                logger.warning("   This would FAIL in production! Fix before deployment.")
                return False
            else:
                logger.debug(f"🔧 Development mode: {error_msg}")
                logger.debug("   Install Circom/SnarkJS for real proofs in production")
                return False

    # Check for suspiciously small proof data (likely mock)
    if isinstance(proof_data, dict):
        # Real ZK proofs should have substantial cryptographic data
        proof_data_size = len(str(proof_data))
        if proof_data_size < 100:  # Real proofs are typically much larger
            error_msg = (
                f"Suspicious proof size in {context}: {proof_data_size} bytes (expected >100)"
            )

            if is_production():
                logger.error(f"🚨 PRODUCTION SECURITY VIOLATION: {error_msg}")
                logger.error(f"   Context: {context}")
                logger.error("   Real Groth16 proofs typically >500 bytes")
                raise ProductionSafetyError(
                    f"Proof data too small for real cryptographic proof: {proof_data_size} bytes. "
                    f"Context: {context}. "
                    f"This likely indicates a mock or invalid proof."
                )
            elif is_staging():
                logger.warning(f"⚠️  STAGING WARNING: {error_msg}")
                logger.warning("   Real proofs are typically >500 bytes")
                return False

    return True


def validate_proof_structure(proof: Any) -> bool:
    """
    Validate that proof has expected cryptographic structure.

    Args:
        proof: Proof to validate

    Returns:
        True if structure is valid

    Raises:
        ProductionSafetyError: If proof structure is invalid in production
    """
    # Expected fields in a real Groth16 proof
    expected_groth16_fields = ["pi_a", "pi_b", "pi_c", "protocol"]

    if hasattr(proof, "__dict__"):
        proof_data = proof.__dict__
    elif isinstance(proof, dict):
        proof_data = proof
    else:
        # Can't validate unknown structure
        return True

    # Check if this looks like a Groth16 proof
    has_groth16_fields = any(field in proof_data for field in expected_groth16_fields)

    if has_groth16_fields:
        missing_fields = [field for field in expected_groth16_fields if field not in proof_data]

        if missing_fields:
            error_msg = f"Invalid Groth16 proof structure: missing fields {missing_fields}"

            if is_production():
                logger.error(f"PRODUCTION SECURITY VIOLATION: {error_msg}")
                raise ProductionSafetyError(
                    f"Invalid proof structure in production: {error_msg}. "
                    f"Proof must contain all required Groth16 fields."
                )
            else:
                logger.warning(f"Invalid proof structure: {error_msg}")
                return False

    return True


def fail_loud_in_production(
    error_msg: str, exception: Optional[Exception] = None, context: str = "operation"
):
    """
    Fail loudly in production instead of silent fallback.

    Args:
        error_msg: Error message to log and raise
        exception: Optional original exception
        context: Context description for better debugging

    Raises:
        ProductionSafetyError: Always in production, optionally in staging
    """
    # Create comprehensive error message
    full_msg = f"🚨 GENOMEVAULT SAFETY VIOLATION in {context}: {error_msg}"

    if exception:
        full_msg += f"\nOriginal error: {str(exception)}"
        full_msg += f"\nOriginal traceback: {traceback.format_exc()}"

    # Add environment context
    env = os.environ.get("GENOMEVAULT_ENV", "development")
    full_msg += f"\nEnvironment: {env}"

    if is_production():
        logger.error(full_msg)
        logger.error(f"🔥 IMMEDIATE ACTION REQUIRED: Fix {context} to prevent service degradation")
        raise ProductionSafetyError(full_msg)
    elif is_staging():
        logger.error(f"⚠️  STAGING ERROR (would fail in prod): {full_msg}")
        logger.error("🚧 Fix before production deployment!")
        # Optionally fail in staging too for early detection
        strict_staging = os.environ.get("GENOMEVAULT_STRICT_STAGING", "false").lower() == "true"
        if strict_staging:
            raise ProductionSafetyError(f"Staging failure (strict mode): {error_msg}")
    else:
        logger.warning(f"🔧 Development warning (would fail in prod): {error_msg}")
        logger.info(f"💡 Context: {context}")
        if exception:
            logger.debug(f"Original exception: {exception}")


def require_secure_environment(operation_name: str = None):
    """Decorator to ensure function runs in secure environment."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__

            # Check for insecure environment variables
            insecure_env_vars = [
                "GENOMEVAULT_DISABLE_SECURITY",
                "GENOMEVAULT_FORCE_MOCK",
                "GENOMEVAULT_DEBUG_MODE",
            ]

            for var in insecure_env_vars:
                if os.environ.get(var, "").lower() in ["true", "1", "yes"]:
                    error_msg = f"Insecure environment variable {var} is enabled for {op_name}"

                    if is_production():
                        logger.error(f"🚨 SECURITY VIOLATION: {error_msg}")
                        raise ProductionSafetyError(
                            f"Security violation: {error_msg}. "
                            f"Debug/insecure flags cannot be enabled in production."
                        )
                    else:
                        logger.warning(f"🔧 Development warning: {error_msg}")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_cryptographic_strength(proof: Any, min_strength: int = 128) -> bool:
    """
    Validate cryptographic strength of proof.

    Args:
        proof: Proof to validate
        min_strength: Minimum security strength in bits (default 128)

    Returns:
        True if proof meets strength requirements
    """
    # This is a simplified check - in practice, you'd analyze the actual
    # cryptographic parameters used in the proof generation

    if hasattr(proof, "__dict__"):
        proof_data = proof.__dict__
    elif isinstance(proof, dict):
        proof_data = proof
    else:
        return True

    # Check if proof indicates security level
    security_indicators = ["security_level", "curve", "field_size"]

    for indicator in security_indicators:
        if indicator in proof_data:
            value = proof_data[indicator]
            if isinstance(value, str) and "bn128" in value.lower():
                # BN128 provides approximately 100 bits of security
                if min_strength > 100:
                    error_msg = f"Insufficient cryptographic strength: BN128 (~100 bits) < required {min_strength} bits"

                    if is_production():
                        logger.warning(f"Cryptographic strength warning: {error_msg}")
                        # Could optionally fail here for higher security requirements

                return min_strength <= 100

    # If we can't determine strength, assume it's adequate
    return True


def get_environment_info() -> Dict[str, Any]:
    """Get current environment information for debugging."""
    return {
        "environment": os.environ.get("GENOMEVAULT_ENV", "development"),
        "is_production": is_production(),
        "is_staging": is_staging(),
        "strict_staging": os.environ.get("GENOMEVAULT_STRICT_STAGING", "false").lower() == "true",
        "debug_enabled": os.environ.get("GENOMEVAULT_DEBUG_MODE", "false").lower() == "true",
        "security_disabled": os.environ.get("GENOMEVAULT_DISABLE_SECURITY", "false").lower()
        == "true",
    }


# Production safety configuration
PRODUCTION_SAFETY_CONFIG = {
    "require_real_backend": True,
    "validate_proof_structure": True,
    "validate_cryptographic_strength": True,
    "min_security_bits": 100,  # Minimum for BN128
    "fail_on_mock_proof": True,
    "strict_staging": False,
}
