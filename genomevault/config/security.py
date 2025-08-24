"""Security configuration for production deployments."""

import os
import warnings


class SecurityConfig:
    """Security configuration for production deployments."""

    @staticmethod
    def check_production_mode() -> bool:
        """Check if running in production mode."""
        env = os.environ.get("GENOMEVAULT_ENV", "development")
        return env.lower() in ["production", "prod"]

    @staticmethod
    def validate_proof_backend(backend: str) -> None:
        """Validate proof backend is appropriate for environment.

        Args:
            backend: The proof backend name (e.g., 'mock', 'circom', 'gnark')

        Raises:
            RuntimeError: If mock backend is used in production
        """
        if SecurityConfig.check_production_mode() and backend.lower() == "mock":
            raise RuntimeError(
                "SECURITY ERROR: Mock proof backend cannot be used in production. "
                "Install Circom/snarkjs toolchain or set GENOMEVAULT_ENV=development"
            )

    @staticmethod
    def warn_mock_mode() -> None:
        """Issue warning when using mock proofs."""
        warnings.warn(
            "⚠️  WARNING: Using MOCK proof backend - NOT cryptographically secure! "
            "This is for development only. Install Circom toolchain for real proofs.",
            RuntimeWarning,
            stacklevel=2,
        )

    @staticmethod
    def get_allowed_backends() -> list[str]:
        """Get list of allowed proof backends for current environment.

        Returns:
            List of allowed backend names
        """
        if SecurityConfig.check_production_mode():
            # Production: only real cryptographic backends
            return ["circom", "gnark", "snarkjs", "plonk"]
        else:
            # Development: all backends including mock
            return ["circom", "gnark", "snarkjs", "plonk", "mock"]

    @staticmethod
    def validate_api_keys() -> None:
        """Validate that API keys are properly configured in production."""
        if SecurityConfig.check_production_mode():
            jwt_secret = os.environ.get("JWT_SECRET_KEY", "")
            if not jwt_secret or jwt_secret == "your-secret-key-here":
                raise RuntimeError(
                    "SECURITY ERROR: JWT_SECRET_KEY must be set to a secure value in production"
                )

            # Check for other critical security settings
            if os.environ.get("DEBUG", "false").lower() == "true":
                warnings.warn(
                    "⚠️  WARNING: DEBUG mode is enabled in production environment",
                    RuntimeWarning,
                    stacklevel=2,
                )

    @staticmethod
    def validate_encryption_keys() -> None:
        """Validate encryption keys are properly configured."""
        if SecurityConfig.check_production_mode():
            encryption_key = os.environ.get("ENCRYPTION_KEY", "")
            if not encryption_key:
                raise RuntimeError("SECURITY ERROR: ENCRYPTION_KEY must be set in production")

    @staticmethod
    def validate_all() -> None:
        """Run all security validations.

        Raises:
            RuntimeError: If any security validation fails
        """
        if SecurityConfig.check_production_mode():
            SecurityConfig.validate_api_keys()
            SecurityConfig.validate_encryption_keys()

            # Log that production security checks passed
            import logging

            logger = logging.getLogger(__name__)
            logger.info("✅ Production security validations passed")
