"""
Security Integration Module for GenomeVault.

Provides a unified interface for integrating all security middleware
components into the FastAPI application with proper configuration
for different deployment environments.
"""

import os
from typing import Optional, List, Dict, Any
from enum import Enum

from fastapi import FastAPI, Request

# Import security middleware components
from .middleware.rate_limiting import create_rate_limit_middleware
from .middleware.input_sanitization import InputSanitizationMiddleware
from .middleware.audit_logging import AuditMiddleware, get_audit_logger
from .middleware.cors_security import create_cors_middleware, CORSSecurityLevel
from .middleware.security_headers import create_security_headers_middleware, SecurityProfile


class SecurityEnvironment(str, Enum):
    """Security environment configurations."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    CLINICAL = "clinical"


class GenomeVaultSecurity:
    """Unified security configuration manager for GenomeVault."""

    def __init__(
        self,
        environment: SecurityEnvironment = SecurityEnvironment.PRODUCTION,
        enable_rate_limiting: bool = True,
        enable_input_sanitization: bool = True,
        enable_audit_logging: bool = True,
        enable_cors: bool = True,
        enable_security_headers: bool = True,
        redis_url: Optional[str] = None,
        allowed_origins: Optional[List[str]] = None,
    ):
        """Initialize security configuration.

        Args:
            environment: Deployment environment
            enable_rate_limiting: Enable rate limiting middleware
            enable_input_sanitization: Enable input sanitization
            enable_audit_logging: Enable audit logging
            enable_cors: Enable CORS middleware
            enable_security_headers: Enable security headers
            redis_url: Redis URL for rate limiting and caching
            allowed_origins: Override CORS allowed origins
        """
        self.environment = environment
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_input_sanitization = enable_input_sanitization
        self.enable_audit_logging = enable_audit_logging
        self.enable_cors = enable_cors
        self.enable_security_headers = enable_security_headers
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.allowed_origins = allowed_origins

        # Configure security levels based on environment
        self._configure_security_levels()

    def _configure_security_levels(self):
        """Configure security levels based on environment."""
        if self.environment == SecurityEnvironment.CLINICAL:
            self.cors_security_level = CORSSecurityLevel.CLINICAL
            self.security_profile = SecurityProfile.CLINICAL
            self.strict_mode = True
        elif self.environment == SecurityEnvironment.PRODUCTION:
            self.cors_security_level = CORSSecurityLevel.STRICT
            self.security_profile = SecurityProfile.PRODUCTION
            self.strict_mode = True
        elif self.environment == SecurityEnvironment.STAGING:
            self.cors_security_level = CORSSecurityLevel.STANDARD
            self.security_profile = SecurityProfile.STAGING
            self.strict_mode = False
        else:  # DEVELOPMENT or TESTING
            self.cors_security_level = CORSSecurityLevel.PERMISSIVE
            self.security_profile = SecurityProfile.DEVELOPMENT
            self.strict_mode = False

    def configure_app(self, app: FastAPI) -> FastAPI:
        """Apply all security middleware to the FastAPI application.

        Args:
            app: FastAPI application instance

        Returns:
            Configured FastAPI application
        """
        print(f"🔒 Configuring GenomeVault security for {self.environment} environment...")

        # Apply middleware in reverse order (they execute in LIFO order)

        # 1. Security Headers (outermost)
        if self.enable_security_headers:
            middleware_factory = create_security_headers_middleware(profile=self.security_profile)
            app.add_middleware(middleware_factory)
            print(f"  ✅ Security headers configured ({self.security_profile})")

        # 2. CORS (before other security checks)
        if self.enable_cors:
            if self.allowed_origins is None:
                self.allowed_origins = self._get_default_origins()

            middleware_factory = create_cors_middleware(
                allowed_origins=self.allowed_origins, security_level=self.cors_security_level
            )
            app.add_middleware(middleware_factory)
            print(f"  ✅ CORS configured ({self.cors_security_level})")

        # 3. Audit Logging (log all requests)
        if self.enable_audit_logging:
            audit_logger = get_audit_logger()
            app.add_middleware(AuditMiddleware, audit_logger=audit_logger)
            print("  ✅ Audit logging configured")

        # 4. Rate Limiting
        if self.enable_rate_limiting:
            middleware_factory = create_rate_limit_middleware(
                redis_url=self.redis_url, enable_rate_limiting=True
            )
            app.add_middleware(middleware_factory)
            print("  ✅ Rate limiting configured")

        # 5. Input Sanitization (innermost - validates input first)
        if self.enable_input_sanitization:
            app.add_middleware(InputSanitizationMiddleware)
            print("  ✅ Input sanitization configured")

        # Add security info to app state
        app.state.security_config = self

        print("🛡️  GenomeVault security configuration complete!")
        return app

    def _get_default_origins(self) -> List[str]:
        """Get default allowed origins based on environment."""
        if self.environment == SecurityEnvironment.CLINICAL:
            # Clinical environments should have very restrictive origins
            return ["https://clinical.genomevault.com", "https://secure.genomevault.com"]
        elif self.environment == SecurityEnvironment.PRODUCTION:
            return [
                "https://genomevault.com",
                "https://www.genomevault.com",
                "https://api.genomevault.com",
                "https://app.genomevault.com",
            ]
        elif self.environment == SecurityEnvironment.STAGING:
            return [
                "https://staging.genomevault.com",
                "https://staging-api.genomevault.com",
                "http://localhost:3000",
                "http://localhost:8080",
            ]
        else:  # DEVELOPMENT or TESTING
            return [
                "http://localhost:3000",
                "http://localhost:8080",
                "http://localhost:5173",  # Vite
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8080",
                "http://0.0.0.0:3000",
            ]

    def get_security_info(self) -> Dict[str, Any]:
        """Get information about current security configuration."""
        return {
            "environment": self.environment.value,
            "cors_security_level": self.cors_security_level.value,
            "security_profile": self.security_profile.value,
            "strict_mode": self.strict_mode,
            "middleware_enabled": {
                "rate_limiting": self.enable_rate_limiting,
                "input_sanitization": self.enable_input_sanitization,
                "audit_logging": self.enable_audit_logging,
                "cors": self.enable_cors,
                "security_headers": self.enable_security_headers,
            },
            "allowed_origins": self.allowed_origins,
            "redis_configured": bool(self.redis_url),
        }


def configure_security(app: FastAPI, environment: Optional[str] = None, **kwargs) -> FastAPI:
    """Convenient function to configure security for a FastAPI app.

    Args:
        app: FastAPI application
        environment: Environment name (overrides environment detection)
        **kwargs: Additional configuration options

    Returns:
        Configured FastAPI application
    """

    # Auto-detect environment if not specified
    if environment is None:
        environment = os.getenv("GENOMEVAULT_ENV", "production").lower()
        clinical_mode = os.getenv("GENOMEVAULT_CLINICAL_MODE", "false").lower() == "true"

        if clinical_mode:
            environment = "clinical"

    try:
        env_enum = SecurityEnvironment(environment)
    except ValueError:
        print(f"Warning: Unknown environment '{environment}', defaulting to production")
        env_enum = SecurityEnvironment.PRODUCTION

    # Create security configuration
    security = GenomeVaultSecurity(environment=env_enum, **kwargs)

    # Apply configuration
    return security.configure_app(app)


def create_secure_app(
    title: str = "GenomeVault API",
    description: str = "Privacy-preserving genomic computing platform",
    version: str = "1.0.0",
    environment: Optional[str] = None,
    **security_kwargs,
) -> FastAPI:
    """Create a FastAPI application with security pre-configured.

    Args:
        title: API title
        description: API description
        version: API version
        environment: Deployment environment
        **security_kwargs: Additional security configuration

    Returns:
        Fully configured secure FastAPI application
    """

    # Create FastAPI app
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        docs_url="/docs" if environment != "clinical" else None,  # Disable docs in clinical
        redoc_url="/redoc" if environment != "clinical" else None,
        openapi_url="/openapi.json" if environment != "clinical" else None,
    )

    # Apply security configuration
    app = configure_security(app, environment=environment, **security_kwargs)

    # Add security info endpoint (for monitoring)
    @app.get("/security/info", include_in_schema=False)
    async def security_info(request: Request):
        """Get security configuration information."""
        if hasattr(request.app.state, "security_config"):
            return request.app.state.security_config.get_security_info()
        return {"status": "Security configuration not found"}

    return app


# Environment-specific configurations
SECURITY_CONFIGS = {
    SecurityEnvironment.DEVELOPMENT: {
        "enable_rate_limiting": False,  # Disabled for development
        "enable_input_sanitization": True,
        "enable_audit_logging": False,  # Disabled for development
        "enable_cors": True,
        "enable_security_headers": True,
    },
    SecurityEnvironment.TESTING: {
        "enable_rate_limiting": False,  # Disabled for testing
        "enable_input_sanitization": True,
        "enable_audit_logging": False,  # Disabled for testing
        "enable_cors": False,  # Disabled for testing
        "enable_security_headers": False,  # Disabled for testing
    },
    SecurityEnvironment.STAGING: {
        "enable_rate_limiting": True,
        "enable_input_sanitization": True,
        "enable_audit_logging": True,
        "enable_cors": True,
        "enable_security_headers": True,
    },
    SecurityEnvironment.PRODUCTION: {
        "enable_rate_limiting": True,
        "enable_input_sanitization": True,
        "enable_audit_logging": True,
        "enable_cors": True,
        "enable_security_headers": True,
    },
    SecurityEnvironment.CLINICAL: {
        "enable_rate_limiting": True,
        "enable_input_sanitization": True,
        "enable_audit_logging": True,  # Critical for clinical
        "enable_cors": True,
        "enable_security_headers": True,
    },
}


def get_security_config(environment: SecurityEnvironment) -> Dict[str, Any]:
    """Get security configuration for an environment."""
    return SECURITY_CONFIGS.get(environment, SECURITY_CONFIGS[SecurityEnvironment.PRODUCTION])


# Example usage in main application
if __name__ == "__main__":
    # Create a secure FastAPI application
    app = create_secure_app(
        title="GenomeVault Secure API",
        description="HIPAA-compliant genomic computing API with comprehensive security",
        environment="production",
    )

    @app.get("/")
    async def root():
        return {"message": "GenomeVault Secure API", "status": "protected"}

    @app.get("/health")
    async def health():
        return {"status": "healthy", "security": "enabled"}

    # In a real application, you would run this with:
    # uvicorn genomevault.api.security_integration:app --host 0.0.0.0 --port 8000
