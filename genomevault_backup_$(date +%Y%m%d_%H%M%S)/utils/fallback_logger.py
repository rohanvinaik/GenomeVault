"""Enhanced fallback logging for production safety."""

import logging
import os
from typing import Optional, Any
from datetime import datetime

from .production_safety import is_production, is_staging

logger = logging.getLogger(__name__)


class FallbackLogger:
    """Enhanced logging for backend fallbacks with production safety."""

    def __init__(self, component_name: str):
        self.component = component_name
        self.logger = logging.getLogger(f"genomevault.fallback.{component_name}")

    def log_fallback_attempt(
        self,
        operation: str,
        real_backend_name: str,
        mock_backend_name: str,
        reason: str,
        context: Optional[str] = None,
    ):
        """Log when attempting fallback from real to mock backend."""

        env_context = f"Environment: {os.environ.get('GENOMEVAULT_ENV', 'development')}"
        full_context = f"{env_context}"
        if context:
            full_context += f" | Context: {context}"

        if is_production():
            self.logger.error(
                f"🚨 PRODUCTION FALLBACK BLOCKED: {self.component}.{operation} "
                f"attempted fallback from {real_backend_name} to {mock_backend_name}. "
                f"Reason: {reason}. {full_context}"
            )
            self.logger.error("🔥 CRITICAL: Mock backends are FORBIDDEN in production!")

        elif is_staging():
            self.logger.warning(
                f"⚠️  STAGING FALLBACK WARNING: {self.component}.{operation} "
                f"falling back from {real_backend_name} to {mock_backend_name}. "
                f"Reason: {reason}. {full_context}"
            )
            self.logger.warning("🚧 Fix before production deployment!")

        else:
            self.logger.info(
                f"🔧 DEV FALLBACK: {self.component}.{operation} "
                f"using {mock_backend_name} (real: {real_backend_name}). "
                f"Reason: {reason}. {full_context}"
            )

    def log_successful_real_backend(self, operation: str, backend_name: str):
        """Log successful use of real backend."""
        self.logger.info(f"✅ {self.component}.{operation} using real backend: {backend_name}")

    def log_mock_backend_usage(self, operation: str, mock_name: str, warning: bool = True):
        """Log usage of mock backend."""
        if is_production():
            self.logger.error(
                f"🚨 PRODUCTION VIOLATION: {self.component}.{operation} "
                f"using mock backend {mock_name} in PRODUCTION!"
            )
        elif is_staging() and warning:
            self.logger.warning(
                f"⚠️  STAGING: {self.component}.{operation} using mock {mock_name} "
                f"(would FAIL in production)"
            )
        else:
            self.logger.debug(f"🔧 DEV: {self.component}.{operation} using mock {mock_name}")

    def log_backend_initialization(
        self, backend_name: str, is_real: bool, init_details: Optional[str] = None
    ):
        """Log backend initialization status."""
        backend_type = "REAL" if is_real else "MOCK"
        status_emoji = "🚀" if is_real else "🔧"

        msg = f"{status_emoji} {self.component} initialized with {backend_type} backend: {backend_name}"
        if init_details:
            msg += f" | {init_details}"

        if is_production() and not is_real:
            self.logger.error(f"🚨 PRODUCTION ERROR: {msg}")
        elif is_staging() and not is_real:
            self.logger.warning(f"⚠️  STAGING WARNING: {msg}")
        else:
            self.logger.info(msg)

    def log_configuration_check(self, config_items: dict[str, Any]):
        """Log configuration validation results."""
        timestamp = datetime.now().isoformat()

        self.logger.info(f"🔍 {self.component} Configuration Check [{timestamp}]")
        for key, value in config_items.items():
            if isinstance(value, bool):
                emoji = "✅" if value else "❌"
                self.logger.info(f"  {emoji} {key}: {value}")
            else:
                self.logger.info(f"  📋 {key}: {value}")

    def log_dependency_status(self, dependencies: dict[str, bool]):
        """Log external dependency availability."""
        self.logger.info(f"🔗 {self.component} Dependency Status:")

        missing_deps = []
        for dep_name, available in dependencies.items():
            if available:
                self.logger.info(f"  ✅ {dep_name}: Available")
            else:
                self.logger.warning(f"  ❌ {dep_name}: Missing")
                missing_deps.append(dep_name)

        if missing_deps and is_production():
            self.logger.error(
                f"🚨 PRODUCTION DEPENDENCY ERROR: Missing {missing_deps} " f"for {self.component}"
            )
        elif missing_deps and is_staging():
            self.logger.warning(f"⚠️  STAGING: Missing {missing_deps} - fix before production!")

    def log_performance_degradation(
        self, operation: str, expected_performance: str, actual_performance: str, impact: str
    ):
        """Log when fallback causes performance degradation."""
        msg = (
            f"📉 {self.component}.{operation} performance degradation: "
            f"Expected {expected_performance}, got {actual_performance}. "
            f"Impact: {impact}"
        )

        if is_production():
            self.logger.warning(f"⚠️  PRODUCTION PERFORMANCE: {msg}")
        else:
            self.logger.info(f"📊 PERFORMANCE: {msg}")


# Global fallback loggers for common components
zk_fallback_logger = FallbackLogger("ZKProofs")
hdc_fallback_logger = FallbackLogger("HDC")
pir_fallback_logger = FallbackLogger("PIR")
hardware_fallback_logger = FallbackLogger("Hardware")


def get_fallback_logger(component_name: str) -> FallbackLogger:
    """Get a fallback logger for the specified component."""
    return FallbackLogger(component_name)
