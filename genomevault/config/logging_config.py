"""Logging configuration presets for different environments."""

from pathlib import Path
from typing import Dict, Any
import logging
import os

from ..utils.logging import configure_logging


def get_development_config() -> Dict[str, Any]:
    """Get logging configuration for development environment."""
    return {
        "level": logging.DEBUG,
        "enable_file_logging": True,
        "enable_console_logging": True,
        "enable_json_logging": False,
        "max_bytes": 50 * 1024 * 1024,  # 50MB for development
        "backup_count": 3,
    }


def get_production_config() -> Dict[str, Any]:
    """Get logging configuration for production environment."""
    return {
        "level": logging.INFO,
        "enable_file_logging": True,
        "enable_console_logging": True,
        "enable_json_logging": True,  # JSON format for production monitoring
        "max_bytes": 100 * 1024 * 1024,  # 100MB for production
        "backup_count": 10,
    }


def get_testing_config() -> Dict[str, Any]:
    """Get logging configuration for testing environment."""
    return {
        "level": logging.WARNING,  # Reduce noise during tests
        "enable_file_logging": False,  # No file logging during tests
        "enable_console_logging": True,
        "enable_json_logging": False,
        "max_bytes": 10 * 1024 * 1024,  # 10MB
        "backup_count": 1,
    }


def get_staging_config() -> Dict[str, Any]:
    """Get logging configuration for staging environment."""
    return {
        "level": logging.INFO,
        "enable_file_logging": True,
        "enable_console_logging": True,
        "enable_json_logging": True,
        "max_bytes": 50 * 1024 * 1024,  # 50MB
        "backup_count": 5,
    }


def configure_for_environment(env: str = None) -> None:
    """Configure logging based on environment.

    Args:
        env: Environment name (development, production, testing, staging)
             If None, reads from GENOMEVAULT_ENV environment variable
    """
    if env is None:
        env = os.getenv("GENOMEVAULT_ENV", "development").lower()

    configs = {
        "development": get_development_config,
        "production": get_production_config,
        "testing": get_testing_config,
        "staging": get_staging_config,
    }

    if env not in configs:
        raise ValueError(f"Unknown environment: {env}. Supported: {list(configs.keys())}")

    config = configs[env]()

    # Override with environment-specific log directory if needed
    log_dir_env_var = f"GENOMEVAULT_{env.upper()}_LOG_DIR"
    if env_log_dir := os.getenv(log_dir_env_var):
        config["log_dir"] = Path(env_log_dir)

    configure_logging(**config)


def configure_for_docker() -> None:
    """Configure logging for Docker containers.

    This configuration is optimized for containerized environments:
    - Logs to stdout/stderr for container log aggregation
    - JSON format for structured logging
    - No file logging (containers are ephemeral)
    """
    configure_logging(
        level=logging.INFO,
        enable_file_logging=False,  # No file logging in containers
        enable_console_logging=True,
        enable_json_logging=True,  # Structured logs for aggregation
    )


def configure_for_kubernetes() -> None:
    """Configure logging for Kubernetes deployments.

    Similar to Docker but with additional considerations:
    - Structured logging for log aggregation
    - Performance-optimized configuration
    """
    log_level = logging.INFO

    # Use WARNING level in production Kubernetes
    if os.getenv("GENOMEVAULT_ENV") == "production":
        log_level = logging.WARNING

    configure_logging(
        level=log_level,
        enable_file_logging=False,
        enable_console_logging=True,
        enable_json_logging=True,
    )


def configure_for_lambda() -> None:
    """Configure logging for AWS Lambda functions.

    Lambda-specific configuration:
    - Console logging only (CloudWatch integration)
    - Minimal overhead
    - JSON format for CloudWatch insights
    """
    configure_logging(
        level=logging.INFO,
        enable_file_logging=False,
        enable_console_logging=True,
        enable_json_logging=True,
    )


# Environment-specific configuration mapping
ENVIRONMENT_CONFIGS = {
    "development": configure_for_environment,
    "production": configure_for_environment,
    "testing": configure_for_environment,
    "staging": configure_for_environment,
    "docker": configure_for_docker,
    "kubernetes": configure_for_kubernetes,
    "lambda": configure_for_lambda,
}


def auto_configure() -> None:
    """Automatically configure logging based on detected environment."""
    # Check for container environments
    if os.path.exists("/.dockerenv") or os.getenv("KUBERNETES_SERVICE_HOST"):
        if os.getenv("KUBERNETES_SERVICE_HOST"):
            configure_for_kubernetes()
        else:
            configure_for_docker()
    elif os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        configure_for_lambda()
    else:
        # Use standard environment-based configuration
        configure_for_environment()
