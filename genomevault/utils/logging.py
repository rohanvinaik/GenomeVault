"""Centralized logging configuration for GenomeVault.

This module provides comprehensive logging functionality including:
- Environment variable configuration
- Log rotation and file handlers
- Different log levels for different components
- Structured logging with JSON support
- Performance logging capabilities
"""

import json
import logging
import logging.config
import logging.handlers
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

# Configuration constants
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
DETAILED_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)s:%(lineno)d | %(message)s"
)
JSON_FORMAT = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'

# Global configuration flag
_logging_configured = False


def get_log_level_from_env() -> int:
    """Get log level from environment variable."""
    level_name = os.getenv("GENOMEVAULT_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    return getattr(logging, level_name, logging.INFO)


def get_log_dir() -> Path:
    """Get the logging directory, creating it if necessary."""
    log_dir = Path(os.getenv("GENOMEVAULT_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def configure_logging(
    level: Optional[int] = None,
    log_dir: Optional[Path] = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    enable_json_logging: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    force_reconfigure: bool = False,
) -> None:
    """Configure logging for GenomeVault.

    Args:
        level: Log level (uses environment variable if None)
        log_dir: Directory for log files (uses environment variable if None)
        enable_file_logging: Whether to enable file logging
        enable_console_logging: Whether to enable console logging
        enable_json_logging: Whether to use JSON format for file logs
        max_bytes: Maximum size of log files before rotation
        backup_count: Number of backup log files to keep
        force_reconfigure: Force reconfiguration even if already configured
    """
    global _logging_configured

    if _logging_configured and not force_reconfigure:
        return

    # Clear existing handlers if reconfiguring
    if force_reconfigure:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

    # Get configuration values
    if level is None:
        level = get_log_level_from_env()
    if log_dir is None:
        log_dir = get_log_dir()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    handlers: List[Union[logging.StreamHandler, logging.handlers.RotatingFileHandler]] = []

    # Console handler
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)

        # Use detailed format for DEBUG level
        console_format = DETAILED_FORMAT if level <= logging.DEBUG else DEFAULT_LOG_FORMAT
        console_formatter = logging.Formatter(console_format)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

    # File handlers
    if enable_file_logging:
        # Ensure log directory exists
        log_dir.mkdir(parents=True, exist_ok=True)

        # Main application log
        app_log_file = log_dir / "genomevault.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        app_handler.setLevel(level)

        # Use JSON format if requested
        if enable_json_logging:
            app_formatter = logging.Formatter(JSON_FORMAT)
        else:
            app_formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
        app_handler.setFormatter(app_formatter)
        handlers.append(app_handler)

        # Error log (only ERROR and CRITICAL)
        error_log_file = log_dir / "genomevault_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(DETAILED_FORMAT)
        error_handler.setFormatter(error_formatter)
        handlers.append(error_handler)

        # Performance log
        perf_log_file = log_dir / "genomevault_performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        perf_handler.setLevel(logging.INFO)
        perf_formatter = logging.Formatter("%(asctime)s | PERF | %(message)s")
        perf_handler.setFormatter(perf_formatter)

        # Only add performance logs to performance logger
        perf_logger = logging.getLogger("genomevault.performance")
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False  # Don't propagate to root logger

    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)

    # Set specific logger levels
    component_levels = {
        "genomevault.hypervector": get_component_log_level("HYPERVECTOR"),
        "genomevault.zk_proofs": get_component_log_level("ZK_PROOFS"),
        "genomevault.pir": get_component_log_level("PIR"),
        "genomevault.federated": get_component_log_level("FEDERATED"),
        "genomevault.blockchain": get_component_log_level("BLOCKCHAIN"),
        "genomevault.api": get_component_log_level("API"),
    }

    for logger_name, component_level in component_levels.items():
        if component_level is not None:
            logging.getLogger(logger_name).setLevel(component_level)

    _logging_configured = True

    # Log the configuration
    config_logger = logging.getLogger("genomevault.logging")
    config_logger.info(
        f"Logging configured: level={logging.getLevelName(level)}, file_logging={enable_file_logging}"
    )
    config_logger.debug(f"Log directory: {log_dir}")


def get_component_log_level(component: str) -> Optional[int]:
    """Get log level for a specific component from environment variables."""
    env_var = f"GENOMEVAULT_{component}_LOG_LEVEL"
    level_name = os.getenv(env_var)
    if level_name:
        return getattr(logging, level_name.upper(), None)
    return None


def get_logger(name: str | None = None, level: Optional[int] = None) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (uses calling module if None)
        level: Override log level for this logger

    Returns:
        Configured logger instance
    """
    # Ensure logging is configured
    if not _logging_configured:
        configure_logging()

    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    return logger


def log_performance(operation: str, duration: float, **metadata: Any) -> None:
    """Log performance metrics.

    Args:
        operation: Name of the operation
        duration: Duration in seconds
        **metadata: Additional metadata to include
    """
    perf_logger = get_logger("genomevault.performance")

    message_data = {
        "operation": operation,
        "duration_ms": round(duration * 1000, 2),
        **metadata,
    }

    # Log as JSON for easy parsing
    perf_logger.info(json.dumps(message_data))


def log_operation(func: Callable) -> Callable:
    """Decorator to automatically log function execution time and results.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        operation_name = f"{func.__module__}.{func.__name__}"

        start_time = time.perf_counter()

        try:
            logger.debug(f"Starting {operation_name}")
            result = func(*args, **kwargs)

            duration = time.perf_counter() - start_time
            logger.debug(f"Completed {operation_name} in {duration:.3f}s")

            # Log performance metrics for significant operations
            if duration > 0.1:  # Log operations taking more than 100ms
                log_performance(operation_name, duration)

            return result

        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"Failed {operation_name} after {duration:.3f}s: {e}")
            raise

    return wrapper


class ContextLogger:
    """Context manager for logging with additional context."""

    def __init__(self, logger: logging.Logger, operation: str, **context: Any):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        self.logger.info(
            f"Starting {self.operation}" + (f" ({context_str})" if context_str else "")
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time

        if exc_type is None:
            self.logger.info(f"Completed {self.operation} in {duration:.3f}s")
            log_performance(self.operation, duration, **self.context)
        else:
            self.logger.error(f"Failed {self.operation} after {duration:.3f}s: {exc_val}")

        return False  # Don't suppress exceptions


# Convenience functions for different log levels
def debug(message: str, logger_name: Optional[str] = None) -> None:
    """Log a debug message."""
    get_logger(logger_name).debug(message)


def info(message: str, logger_name: Optional[str] = None) -> None:
    """Log an info message."""
    get_logger(logger_name).info(message)


def warning(message: str, logger_name: Optional[str] = None) -> None:
    """Log a warning message."""
    get_logger(logger_name).warning(message)


def error(message: str, logger_name: Optional[str] = None) -> None:
    """Log an error message."""
    get_logger(logger_name).error(message)


def critical(message: str, logger_name: Optional[str] = None) -> None:
    """Log a critical message."""
    get_logger(logger_name).critical(message)


# Default logger instances for backward compatibility
logger = get_logger(__name__)
performance_logger = get_logger("genomevault.performance")


# Initialize logging when module is imported
# This ensures basic logging is always available
configure_logging()
