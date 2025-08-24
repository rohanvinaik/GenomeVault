"""Logging module for GenomeVault observability."""

# Import from the structured logging module
from .structured import (
    get_structured_logger,
    configure_structured_logging,
    set_request_context,
    generate_request_id,
    get_request_id,
    GenomeVaultLogger,
)

# Import from parent logging.py module to maintain compatibility
import os
import sys
import logging

_LEVEL = os.getenv("GENOMEVAULT_LOG_LEVEL", "INFO").upper()


def configure_logging() -> logging.Logger:
    """Configure basic logging (compatibility function)."""
    logger = logging.getLogger("genomevault")
    if not logger.handlers:
        logger.setLevel(_LEVEL)
        h = logging.StreamHandler(stream=sys.stderr)
        fmt = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a basic logger instance (compatibility function)."""
    return configure_logging()


__all__ = [
    "get_structured_logger",
    "configure_structured_logging",
    "set_request_context",
    "generate_request_id",
    "get_request_id",
    "GenomeVaultLogger",
    "configure_logging",
    "get_logger",
]
