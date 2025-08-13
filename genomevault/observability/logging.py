"""Logging module."""

from __future__ import annotations

import logging
import os
import sys

_LEVEL = os.getenv("GENOMEVAULT_LOG_LEVEL", "INFO").upper()


def configure_logging() -> logging.Logger:
    """Configure logging.

    Returns:
        Operation result.
    """
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
