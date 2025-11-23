"""Helpers for working with Python's logging package.

This module provides a single convenience function, :func:`get_logger`, which
returns a logger configured for use across GenomeVault.  Typical usage:

    >>> from genomevault.logging_utils import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("example message")

If *name* is omitted, a logger for this module is returned.
"""

import logging


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger using :func:`logging.getLogger`.

    Args:
        name: Optional name for the logger. Defaults to ``__name__``.

    Returns:
        A :class:`logging.Logger` instance.
    """

    return logging.getLogger(name or __name__)

