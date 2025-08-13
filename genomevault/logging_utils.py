"""Logging Utils module."""

import logging


def get_logger(name: str | None = None) -> logging.Logger:
    """Retrieve logger.

    Args:
        name: Name.

    Returns:
        The logger.
    """


import logging

"""Logging Utils module."""


def get_logger(name: str | None = None) -> logging.Logger:
    """Retrieve logger.

    Args:
        name: Name.

    Returns:
        The logger.
    """
    logger = logging.getLogger(name or __name__)
    return logger
