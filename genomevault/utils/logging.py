import logging
import sys

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s"


def get_logger(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    if not logging.getLogger().handlers:  # configure root once
        logging.basicConfig(stream=sys.stderr, level=level, format=_FMT)
    return logging.getLogger(name)


def log_operation(func):
    """Placeholder logging decorator - replace with real implementation"""

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
