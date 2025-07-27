import logging


def get_logger(name: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name or __name__)
    return logger
