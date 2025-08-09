import logging


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration.

    Args:
        level: Logging level

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    return logging.getLogger("genomevault")
