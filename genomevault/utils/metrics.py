from collections import Counter


class MetricsCollector(Counter):
    """Simple metrics collection based on Counter."""

    pass


def get_metrics():
    """Get global metrics collector instance."""
    return _G


_G = MetricsCollector()
