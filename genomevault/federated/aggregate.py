from __future__ import annotations


"""Aggregate module."""


def aggregate(client_stats: list[dict]) -> dict:
    """Aggregate statistics from multiple clients.

    Args:
        client_stats: List of client statistics dictionaries

    Returns:
        Aggregated statistics with count and means
    """
    if not client_stats:
        return {"count": 0, "means": {}}

    keys = set().union(*(d.keys() for d in client_stats))
    numeric = {k for k in keys if all(isinstance(d.get(k, 0), (int, float)) for d in client_stats)}
    means = {k: sum(d.get(k, 0.0) for d in client_stats) / len(client_stats) for k in numeric}
    return {"count": len(client_stats), "means": means}
