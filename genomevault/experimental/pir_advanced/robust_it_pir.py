from __future__ import annotations

"""Robust It Pir module."""
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class RobustServer:
    """Server handling robust requests."""

    data: Sequence[int]
    alive: bool = True

    def get(self, i: int) -> int:
        """Get.

        Args:
            i: I.

        Returns:
            Integer result.

        Raises:
            RuntimeError: When operation fails.
        """
        if not self.alive:
            raise RuntimeError("server down")
        return int(self.data[i])


class RobustITPIR:
    """
    Majority-vote robustness layer on top of replicated storage across k servers.
    Not private—this is a correctness/robustness shim for tests.
    """

    def __init__(self, servers: List[RobustServer]):
        """Initialize instance.

        Args:
            servers: Servers.

        Raises:
            ValueError: When operation fails.
        """
        if not servers:
            raise ValueError("no servers")
        n = len(servers[0].data)
        if any(len(s.data) != n for s in servers):
            raise ValueError("inconsistent lengths")
        self.servers = servers
        self.n = n

    def get(self, index: int) -> int:
        """Get.

        Args:
            index: Index position.

        Returns:
            Integer result.

        Raises:
            RuntimeError: When operation fails.
        """
        vals = []
        for s in self.servers:
            try:
                vals.append(s.get(index))
            except Exception:
                continue
        if not vals:
            raise RuntimeError("all servers failed")
        # majority (or fallback to first)
        return int(max(set(vals), key=vals.count))
