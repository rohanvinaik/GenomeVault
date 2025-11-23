"""Coordinator module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class NodeRef:
    """Data container for noderef information."""

    name: str
    healthy: bool = True


class PIRCoordinator:
    """Routes read-only queries to healthy nodes; trivial round-robin."""

    def __init__(self, nodes: List[NodeRef]):
        """Initialize instance.

        Args:
            nodes: Nodes.

        Raises:
            ValueError: When operation fails.
        """
        if not nodes:
            raise ValueError("no nodes")
        self.nodes = nodes
        self._idx = 0

    def next_node(self) -> NodeRef:
        """Next node.

        Returns:
            NodeRef instance.

        Raises:
            RuntimeError: When operation fails.
        """
        for _ in range(len(self.nodes)):
            node = self.nodes[self._idx]
            self._idx = (self._idx + 1) % len(self.nodes)
            if node.healthy:
                return node
        raise RuntimeError("no healthy nodes")

    def mark_unhealthy(self, name: str) -> None:
        """Mark unhealthy.

        Args:
            name: Name.
        """
        for n in self.nodes:
            if n.name == name:
                n.healthy = False
                return
