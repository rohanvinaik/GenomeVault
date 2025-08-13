"""In Memory module."""
from __future__ import annotations

import numpy as np
class InMemoryStore:
    """Extremely simple in-memory vector store keyed by id."""

    def __init__(self) -> None:
        """Initialize instance."""
        self._data: dict[str, np.ndarray] = {}

    def put(self, vid: str, vector: np.ndarray) -> None:
        """Put.

        Args:
            vid: Vid.
            vector: Vector.
        """
        self._data[vid] = vector

    def get(self, vid: str) -> np.ndarray | None:
        """Get.

        Args:
            vid: Vid.

        Returns:
            Operation result.
        """
        return self._data.get(vid)
