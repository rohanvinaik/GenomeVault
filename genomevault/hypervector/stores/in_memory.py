from __future__ import annotations

from typing import Dict, Optional
import numpy as np


class InMemoryStore:
    """Extremely simple in-memory vector store keyed by id."""

    def __init__(self) -> None:
        self._data: Dict[str, np.ndarray] = {}

    def put(self, vid: str, vector: np.ndarray) -> None:
        self._data[vid] = vector

    def get(self, vid: str) -> Optional[np.ndarray]:
        return self._data.get(vid)
