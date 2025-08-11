from __future__ import annotations

"""Model Snapshot module."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ModelSnapshot:
    """Data container for modelsnapshot information."""

    version: str
    meta: Dict[str, Any]
    weights_sha256: str

    def save(self, path: str | Path) -> None:
        """Save.

        Args:
            path: File or directory path.
        """
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path: str | Path) -> "ModelSnapshot":
        """Load.

        Args:
            path: File or directory path.

        Returns:
            Operation result.
        """
        obj = json.loads(Path(path).read_text())
        return ModelSnapshot(**obj)
