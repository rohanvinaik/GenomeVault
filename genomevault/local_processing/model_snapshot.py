from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any
import json
from pathlib import Path

@dataclass
class ModelSnapshot:
    version: str
    meta: Dict[str, Any]
    weights_sha256: str

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path: str | Path) -> "ModelSnapshot":
        obj = json.loads(Path(path).read_text())
        return ModelSnapshot(**obj)