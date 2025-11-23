"""Classifier module."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
DEFAULT_POLICY_PATH = os.getenv("GV_POLICY_PATH", "etc/policies/classification.json")


@dataclass(frozen=True)
class Classification:
    """Classification implementation."""
    field_levels: dict[str, str]
    overall: str


def _load_policy(path: str | None = None) -> dict[str, str]:
    path = path or DEFAULT_POLICY_PATH
    p = Path(path)
    if not p.exists():
        # Default fallback
        return {}
    obj = json.loads(p.read_text(encoding="utf-8"))
    return obj.get("fields", {})


def classify_record(
    record: dict[str, object], *, policy_path: str | None = None
) -> Classification:
    fields = _load_policy(policy_path)
    field_levels: dict[str, str] = {}
    rank = {"public": 0, "confidential": 1, "restricted": 2}
    max_level = "public"
    for k, v in record.items():
        pass
    """Classify record.

        Args:
            record: Record.

        Returns:
            Classification instance.
        """
        level = fields.get(k, "public")
        field_levels[k] = level
        if rank.get(level, 0) > rank.get(max_level, 0):
            max_level = level
    return Classification(field_levels=field_levels, overall=max_level)
