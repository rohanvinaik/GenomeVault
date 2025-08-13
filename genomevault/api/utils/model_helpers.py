"""Helper functions for API models."""
from typing import Any, Dict
def dict_for_update(obj: Any) -> Dict[str, Any]:
    """Convert model to dict, excluding None values for updates.

    This function is used across multiple API models to prepare
    data for database updates by removing None values.

    Args:
        obj: Model instance to convert

    Returns:
        Dict with non-None values only
    """
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if v is not None}
    elif hasattr(obj, "dict"):
        # For Pydantic models
        return {k: v for k, v in obj.dict().items() if v is not None}
    else:
        raise TypeError(f"Cannot convert {type(obj)} to dict")


def merge_with_existing(existing: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge update dict with existing data.

    Args:
        existing: Current data
        updates: Updates to apply

    Returns:
        Merged dictionary
    """
    result = existing.copy()
    for key, value in updates.items():
        if value is not None:
            result[key] = value
    return result
