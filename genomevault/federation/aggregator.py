"""Aggregator module."""
from __future__ import annotations

from typing import Dict, List

import torch
class FedAvgAggregator:
    """Simple FedAvg over state_dicts (float tensors only)."""

    def aggregate(
        """Aggregate.

            Args:
                client_models: Model instance.

            Returns:
                Operation result.
            """
        self, client_models: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        if not client_models:
            return {}
        keys = client_models[0].keys()
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            tensors = [m[k].float() for m in client_models]
            out[k] = sum(tensors) / len(tensors)
        return out
