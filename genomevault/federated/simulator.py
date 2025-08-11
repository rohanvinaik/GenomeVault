from __future__ import annotations

"""Simulator module."""
from dataclasses import dataclass

import numpy as np

from genomevault.federated.aggregator import FedAvgAggregator
from genomevault.federated.models import ModelUpdate


@dataclass
class ClientSim:
    """Data container for clientsim information."""

    dim: int = 16
    seed: int = 0

    def update(self) -> ModelUpdate:
        """Update.

        Returns:
            ModelUpdate instance.
        """
        rng = np.random.default_rng(self.seed)
        grad = rng.normal(0, 0.1, size=self.dim).astype("float32").tolist()
        return ModelUpdate(gradient=grad, weight=1.0)


def simulate_round(n_clients=5, dim=16):
    """Simulate round.

    Args:
        n_clients: N clients.
        dim: Dimension value.

    Returns:
        Operation result.
    """
    agg = FedAvgAggregator()
    updates = [ClientSim(dim, i).update() for i in range(n_clients)]
    return agg.aggregate(updates)
