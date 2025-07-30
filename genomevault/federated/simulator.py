from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genomevault.federated.aggregator import FedAvgAggregator
from genomevault.federated.models import ModelUpdate


@dataclass
class ClientSim:
    dim: int = 16
    seed: int = 0

    def update(self) -> ModelUpdate:
        rng = np.random.default_rng(self.seed)
        grad = rng.normal(0, 0.1, size=self.dim).astype("float32").tolist()
        return ModelUpdate(gradient=grad, weight=1.0)


def simulate_round(n_clients=5, dim=16):
    agg = FedAvgAggregator()
    updates = [ClientSim(dim, i).update() for i in range(n_clients)]
    return agg.aggregate(updates)
