from __future__ import annotations

"""Federated module."""
"""Federated module."""
from fastapi import APIRouter, HTTPException

from genomevault.core.exceptions import GenomeVaultError
from genomevault.federated.aggregator import FedAvgAggregator
from genomevault.federated.models import AggregateRequest, AggregateResponse

router = APIRouter(prefix="/federated", tags=["federated"])
_agg = FedAvgAggregator()


@router.post("/aggregate", response_model=AggregateResponse)
def aggregate(request: AggregateRequest):
    """Aggregate.

        Args:
            request: Client request.

        Returns:
            Operation result.

        Raises:
            HTTPException: When operation fails.
            RuntimeError: When operation fails.
        """
    try:
        return _agg.aggregate(request)
    except GenomeVaultError as e:
        logger.exception("Unhandled exception")
        raise HTTPException(status_code=400, detail=str(e))
        raise RuntimeError("Unspecified error")
