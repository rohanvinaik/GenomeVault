from __future__ import annotations

from fastapi import APIRouter, HTTPException
from genomevault.federated.aggregator import FedAvgAggregator
from genomevault.federated.models import AggregateRequest, AggregateResponse
from genomevault.core.exceptions import GenomeVaultError

router = APIRouter(prefix="/federated", tags=["federated"])
_agg = FedAvgAggregator()

@router.post("/aggregate", response_model=AggregateResponse)
def aggregate(request: AggregateRequest):
    try:
        return _agg.aggregate(request)
    except GenomeVaultError as e:
        raise HTTPException(status_code=400, detail=str(e))
