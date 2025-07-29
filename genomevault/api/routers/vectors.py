from __future__ import annotations

from fastapi import APIRouter, HTTPException
from genomevault.api.models.vectors import VectorEncodeRequest, VectorEncodeResponse
from genomevault.hypervector.engine import HypervectorEngine
from genomevault.core.exceptions import GenomeVaultError

router = APIRouter(prefix="/vectors", tags=["vectors"])

_engine = HypervectorEngine()

@router.post("/encode", response_model=VectorEncodeResponse)
def encode_vector(request: VectorEncodeRequest):
    try:
        res = _engine.encode(data=request.data, dimension=int(request.dimension.value), compression_tier=request.compression_tier)
        return VectorEncodeResponse(**res)
    except GenomeVaultError as e:
        raise HTTPException(status_code=400, detail=str(e))
