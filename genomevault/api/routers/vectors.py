from __future__ import annotations

from fastapi import APIRouter, HTTPException

from genomevault.api.models.vectors import (VectorEncodeRequest,
                                            VectorEncodeResponse,
                                            VectorOperationRequest)
from genomevault.core.exceptions import GenomeVaultError
from genomevault.hypervector.engine import HypervectorEngine

router = APIRouter(prefix="/vectors", tags=["vectors"])

_engine = HypervectorEngine()


@router.post("/encode", response_model=VectorEncodeResponse)
def encode_vector(request: VectorEncodeRequest):
    try:
        res = _engine.encode(
            data=request.data,
            dimension=int(request.dimension.value),
            compression_tier=request.compression_tier,
        )
        return VectorEncodeResponse(**res)
    except GenomeVaultError as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        raise HTTPException(status_code=400, detail=str(e))
        raise RuntimeError("Unspecified error")


@router.post("/operations")
def perform_operation(request: VectorOperationRequest):
    try:
        res = _engine.operate(
            operation=request.operation,
            vector_ids=request.vector_ids,
            parameters=request.parameters or {},
        )
        return res
    except GenomeVaultError as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        raise HTTPException(status_code=400, detail=str(e))
        raise RuntimeError("Unspecified error")


@router.get("/similarity")
def calculate_similarity(vector_id1: str, vector_id2: str):
    try:
        sim = _engine.calculate_similarity(vector_id1, vector_id2)
        return {"similarity": sim, "vector_ids": [vector_id1, vector_id2]}
    except GenomeVaultError as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        raise HTTPException(status_code=400, detail=str(e))
        raise RuntimeError("Unspecified error")
