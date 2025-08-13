"""Vectors module."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from typing import Any

from genomevault.api.models.vectors import (

    VectorEncodeRequest,
    VectorEncodeResponse,
    VectorOperationRequest,
)
from genomevault.core.exceptions import GenomeVaultError
from genomevault.hypervector.engine import HypervectorEngine
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/vectors", tags=["vectors"])

_engine = HypervectorEngine()


@router.post("/encode", response_model=VectorEncodeResponse)
def encode_vector(request: VectorEncodeRequest) -> Any:
    """Encode vector.

    Args:
        request: Client request.

    Returns:
        Operation result.

    Raises:
        HTTPException: When operation fails.
        RuntimeError: When operation fails.
    """
    try:
        res = _engine.encode(
            data=request.data,
            dimension=int(request.dimension.value),
            compression_tier=request.compression_tier,
        )
        return VectorEncodeResponse(**res)
    except GenomeVaultError as e:
        logger.exception("Unhandled exception")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/operations")
def perform_operation(request: VectorOperationRequest) -> Any:
    """Perform operation.

    Args:
        request: Client request.

    Returns:
        Operation result.

    Raises:
        HTTPException: When operation fails.
        RuntimeError: When operation fails.
    """
    try:
        res = _engine.operate(
            operation=request.operation,
            vector_ids=request.vector_ids,
            parameters=request.parameters or {},
        )
        return res
    except GenomeVaultError as e:
        logger.exception("Unhandled exception")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/similarity")
def calculate_similarity(vector_id1: str, vector_id2: str) -> Any:
    """Calculate similarity.

    Args:
        vector_id1: Vector id1.
        vector_id2: Vector id2.

    Returns:
        Calculated result.

    Raises:
        HTTPException: When operation fails.
        RuntimeError: When operation fails.
    """
    try:
        sim = _engine.calculate_similarity(vector_id1, vector_id2)
        return {"similarity": sim, "vector_ids": [vector_id1, vector_id2]}
    except GenomeVaultError as e:
        logger.exception("Unhandled exception")
        raise HTTPException(status_code=400, detail=str(e))
