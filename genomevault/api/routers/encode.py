from __future__ import annotations

"""Encode module."""
"""Encode module."""
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

from genomevault.hdc.core import encode
from genomevault.local_processing.common import process


class EncodeIn(BaseModel):
    """EncodeIn implementation."""
    data: list[list[float]] = Field(..., description="Tabular numeric data")
    seed: int = 0


class EncodeOut(BaseModel):
    """EncodeOut implementation."""
    dim: int
    n: int


router = APIRouter(prefix="/encode", tags=["encode"])


@router.post("")
def do_encode(payload: EncodeIn) -> EncodeOut:
    """Encode data into hypervectors."""
    df = pd.DataFrame(payload.data)
    X = process(df, {})
    V = encode(X, seed=payload.seed)
    return EncodeOut(dim=int(V.shape[1]), n=int(V.shape[0]))
