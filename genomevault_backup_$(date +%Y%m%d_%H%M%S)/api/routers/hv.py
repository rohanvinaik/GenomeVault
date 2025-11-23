from __future__ import annotations
from typing import List, Dict, Union
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from genomevault.hypervector.engine import HypervectorEngine, HypervectorConfig
from genomevault.hypervector.featurizers.variants import featurize_variants

router = APIRouter(prefix="/hv", tags=["hypervector"])


class EncodeRequest(BaseModel):
    # Either provide numeric features directly...
    numeric: Union[List[float], None] = Field(default=None)
    # ...or a list of variant dicts (chrom, pos, ref, alt, impact)
    variants: Union[List[Dict], None] = Field(default=None)
    dim: int = 8192
    binary: bool = False


class EncodeResponse(BaseModel):
    dim: int
    binary: bool
    vector: List[int] | List[float]


@router.post("/encode", response_model=EncodeResponse)
def encode(req: EncodeRequest) -> EncodeResponse:
    if req.numeric is None and not req.variants:
        raise HTTPException(status_code=400, detail="Provide either 'numeric' or 'variants'.")

    cfg = HypervectorConfig(dim=req.dim, binary=req.binary)
    engine = HypervectorEngine(cfg)

    if req.numeric is not None:
        hv = engine.encode_numeric(req.numeric)
    else:
        feats = featurize_variants(req.variants or [])
        if feats.size == 0:
            raise HTTPException(status_code=400, detail="Empty variants payload.")
        hv = engine.encode_numeric(feats)

    if req.binary:
        vec = hv.astype(int).tolist()
    else:
        vec = hv.astype(float).tolist()

    return EncodeResponse(dim=len(vec), binary=req.binary, vector=vec)
