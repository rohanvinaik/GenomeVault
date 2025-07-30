from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException

from genomevault.api.models.pir import PIRQueryRequest, PIRQueryResponse
from genomevault.pir.engine import PIREngine

router = APIRouter(prefix="/pir", tags=["pir"])

# Static demo dataset (hashed to 32 bytes)
_DATASET = [b"alpha", b"bravo", b"charlie", b"delta"]
_ENGINE = PIREngine(_DATASET, n_servers=3)


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


@router.post("/query", response_model=PIRQueryResponse)
def pir_query(request: PIRQueryRequest):
    try:
        out = _ENGINE.query(request.index)
        return PIRQueryResponse(index=request.index, item_base64=_b64(out))
    except Exception as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        raise HTTPException(status_code=400, detail=str(e))
        raise
