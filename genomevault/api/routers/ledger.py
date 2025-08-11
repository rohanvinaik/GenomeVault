from __future__ import annotations

"""Ledger module."""
from fastapi import APIRouter, HTTPException

from genomevault.ledger.models import (
    LedgerAppendRequest,
    LedgerEntriesResponse,
    LedgerEntryModel,
    LedgerVerifyResponse,
)
from genomevault.ledger.store import InMemoryLedger

router = APIRouter(prefix="/ledger", tags=["ledger"])
_L = InMemoryLedger()


@router.post("/append", response_model=LedgerEntryModel)
def append_entry(req: LedgerAppendRequest):
    """Append entry.

    Args:
        req: Req.

    Returns:
        Operation result.

    Raises:
        HTTPException: When operation fails.
        RuntimeError: When operation fails.
    """
    try:
        e = _L.append(req.data)
        return LedgerEntryModel(**e.__dict__)
    except Exception as e:
        logger.exception("Unhandled exception")
        raise HTTPException(status_code=400, detail=str(e))
        raise RuntimeError("Unspecified error")


@router.get("/verify", response_model=LedgerVerifyResponse)
def verify_chain():
    """Verify chain.

    Returns:
        Operation result.
    """
    return LedgerVerifyResponse(valid=_L.verify_chain())


@router.get("/entries", response_model=LedgerEntriesResponse)
def list_entries():
    """List entries.

    Returns:
        Operation result.
    """
    return LedgerEntriesResponse(entries=[LedgerEntryModel(**e.__dict__) for e in _L.entries()])
