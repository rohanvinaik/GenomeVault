from __future__ import annotations

from fastapi import APIRouter, HTTPException
from genomevault.ledger.store import InMemoryLedger
from genomevault.ledger.models import LedgerAppendRequest, LedgerEntryModel, LedgerVerifyResponse, LedgerEntriesResponse

router = APIRouter(prefix="/ledger", tags=["ledger"])
_L = InMemoryLedger()

@router.post("/append", response_model=LedgerEntryModel)
def append_entry(req: LedgerAppendRequest):
    try:
        e = _L.append(req.data)
        return LedgerEntryModel(**e.__dict__)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/verify", response_model=LedgerVerifyResponse)
def verify_chain():
    return LedgerVerifyResponse(valid=_L.verify_chain())

@router.get("/entries", response_model=LedgerEntriesResponse)
def list_entries():
    return LedgerEntriesResponse(entries=[LedgerEntryModel(**e.__dict__) for e in _L.entries()])
