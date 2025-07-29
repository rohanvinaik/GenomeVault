from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/proofs", tags=["proofs"])

@router.post("/create")
def create_proof():
    return JSONResponse({"detail": "Not implemented"}, status_code=501)

@router.post("/verify")
def verify_proof():
    return JSONResponse({"detail": "Not implemented"}, status_code=501)
