from __future__ import annotations

from pydantic import BaseModel
from typing import List


class PIRQueryRequest(BaseModel):
    index: int


class PIRQueryResponse(BaseModel):
    index: int
    item_base64: str  # base64 of 32-byte record
