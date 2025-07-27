from __future__ import annotations

"""
SecurePIRServer: side-channel mitigations (padding, jitter, query mixing) for PIR responses.

Wraps a base PIR server's async answer() with:
- Fixed-size response padding to FIXED_RESPONSE_SIZE_BYTES (configurable)
- Random jitter (0..JITTER_MS) to decorrelate index → timing
- Optional batch mixing: process queries slightly out-of-order within MIX_WINDOW
"""

import asyncio
import os
import random
import time
from dataclasses import dataclass
from typing import Any, List, Tuple

FIXED_RESPONSE_SIZE_BYTES = int(os.environ.get("GV_PIR_FIXED_RESP_BYTES", "65536"))
JITTER_MS = int(os.environ.get("GV_PIR_JITTER_MS", "8"))
MIX_WINDOW = int(os.environ.get("GV_PIR_MIX_WINDOW", "16"))  # max queue size before flush


@dataclass
class QueryEnvelope:
    query_id: str
    payload: bytes
    submitted_ts: float


class SecurePIRServer:
    def __init__(self, base_server: Any) -> None:
        self.base = base_server
        self._queue: List[QueryEnvelope] = []
        self._lock = asyncio.Lock()

    async def answer_query_async(self, query_id: str, payload: bytes) -> bytes:
        env = QueryEnvelope(query_id=query_id, payload=payload, submitted_ts=time.time())
        async with self._lock:
            self._queue.append(env)
            # Mix queries: randomly pop within window
            if len(self._queue) >= MIX_WINDOW:
                idx = random.randrange(0, len(self._queue))
                env = self._queue.pop(idx)
            else:
                # Randomly decide to process now or defer
                if random.random() < 0.5:
                    idx = random.randrange(0, len(self._queue))
                    env = self._queue.pop(idx)
                else:
                    # process immediately
                    self._queue.remove(env)

        # Delegate to base
        raw = await self.base.answer_query_async(env.payload)

        # Pad to fixed size
        if len(raw) > FIXED_RESPONSE_SIZE_BYTES:
            # Chunk if oversized: deterministic chunking to fixed-size blocks
            blocks = [
                raw[i : i + FIXED_RESPONSE_SIZE_BYTES]
                for i in range(0, len(raw), FIXED_RESPONSE_SIZE_BYTES)
            ]
            raw = blocks[0]
        pad_len = FIXED_RESPONSE_SIZE_BYTES - len(raw)
        if pad_len > 0:
            raw = raw + b"\x00" * pad_len

        # Add jitter
        if JITTER_MS > 0:
            await asyncio.sleep(random.uniform(0, JITTER_MS / 1000.0))

        return raw
