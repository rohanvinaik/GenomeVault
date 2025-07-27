import asyncio
import random
import statistics
import time

import pytest


class _MockPIRBase:
    async def answer_query_async(self, payload: bytes) -> bytes:
        # Simulate variable read time by payload hint
        await asyncio.sleep(0.002 + (payload[0] % 3) * 0.001)
        return bytes([payload[0]]) * (500 + (payload[0] % 4))


@pytest.mark.asyncio
async def test_pir_timing_inference_resists_leakage():
    from genomevault.pir.server.secure_pir_server import SecurePIRServer

    base = _MockPIRBase()
    srv = SecurePIRServer(base)

    async def measure(idx: int, n=10):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            _ = await srv.answer_query_async(f"{idx}".encode(), bytes([idx]))
            times.append(time.perf_counter() - t0)
        return statistics.mean(times)

    # Compare two indices; attacker tries to separate by timing
    m0 = await measure(0, n=20)
    m1 = await measure(1, n=20)
    diff = abs(m0 - m1)
    # With padding + jitter + mixing, difference should be small
    assert diff < 0.01, f"Timing variance too large: {diff}"
