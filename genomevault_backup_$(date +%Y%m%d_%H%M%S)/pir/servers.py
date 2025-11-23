"""Servers module with shard health monitoring and auto-ejection."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    """xor bytes.
    Args:        a: Parameter value.        b: Parameter value.
    Returns:
        bytes"""
    return bytes(x ^ y for x, y in zip(a, b))


class PIRServer:
    """Information-theoretic PIR server holding a replicated DB of equal-length byte records."""

    def __init__(self, db: list[bytes]):
        """Initialize instance.

        Args:
            db: Db.

        Raises:
            ValueError: When operation fails.
        """
        if not db:
            raise ValueError("db must be non-empty")
        L = len(db[0])
        if any(len(x) != L for x in db):
            raise ValueError("all records must be the same length")
        self.db = list(db)
        self.record_len = L

    def answer(self, mask: np.ndarray) -> bytes:
        """Return XOR of records where mask[k] == 1."""
        if mask.ndim != 1 or mask.dtype != np.uint8:
            raise ValueError("mask must be 1-D uint8 array")
        res = bytes([0] * self.record_len)
        for k, bit in enumerate(mask):
            if bit & 1:
                res = _xor_bytes(res, self.db[k])
        return res


@dataclass
class ShardHealth:
    """Track health metrics for PIR shards."""

    shard_id: str
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    last_check: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    is_healthy: bool = True

    def record_response(self, response_time: float) -> None:
        """Record successful response."""
        self.response_times.append(response_time)
        self.consecutive_failures = 0
        self.last_check = time.time()

    def record_error(self) -> None:
        """Record failed response."""
        self.error_count += 1
        self.consecutive_failures += 1
        self.last_check = time.time()

        # Auto-eject after 3 consecutive failures
        if self.consecutive_failures >= 3:
            self.is_healthy = False
            logger.warning(
                f"Shard {self.shard_id} ejected after {self.consecutive_failures} consecutive failures"
            )

    def get_p95_latency(self) -> Optional[float]:
        """Get 95th percentile latency."""
        if not self.response_times:
            return None
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx] if idx < len(sorted_times) else sorted_times[-1]

    def health_score(self) -> float:
        """Calculate health score [0, 1]."""
        if not self.is_healthy:
            return 0.0

        # Factor in error rate
        total_requests = len(self.response_times) + self.error_count
        if total_requests == 0:
            return 1.0

        success_rate = len(self.response_times) / total_requests

        # Factor in latency
        p95 = self.get_p95_latency()
        latency_score = 1.0 if p95 is None else max(0, 1 - p95 / 5.0)  # 5s threshold

        # Combine success rate and latency
        return success_rate * 0.7 + latency_score * 0.3


class ShardManager:
    """Manage PIR shard health and auto-ejection."""

    def __init__(self, min_shards: int = 2):
        """Initialize shard manager.

        Args:
            min_shards: Minimum number of healthy shards to maintain
        """
        self.shards: Dict[str, ShardHealth] = {}
        self.min_shards = min_shards
        self._lock = None  # For thread safety if needed

    def add_shard(self, shard_id: str) -> None:
        """Add a new shard to monitor."""
        self.shards[shard_id] = ShardHealth(shard_id=shard_id)
        logger.info(f"Added shard {shard_id} to monitoring")

    def remove_shard(self, shard_id: str) -> None:
        """Remove a shard from monitoring."""
        if shard_id in self.shards:
            del self.shards[shard_id]
            logger.info(f"Removed shard {shard_id} from monitoring")

    def get_healthy_shards(self) -> List[str]:
        """Get list of healthy shard IDs."""
        return [sid for sid, health in self.shards.items() if health.is_healthy]

    def select_shards(self, n: int) -> List[str]:
        """Select n best shards based on health scores.

        Args:
            n: Number of shards to select

        Returns:
            List of selected shard IDs
        """
        healthy = [(sid, self.shards[sid].health_score()) for sid in self.get_healthy_shards()]
        healthy.sort(key=lambda x: x[1], reverse=True)
        selected = [sid for sid, _ in healthy[:n]]

        if len(selected) < n:
            logger.warning(f"Only {len(selected)} healthy shards available, requested {n}")

        return selected

    def record_shard_response(self, shard_id: str, response_time: float) -> None:
        """Record successful shard response.

        Args:
            shard_id: ID of the shard
            response_time: Response time in seconds
        """
        if shard_id in self.shards:
            self.shards[shard_id].record_response(response_time)

    def record_shard_error(self, shard_id: str) -> None:
        """Record shard error and potentially eject.

        Args:
            shard_id: ID of the shard that failed
        """
        if shard_id in self.shards:
            self.shards[shard_id].record_error()

            # Check if we still have minimum shards
            healthy_count = len(self.get_healthy_shards())
            if healthy_count < self.min_shards:
                logger.warning(f"Only {healthy_count} healthy shards, attempting recovery")
                self._attempt_recovery()

    def _attempt_recovery(self) -> None:
        """Attempt to recover an ejected shard."""
        unhealthy = [(sid, health) for sid, health in self.shards.items() if not health.is_healthy]

        if unhealthy:
            # Reset the least problematic shard
            unhealthy.sort(key=lambda x: x[1].error_count)
            recovering_id, recovering_health = unhealthy[0]
            recovering_health.is_healthy = True
            recovering_health.consecutive_failures = 0
            logger.info(f"Recovering shard {recovering_id} to maintain minimum shards")

    def get_shard_status(self) -> Dict[str, Dict]:
        """Get status of all shards.

        Returns:
            Dictionary with shard status information
        """
        status = {}
        for sid, health in self.shards.items():
            status[sid] = {
                "is_healthy": health.is_healthy,
                "health_score": health.health_score(),
                "error_count": health.error_count,
                "consecutive_failures": health.consecutive_failures,
                "p95_latency": health.get_p95_latency(),
                "last_check": health.last_check,
            }
        return status


# Forward Error Correction support
try:
    import pyeclib

    class FECEncoder:
        """Forward Error Correction using erasure codes."""

        def __init__(self, k: int = 10, m: int = 4):
            """Initialize with k data fragments and m parity fragments.

            Args:
                k: Number of data fragments
                m: Number of parity fragments (can tolerate m failures)
            """
            try:
                self.ec = pyeclib.ECDriver(k=k, m=m, ec_type="liberasurecode_rs_vand")
                self.k = k
                self.m = m
                self.available = True
                logger.info(f"FEC initialized with k={k}, m={m} (can tolerate {m} failures)")
            except Exception as e:
                logger.warning(f"Failed to initialize FEC: {e}")
                self.available = False

        def encode(self, data: bytes) -> List[bytes]:
            """Encode data into fragments with redundancy.

            Args:
                data: Data to encode

            Returns:
                List of encoded fragments
            """
            if not self.available:
                return [data]  # Fallback to no encoding
            return self.ec.encode(data)

        def decode(self, fragments: List[bytes], fragment_ids: Optional[List[int]] = None) -> bytes:
            """Decode from available fragments (can tolerate m losses).

            Args:
                fragments: Available fragments
                fragment_ids: IDs of the fragments (if not sequential)

            Returns:
                Decoded data
            """
            if not self.available:
                return fragments[0] if fragments else b""

            # Reconstruct with available fragments
            return self.ec.decode(fragments)

except ImportError:
    logger.info("pyeclib not available, FEC disabled. Install with: pip install pyeclib")

    class FECEncoder:
        """Dummy FEC encoder when pyeclib is not available."""

        def __init__(self, k: int = 10, m: int = 4):
            self.available = False
            self.k = k
            self.m = m

        def encode(self, data: bytes) -> List[bytes]:
            return [data]

        def decode(self, fragments: List[bytes], fragment_ids: Optional[List[int]] = None) -> bytes:
            return fragments[0] if fragments else b""


class ShardedPIRServer:
    """PIR server with sharding and health monitoring."""

    def __init__(self, db: List[bytes], num_shards: int = 3, use_fec: bool = True):
        """Initialize sharded PIR server.

        Args:
            db: Database of equal-length records
            num_shards: Number of shards to create
            use_fec: Whether to use forward error correction
        """
        self.base_server = PIRServer(db)
        self.shard_manager = ShardManager(min_shards=2)
        self.num_shards = num_shards

        # Initialize FEC if requested
        if use_fec:
            self.fec = FECEncoder(k=num_shards, m=2)  # Can tolerate 2 shard failures
        else:
            self.fec = None

        # Create shards
        self._create_shards()

    def _create_shards(self) -> None:
        """Create database shards."""
        self.shards = {}

        for i in range(self.num_shards):
            shard_id = f"shard_{i}"
            self.shards[shard_id] = self.base_server  # In practice, would be separate servers
            self.shard_manager.add_shard(shard_id)

        logger.info(f"Created {self.num_shards} shards")

    def answer_with_sharding(
        self, mask: np.ndarray, requested_shards: Optional[int] = None
    ) -> bytes:
        """Answer PIR query using healthy shards.

        Args:
            mask: Query mask
            requested_shards: Number of shards to query (default: all healthy)

        Returns:
            XOR of selected records
        """
        start_time = time.time()

        # Select shards to query
        if requested_shards is None:
            requested_shards = len(self.shard_manager.get_healthy_shards())

        selected_shards = self.shard_manager.select_shards(requested_shards)

        if not selected_shards:
            raise RuntimeError("No healthy shards available")

        # Query selected shards
        responses = []
        for shard_id in selected_shards:
            try:
                shard_start = time.time()
                response = self.shards[shard_id].answer(mask)
                response_time = time.time() - shard_start

                self.shard_manager.record_shard_response(shard_id, response_time)
                responses.append(response)

            except Exception as e:
                logger.error(f"Shard {shard_id} failed: {e}")
                self.shard_manager.record_shard_error(shard_id)

        # Combine responses
        if not responses:
            raise RuntimeError("All shards failed")

        # XOR all responses together
        result = responses[0]
        for response in responses[1:]:
            result = _xor_bytes(result, response)

        total_time = time.time() - start_time
        logger.debug(f"PIR query answered in {total_time:.3f}s using {len(responses)} shards")

        return result

    def get_health_report(self) -> Dict:
        """Get health report for all shards.

        Returns:
            Health status dictionary
        """
        return {
            "shard_status": self.shard_manager.get_shard_status(),
            "healthy_shards": len(self.shard_manager.get_healthy_shards()),
            "total_shards": len(self.shards),
            "fec_enabled": self.fec is not None and self.fec.available,
        }
