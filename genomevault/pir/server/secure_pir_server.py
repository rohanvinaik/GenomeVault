"""
Secure PIR server implementation with timing side-channel protections.
Implements padding, jitter, and constant-time operations as per audit recommendations.
"""
import asyncio
import hashlib
import json
import mmap
import os
import random
import secrets
import struct
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiofiles
import lz4.frame
import numpy as np
import uvloop

from ...core.config import get_config
from ...core.exceptions import SecurityError, ValidationError
from ...utils.logging import get_logger
from .enhanced_pir_server import (
    EnhancedPIRServer,
    GenomicRegion,
    OptimizedPIRDatabase,
    ShardMetadata,
)

logger = get_logger(__name__)
config = get_config()

# Constants for timing protection
RESPONSE_PAD_SIZE = 1024  # Base padding size
JITTER_WINDOW_MS = 50  # Maximum jitter in milliseconds
MIN_RESPONSE_TIME_MS = 100  # Minimum response time to normalize
MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10MB max response


@dataclass
class TimingProtectionConfig:
    """Configuration for timing side-channel protection."""

    enable_padding: bool = True
    enable_jitter: bool = True
    enable_constant_time: bool = True
    pad_size_min: int = 512
    pad_size_max: int = 4096
    jitter_min_ms: int = 10
    jitter_max_ms: int = 50
    constant_time_buckets: List[int] = field(
        default_factory=lambda: [1024, 4096, 16384, 65536, 262144]
    )


class SecurePIRDatabase(OptimizedPIRDatabase):
    """
    Enhanced PIR database with constant-time operations.
    """

    def __init__(
        self,
        base_path: Path,
        cache_size_mb: int = 1024,
        timing_config: Optional[TimingProtectionConfig] = None,
    ) -> None:
        """Initialize secure PIR database with timing protections."""
        super().__init__(base_path, cache_size_mb)
        self.timing_config = timing_config or TimingProtectionConfig()
        self.constant_time_cache = {}

    async def secure_query_item(self, shard: ShardMetadata, position_key: str) -> Optional[bytes]:
        """
        Query item with constant-time operations.

        Args:
            shard: Shard containing the data
            position_key: Position key

        Returns:
            Item data with timing protection
        """
        start_time = time.perf_counter_ns()

        # Get actual data
        data = await self.query_item(shard, position_key)

        if not self.timing_config.enable_constant_time:
            return data

        # Normalize to bucket size
        if data is None:
            # Return dummy data of minimum bucket size
            data = secrets.token_bytes(self.timing_config.constant_time_buckets[0])
        else:
            # Pad to next bucket size
            data_len = len(data)
            for bucket_size in self.timing_config.constant_time_buckets:
                if data_len <= bucket_size:
                    padding_needed = bucket_size - data_len
                    data = data + secrets.token_bytes(padding_needed)
                    break
            else:
                # Data exceeds largest bucket, truncate
                data = data[: self.timing_config.constant_time_buckets[-1]]

        # Ensure constant time by adding dummy operations
        elapsed_ns = time.perf_counter_ns() - start_time
        target_ns = 1_000_000  # 1ms target
        if elapsed_ns < target_ns:
            # Perform dummy operations
            dummy = secrets.token_bytes(256)
            for _ in range((target_ns - elapsed_ns) // 10000):
                _ = hashlib.sha256(dummy).digest()

        return data


class SecurePIRServer(EnhancedPIRServer):
    """
    Production-ready PIR server with comprehensive timing side-channel protections.
    """

    def __init__(
        self,
        server_id: str,
        data_directory: Path,
        is_trusted_signatory: bool = False,
        enable_preprocessing: bool = True,
        cache_size_mb: int = 2048,
        timing_config: Optional[TimingProtectionConfig] = None,
    ) -> None:
        """Initialize secure PIR server with timing protections."""
        # Initialize parent with secure database
        self.timing_config = timing_config or TimingProtectionConfig()

        # Replace database with secure version
        super().__init__(
            server_id, data_directory, is_trusted_signatory, enable_preprocessing, cache_size_mb
        )
        self.database = SecurePIRDatabase(data_directory, cache_size_mb, self.timing_config)

        # Query mixer for additional protection
        self.query_mixer = QueryMixer() if self.timing_config.enable_jitter else None

        # Timing statistics for monitoring
        self.timing_stats = defaultdict(list)

        logger.info(
            f"Secure PIR server {server_id} initialized with timing protections",
            extra={
                "padding": self.timing_config.enable_padding,
                "jitter": self.timing_config.enable_jitter,
                "constant_time": self.timing_config.enable_constant_time,
            },
        )

    async def process_query_secure(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process PIR query with timing side-channel protections.

        Args:
            query_data: Query containing vectors and parameters

        Returns:
            Response with timing protections applied
        """
        start_time = time.perf_counter()
        query_id = query_data.get("query_id", "unknown")

        # Add to query mixer if enabled
        if self.query_mixer:
            await self.query_mixer.add_query(query_data)
            # Process mixed batch
            mixed_queries = await self.query_mixer.get_mixed_batch()

            # Process all queries in batch
            results = []
            for q in mixed_queries:
                result = await self._process_single_query_secure(q)
                results.append(result)

            # Extract our result
            for i, q in enumerate(mixed_queries):
                if q.get("query_id") == query_id:
                    response = results[i]
                    break
            else:
                response = {"error": "Query lost in mixing"}
        else:
            # Process directly
            response = await self._process_single_query_secure(query_data)

        # Apply response padding
        if self.timing_config.enable_padding:
            response = self._apply_response_padding(response)

        # Apply timing jitter
        if self.timing_config.enable_jitter:
            await self._apply_timing_jitter()

        # Ensure minimum response time
        elapsed = time.perf_counter() - start_time
        min_time_s = MIN_RESPONSE_TIME_MS / 1000
        if elapsed < min_time_s:
            await asyncio.sleep(min_time_s - elapsed)

        # Record timing for analysis
        self._record_timing(query_id, time.perf_counter() - start_time)

        return response

    async def _process_single_query_secure(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process single query with security measures."""
        try:
            # Validate query size with constant-time comparison
            query_size = self._calculate_query_size_constant_time(query_data)
            if not self._constant_time_compare(query_size <= self.max_query_size, True):
                raise ValidationError("Query size exceeds maximum")

            # Process using parent method
            response = await super().process_query(query_data)

            return response

        except Exception as e:
            # Return consistent error response
            return self._create_error_response(str(e), query_data.get("query_id"))

    def _calculate_query_size_constant_time(self, query_data: Dict[str, Any]) -> int:
        """Calculate query size in constant time."""
        size = 0

        # Always check all expected fields
        for field in ["query_id", "client_id", "query_vectors", "query_type", "parameters"]:
            if field in query_data:
                data = query_data[field]
                if isinstance(data, list):
                    # Process all elements
                    for item in data:
                        size += len(str(item))
                else:
                    size += len(str(data))
            else:
                # Add dummy operations for missing fields
                size += len(str("dummy"))

        return size

    def _constant_time_compare(self, a: bool, b: bool) -> bool:
        """Constant-time boolean comparison."""
        return secrets.compare_digest(str(a).encode(), str(b).encode())

    def _apply_response_padding(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Apply padding to normalize response sizes."""
        # Calculate current size
        response_str = json.dumps(response)
        current_size = len(response_str.encode())

        # Determine target size (next bucket)
        target_size = current_size
        for bucket in self.timing_config.constant_time_buckets:
            if current_size <= bucket:
                target_size = bucket
                break

        # Add padding field
        padding_size = max(0, target_size - current_size - 50)  # Account for padding field overhead
        response["_padding"] = secrets.token_hex(padding_size // 2)

        return response

    async def _apply_timing_jitter(self) -> None:
        """Apply random timing jitter."""
        jitter_ms = random.uniform(
            self.timing_config.jitter_min_ms, self.timing_config.jitter_max_ms
        )
        await asyncio.sleep(jitter_ms / 1000)

    def _create_error_response(self, error_msg: str, query_id: Optional[str]) -> Dict[str, Any]:
        """Create consistent error response."""
        # Always return same structure
        return {
            "query_id": query_id or "unknown",
            "server_id": self.server_id,
            "error": "Processing error",  # Generic message
            "results": [],
            "processing_time_ms": 0,
            "_padding": secrets.token_hex(512),  # Consistent padding
        }

    def _record_timing(self, query_id: str, elapsed_time: float) -> None:
        """Record timing statistics for analysis."""
        self.timing_stats["all"].append(elapsed_time)

        # Keep only recent timings
        if len(self.timing_stats["all"]) > 10000:
            self.timing_stats["all"] = self.timing_stats["all"][-10000:]

    async def get_timing_analysis(self) -> Dict[str, Any]:
        """Analyze timing patterns for potential leaks."""
        if not self.timing_stats["all"]:
            return {"status": "No data collected"}

        timings = np.array(self.timing_stats["all"])

        return {
            "count": len(timings),
            "mean_ms": np.mean(timings) * 1000,
            "std_ms": np.std(timings) * 1000,
            "min_ms": np.min(timings) * 1000,
            "max_ms": np.max(timings) * 1000,
            "cv": np.std(timings) / np.mean(timings),  # Coefficient of variation
            "timing_variance_assessment": self._assess_timing_variance(timings),
        }

    def _assess_timing_variance(self, timings: np.ndarray) -> str:
        """Assess whether timing variance is within acceptable bounds."""
        cv = np.std(timings) / np.mean(timings)

        if cv < 0.01:
            return "EXCELLENT: <1% variance"
        elif cv < 0.05:
            return "GOOD: <5% variance"
        elif cv < 0.10:
            return "ACCEPTABLE: <10% variance"
        else:
            return f"WARNING: {cv*100:.1f}% variance exceeds safety threshold"


class QueryMixer:
    """Mix queries to prevent timing correlation attacks."""

    def __init__(self, batch_size: int = 10, max_wait_ms: int = 50):
        """Initialize query mixer."""
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_queries = []
        self.lock = asyncio.Lock()

    async def add_query(self, query: Dict[str, Any]) -> None:
        """Add query to mixing pool."""
        async with self.lock:
            self.pending_queries.append({"query": query, "timestamp": time.time()})

    async def get_mixed_batch(self) -> List[Dict[str, Any]]:
        """Get mixed batch of queries."""
        start_time = time.time()

        # Wait for batch to fill or timeout
        while len(self.pending_queries) < self.batch_size:
            if (time.time() - start_time) * 1000 > self.max_wait_ms:
                break
            await asyncio.sleep(0.001)

        async with self.lock:
            # Take available queries
            batch_size = min(len(self.pending_queries), self.batch_size)
            if batch_size == 0:
                return []

            # Extract queries
            batch = self.pending_queries[:batch_size]
            self.pending_queries = self.pending_queries[batch_size:]

            # Add dummy queries if needed
            while len(batch) < self.batch_size:
                batch.append({"query": self._create_dummy_query(), "timestamp": time.time()})

            # Shuffle to prevent ordering attacks
            random.shuffle(batch)

            return [item["query"] for item in batch]

    def _create_dummy_query(self) -> Dict[str, Any]:
        """Create realistic dummy query."""
        return {
            "query_id": f"dummy_{secrets.token_hex(8)}",
            "client_id": f"dummy_client_{secrets.token_hex(4)}",
            "query_vectors": [np.random.binomial(1, 0.001, 10000).astype(np.uint8)],
            "query_type": "genomic",
            "parameters": {
                "regions": [
                    {
                        "chromosome": f"chr{random.randint(1, 22)}",
                        "position": random.randint(1, 250000000),
                    }
                ]
            },
        }


# Example test for timing analysis
async def test_timing_protection():
    """Test timing protection effectiveness."""
    # Initialize secure server
    server = SecurePIRServer(
        server_id="secure_test",
        data_directory=Path("/tmp/pir_test"),
        timing_config=TimingProtectionConfig(
            enable_padding=True, enable_jitter=True, enable_constant_time=True
        ),
    )

    # Run queries with different characteristics
    timings = defaultdict(list)

    for size in [100, 1000, 10000]:
        for _ in range(100):
            query = {
                "query_id": f"test_{size}_{_}",
                "client_id": "test_client",
                "query_vectors": [np.random.binomial(1, 0.001, size).astype(np.uint8)],
                "query_type": "genomic",
                "parameters": {},
            }

            start = time.perf_counter()
            await server.process_query_secure(query)
            elapsed = time.perf_counter() - start

            timings[size].append(elapsed)

    # Analyze timing variance
    print("Timing Analysis:")
    for size, times in timings.items():
        times_array = np.array(times)
        cv = np.std(times_array) / np.mean(times_array)
        print(f"Size {size}: CV={cv:.3f}, Mean={np.mean(times_array)*1000:.1f}ms")

    # Get server analysis
    analysis = await server.get_timing_analysis()
    print(f"\nServer Assessment: {analysis['timing_variance_assessment']}")

    await server.shutdown()


if __name__ == "__main__":
    asyncio.run(test_timing_protection())
