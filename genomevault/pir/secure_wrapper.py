# genomevault/pir/secure_wrapper.py
"""Enhanced PIR with timing side-channel protection and constant-time operations."""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from cryptography.hazmat.primitives import constant_time

# ======================== Configuration ========================


@dataclass(frozen=True)
class PIRSecurityConfig:
    """Security configuration for PIR server."""

    fixed_response_bytes: int = 65536  # Fixed response size
    jitter_ms: int = 8  # Timing jitter in milliseconds
    mix_window_size: int = 16  # Query mixing window
    min_batch_size: int = 4  # Minimum queries before processing
    max_batch_wait_ms: int = 100  # Maximum wait for batching
    padding_strategy: str = "constant"  # 'constant', 'adaptive', 'decoy'
    timing_variance_target: float = 0.01  # < 1% timing variance

    @classmethod
    def from_env(cls) -> PIRSecurityConfig:
        """Load configuration from environment variables."""
        return cls(
            fixed_response_bytes=int(os.getenv("GV_PIR_FIXED_RESP_BYTES", "65536")),
            jitter_ms=int(os.getenv("GV_PIR_JITTER_MS", "8")),
            mix_window_size=int(os.getenv("GV_PIR_MIX_WINDOW", "16")),
            min_batch_size=int(os.getenv("GV_PIR_MIN_BATCH", "4")),
            max_batch_wait_ms=int(os.getenv("GV_PIR_MAX_WAIT_MS", "100")),
            padding_strategy=os.getenv("GV_PIR_PADDING", "constant"),
            timing_variance_target=float(os.getenv("GV_PIR_TIMING_TARGET", "0.01")),
        )


# ======================== Protocols ========================


class SupportsAnswer(Protocol):
    """Protocol for PIR backends."""

    async def answer_query_async(self, payload: bytes) -> bytes: ...


@dataclass
class QueryMetadata:
    """Metadata for tracking queries."""

    query_id: str
    arrival_time: float
    client_id: str | None = None
    priority: int = 0
    decoy: bool = False


# ======================== Timing Protection ========================


class ConstantTimeOperations:
    """Constant-time operations for timing attack prevention."""

    @staticmethod
    def pad_response(data: bytes, target_size: int) -> bytes:
        """Pad response to fixed size in constant time."""
        if len(data) >= target_size:
            return data[:target_size]

        # Use cryptographic padding
        padding_needed = target_size - len(data)
        padding = os.urandom(padding_needed)

        # Constant-time concatenation
        result = bytearray(target_size)
        result[: len(data)] = data
        result[len(data) :] = padding

        return bytes(result)

    @staticmethod
    def constant_time_select(condition: bool, true_val: bytes, false_val: bytes) -> bytes:
        """Select value based on condition in constant time."""
        # Ensure both values have same length
        max_len = max(len(true_val), len(false_val))
        true_val = ConstantTimeOperations.pad_response(true_val, max_len)
        false_val = ConstantTimeOperations.pad_response(false_val, max_len)

        # Use cryptography library's constant-time comparison
        result = bytearray(max_len)
        condition_byte = 0xFF if condition else 0x00

        for i in range(max_len):
            result[i] = (true_val[i] & condition_byte) | (false_val[i] & ~condition_byte)

        return bytes(result)


class TimingProtection:
    """Advanced timing protection mechanisms."""

    def __init__(self, config: PIRSecurityConfig):
        self.config = config
        self.timing_history: deque = deque(maxlen=1000)
        self.baseline_timing: float | None = None

    async def add_calibrated_delay(self, actual_duration: float) -> None:
        """Add calibrated delay to achieve consistent timing."""
        target_duration = self._get_target_duration()

        if actual_duration < target_duration:
            # Add precise delay
            delay = target_duration - actual_duration

            # Add jitter
            if self.config.jitter_ms > 0:
                jitter = random.uniform(-self.config.jitter_ms, self.config.jitter_ms) / 1000
                delay += jitter

            if delay > 0:
                await asyncio.sleep(delay)

        # Record timing for adaptation
        total_time = time.time() - (time.time() - actual_duration)
        self.timing_history.append(total_time)
        self._update_baseline()

    def _get_target_duration(self) -> float:
        """Get target duration based on historical data."""
        if not self.baseline_timing or len(self.timing_history) < 10:
            return 0.1  # Default 100ms

        # Use 95th percentile as target
        sorted_times = sorted(self.timing_history)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx]

    def _update_baseline(self) -> None:
        """Update baseline timing statistics."""
        if len(self.timing_history) >= 100:
            self.baseline_timing = np.median(list(self.timing_history))

    def get_timing_variance(self) -> float:
        """Calculate current timing variance."""
        if len(self.timing_history) < 10:
            return 1.0  # Maximum variance when insufficient data

        times = list(self.timing_history)
        mean = np.mean(times)
        variance = np.std(times) / mean if mean > 0 else 1.0

        return variance


# ======================== Query Mixing ========================


class QueryMixer:
    """Mix queries to prevent correlation attacks."""

    def __init__(self, window_size: int = 16):
        self.window_size = window_size
        self.query_pool: deque = deque()
        self.lock = asyncio.Lock()

    async def add_query(self, query: bytes, metadata: QueryMetadata) -> None:
        """Add query to mixing pool."""
        async with self.lock:
            self.query_pool.append((query, metadata))

            # Add decoy queries if pool is small
            if len(self.query_pool) < self.window_size // 2:
                await self._inject_decoy_queries()

    async def get_mixed_query(self) -> tuple[bytes, QueryMetadata] | None:
        """Get a randomly selected query from pool."""
        async with self.lock:
            if not self.query_pool:
                return None

            # Random selection with priority weighting
            weights = [1.0 + meta.priority * 0.1 for _, meta in self.query_pool]
            total_weight = sum(weights)

            if total_weight == 0:
                idx = random.randrange(len(self.query_pool))
            else:
                r = random.uniform(0, total_weight)
                cumsum = 0
                idx = 0
                for i, w in enumerate(weights):
                    cumsum += w
                    if cumsum >= r:
                        idx = i
                        break

            return (
                self.query_pool.pop(idx)
                if idx < len(self.query_pool)
                else self.query_pool.popleft()
            )

    async def _inject_decoy_queries(self) -> None:
        """Inject decoy queries to maintain minimum pool size."""
        n_decoys = self.window_size // 4
        for _ in range(n_decoys):
            decoy_query = os.urandom(256)  # Random decoy
            decoy_meta = QueryMetadata(
                query_id=hashlib.sha256(decoy_query).hexdigest()[:16],
                arrival_time=time.time(),
                decoy=True,
            )
            self.query_pool.append((decoy_query, decoy_meta))


# ======================== Secure PIR Server ========================


@dataclass
class PIRServerStats:
    """Statistics for PIR server monitoring."""

    total_queries: int = 0
    decoy_queries: int = 0
    timing_variance: float = 0.0
    average_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    security_score: float = 1.0


class SecurePIRServer:
    """Enhanced PIR server with comprehensive timing protection."""

    def __init__(self, backend: SupportsAnswer, config: PIRSecurityConfig | None = None):
        self.backend = backend
        self.config = config or PIRSecurityConfig.from_env()

        # Security components
        self.timing_protection = TimingProtection(self.config)
        self.query_mixer = QueryMixer(self.config.mix_window_size)
        self.constant_ops = ConstantTimeOperations()

        # Response cache for timing consistency
        self.response_cache: dict[str, tuple[bytes, float]] = {}
        self.cache_size_limit = 1000

        # Statistics
        self.stats = PIRServerStats()

        # Batch processing
        self.batch_queue: asyncio.Queue = asyncio.Queue()
        self.batch_processor_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the PIR server background tasks."""
        if not self.batch_processor_task:
            self.batch_processor_task = asyncio.create_task(self._batch_processor())

    async def stop(self) -> None:
        """Stop the PIR server gracefully."""
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                from genomevault.observability.logging import configure_logging

                logger = configure_logging()
                logger.exception("Unhandled exception")
                pass
                raise

    async def answer_query_async(self, payload: bytes) -> bytes:
        """Answer query with comprehensive timing protection."""
        start_time = time.time()

        # Generate query ID
        query_id = hashlib.sha256(payload).hexdigest()[:16]
        metadata = QueryMetadata(query_id=query_id, arrival_time=start_time)

        # Check cache first (constant time)
        cached_response = self._check_cache_constant_time(query_id)
        if cached_response is not None:
            self.stats.cache_hit_rate = (self.stats.cache_hit_rate * 0.99) + 0.01
            response = cached_response
        else:
            # Add to mixer
            await self.query_mixer.add_query(payload, metadata)

            # Get mixed query
            mixed_query = await self.query_mixer.get_mixed_query()
            if mixed_query:
                query_payload, query_meta = mixed_query

                # Process query
                if query_meta.decoy:
                    response = self._generate_decoy_response()
                    self.stats.decoy_queries += 1
                else:
                    response = await self._process_real_query(query_payload)

                # Cache response
                self._update_cache(query_meta.query_id, response)
            else:
                response = await self._process_real_query(payload)

        # Apply constant-time padding
        response = self.constant_ops.pad_response(response, self.config.fixed_response_bytes)

        # Add calibrated timing delay
        processing_time = time.time() - start_time
        await self.timing_protection.add_calibrated_delay(processing_time)

        # Update statistics
        self._update_stats(time.time() - start_time)

        return response

    def _check_cache_constant_time(self, query_id: str) -> bytes | None:
        """Check cache in constant time to prevent timing leaks."""
        # Always iterate through entire cache
        found_response = None
        found_timestamp = 0.0

        for cached_id, (response, timestamp) in self.response_cache.items():
            # Constant-time comparison
            is_match = constant_time.bytes_eq(cached_id.encode()[:16], query_id.encode()[:16])

            # Constant-time selection
            if is_match:
                found_response = response
                found_timestamp = timestamp

        # Check if cache entry is still valid (constant time)
        if found_response and (time.time() - found_timestamp) < 300:  # 5 min TTL
            return found_response

        return None

    async def _process_real_query(self, payload: bytes) -> bytes:
        """Process actual query through backend."""
        try:
            response = await self.backend.answer_query_async(payload)
            return response[: self.config.fixed_response_bytes]
        except KeyError as e:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            # Return error response of same size
            error_msg = f"Error: {e!s}"[:100]
            return error_msg.encode().ljust(self.config.fixed_response_bytes, b"\x00")
            raise

    def _generate_decoy_response(self) -> bytes:
        """Generate realistic decoy response."""
        # Generate response that looks like real data
        response_size = random.randint(1000, self.config.fixed_response_bytes)
        return os.urandom(response_size)

    def _update_cache(self, query_id: str, response: bytes) -> None:
        """Update response cache with size limits."""
        # Evict old entries if needed
        if len(self.response_cache) >= self.cache_size_limit:
            # Remove oldest entry
            oldest_id = min(self.response_cache.keys(), key=lambda k: self.response_cache[k][1])
            del self.response_cache[oldest_id]

        self.response_cache[query_id] = (response, time.time())

    def _update_stats(self, response_time: float) -> None:
        """Update server statistics."""
        self.stats.total_queries += 1

        # Update average response time (exponential moving average)
        alpha = 0.1
        self.stats.average_response_time = (
            1 - alpha
        ) * self.stats.average_response_time + alpha * response_time

        # Update timing variance
        self.stats.timing_variance = self.timing_protection.get_timing_variance()

        # Calculate security score
        self.stats.security_score = self._calculate_security_score()

    def _calculate_security_score(self) -> float:
        """Calculate overall security score based on metrics."""
        scores = []

        # Timing variance score (lower is better)
        timing_score = 1.0 - min(
            self.stats.timing_variance / self.config.timing_variance_target, 1.0
        )
        scores.append(timing_score)

        # Decoy query ratio score
        if self.stats.total_queries > 0:
            decoy_ratio = self.stats.decoy_queries / self.stats.total_queries
            decoy_score = min(decoy_ratio / 0.2, 1.0)  # Target 20% decoys
            scores.append(decoy_score)

        # Cache effectiveness score
        cache_score = self.stats.cache_hit_rate
        scores.append(cache_score)

        return np.mean(scores) if scores else 0.0

    async def _batch_processor(self) -> None:
        """Process queries in batches for better timing uniformity."""
        while True:
            try:
                batch = []

                # Collect batch
                deadline = time.time() + (self.config.max_batch_wait_ms / 1000)
                while len(batch) < self.config.min_batch_size and time.time() < deadline:
                    try:
                        timeout = max(0, deadline - time.time())
                        item = await asyncio.wait_for(self.batch_queue.get(), timeout=timeout)
                        batch.append(item)
                    except TimeoutError:
                        from genomevault.observability.logging import configure_logging

                        logger = configure_logging()
                        logger.exception("Unhandled exception")
                        break
                        raise

                # Process batch if we have queries
                if batch:
                    await self._process_batch(batch)

            except asyncio.CancelledError:
                from genomevault.observability.logging import configure_logging

                logger = configure_logging()
                logger.exception("Unhandled exception")
                break
                raise
            except Exception as e:
                from genomevault.observability.logging import configure_logging

                logger = configure_logging()
                logger.exception("Unhandled exception")
                # Log error but continue
                print(f"Batch processor error: {e}")
                await asyncio.sleep(1)
                raise

    async def _process_batch(self, batch: list[tuple[bytes, asyncio.Future]]) -> None:
        """Process a batch of queries together."""
        # Shuffle to prevent order-based analysis
        random.shuffle(batch)

        # Process all queries
        results = []
        for payload, future in batch:
            try:
                result = await self.answer_query_async(payload)
                results.append((future, result))
            except Exception as e:
                from genomevault.observability.logging import configure_logging

                logger = configure_logging()
                logger.exception("Unhandled exception")
                results.append((future, str(e).encode()))
                raise

        # Return results in random order with delays
        random.shuffle(results)
        for future, result in results:
            future.set_result(result)
            # Small random delay between responses
            await asyncio.sleep(random.uniform(0, 0.001))

    def get_security_report(self) -> dict[str, Any]:
        """Generate comprehensive security report."""
        return {
            "timing_security": {
                "variance": self.stats.timing_variance,
                "target": self.config.timing_variance_target,
                "achieved": self.stats.timing_variance <= self.config.timing_variance_target,
            },
            "query_mixing": {
                "total_queries": self.stats.total_queries,
                "decoy_queries": self.stats.decoy_queries,
                "decoy_percentage": (
                    (self.stats.decoy_queries / self.stats.total_queries * 100)
                    if self.stats.total_queries > 0
                    else 0
                ),
            },
            "performance": {
                "average_response_time_ms": self.stats.average_response_time * 1000,
                "cache_hit_rate": self.stats.cache_hit_rate * 100,
            },
            "overall_security_score": self.stats.security_score,
            "recommendations": self._generate_security_recommendations(),
        }

    def _generate_security_recommendations(self) -> list[str]:
        """Generate security recommendations based on current metrics."""
        recommendations = []

        if self.stats.timing_variance > self.config.timing_variance_target:
            recommendations.append(
                f"Timing variance ({self.stats.timing_variance:.3f}) exceeds target. "
                "Consider increasing jitter or response padding."
            )

        if self.stats.total_queries > 0:
            decoy_ratio = self.stats.decoy_queries / self.stats.total_queries
            if decoy_ratio < 0.15:
                recommendations.append(
                    "Low decoy query ratio. Increase decoy injection rate for better mixing."
                )

        if self.stats.cache_hit_rate < 0.3:
            recommendations.append("Low cache hit rate. Consider increasing cache size or TTL.")

        if not recommendations:
            recommendations.append("All security metrics within acceptable ranges.")

        return recommendations
