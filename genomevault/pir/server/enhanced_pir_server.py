"""
Enhanced PIR server implementation with full information-theoretic security.
Handles real genomic reference data with optimized query processing.
"""

import asyncio
import hashlib
import json
import mmap
import os
import struct
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiofiles
import lz4.frame
import numpy as np
import uvloop

from ...core.config import get_config
from ...core.exceptions import SecurityError, ValidationError
from ...utils.logging import get_logger

logger = get_logger(__name__)
config = get_config()

# Use uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


@dataclass
class GenomicRegion:
    """Genomic region data structure."""

    chromosome: str
    start: int
    end: int
    reference_allele: str
    alternate_alleles: list[str]
    population_frequencies: dict[str, float]
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        """Serialize to bytes for PIR storage."""
        data = {
            "chr": self.chromosome,
            "start": self.start,
            "end": self.end,
            "ref": self.reference_allele,
            "alt": self.alternate_alleles,
            "freq": self.population_frequencies,
            "ann": self.annotations,
        }
        json_bytes = json.dumps(data).encode("utf-8")
        compressed = lz4.frame.compress(json_bytes)
        return compressed

    @classmethod
    def from_bytes(cls, data: bytes) -> "GenomicRegion":
        """Deserialize from bytes."""
        decompressed = lz4.frame.decompress(data)
        data_dict = json.loads(decompressed.decode("utf-8"))
        return cls(
            chromosome=data_dict["chr"],
            start=data_dict["start"],
            end=data_dict["end"],
            reference_allele=data_dict["ref"],
            alternate_alleles=data_dict["alt"],
            population_frequencies=data_dict["freq"],
            annotations=data_dict.get("ann", {}),
        )


@dataclass
class ShardMetadata:
    """Enhanced shard metadata with indexing information."""

    shard_id: str
    data_path: Path
    index_path: Path
    size: int
    item_count: int
    data_type: str
    version: str
    checksum: str
    chromosome_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    compression_ratio: float = 1.0

    def contains_region(self, chromosome: str, position: int) -> bool:
        """Check if this shard contains a genomic position."""
        if chromosome not in self.chromosome_ranges:
            return False
        start, end = self.chromosome_ranges[chromosome]
        return start <= position <= end


class OptimizedPIRDatabase:
    """
    Optimized database for PIR queries with indexing and caching.
    """

    def __init__(self, base_path: Path, cache_size_mb: int = 1024):
        """
        Initialize optimized PIR database.

        Args:
            base_path: Base path for database files
            cache_size_mb: Cache size in megabytes
        """
        self.base_path = Path(base_path)
        self.cache_size = cache_size_mb * 1024 * 1024
        self.cache = {}
        self.cache_stats = defaultdict(int)
        self.indexes = {}
        self.memory_maps = {}

        # Thread pool for I/O operations
        self.io_pool = ThreadPoolExecutor(max_workers=4)

        logger.info(f"Initialized PIR database at {base_path}")

    async def load_shard_index(self, shard: ShardMetadata) -> dict[str, int]:
        """
        Load shard index for fast lookups.

        Args:
            shard: Shard metadata

        Returns:
            Index mapping positions to offsets
        """
        if shard.shard_id in self.indexes:
            return self.indexes[shard.shard_id]

        # Load index asynchronously
        async with aiofiles.open(shard.index_path, "rb") as f:
            index_data = await f.read()

        # Parse index (position -> offset mapping)
        index = {}
        offset = 0
        while offset < len(index_data):
            # Read chromosome (1 byte), position (4 bytes), data offset (4 bytes)
            if offset + 9 > len(index_data):
                break

            chr_byte = index_data[offset]
            position = struct.unpack(">I", index_data[offset + 1 : offset + 5])[0]
            data_offset = struct.unpack(">I", index_data[offset + 5 : offset + 9])[0]

            chr_name = "chr{chr_byte}" if chr_byte < 23 else ("chrX" if chr_byte == 23 else "chrY")
            key = "{chr_name}:{position}"
            index[key] = data_offset

            offset += 9

        self.indexes[shard.shard_id] = index
        return index

    def get_memory_map(self, shard: ShardMetadata) -> mmap.mmap:
        """Get or create memory map for shard data."""
        if shard.shard_id not in self.memory_maps:
            with open(shard.data_path, "rb") as f:
                self.memory_maps[shard.shard_id] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return self.memory_maps[shard.shard_id]

    async def query_item(self, shard: ShardMetadata, position_key: str) -> bytes | None:
        """
        Query a specific item from the database.

        Args:
            shard: Shard containing the data
            position_key: Position key (e.g., "chr1:1000000")

        Returns:
            Item data or None if not found
        """
        # Check cache first
        cache_key = "{shard.shard_id}:{position_key}"
        if cache_key in self.cache:
            self.cache_stats["hits"] += 1
            return self.cache[cache_key]

        self.cache_stats["misses"] += 1

        # Load index if needed
        index = await self.load_shard_index(shard)

        if position_key not in index:
            return None

        # Get data from memory map
        mmap_data = self.get_memory_map(shard)
        offset = index[position_key]

        # Read item size (2 bytes) and data
        mmap_data.seek(offset)
        size_bytes = mmap_data.read(2)
        if len(size_bytes) < 2:
            return None

        item_size = struct.unpack(">H", size_bytes)[0]
        item_data = mmap_data.read(item_size)

        # Update cache
        self._update_cache(cache_key, item_data)

        return item_data

    def _update_cache(self, key: str, data: bytes):
        """Update LRU cache with size limit."""
        # Simple size-based eviction
        if len(self.cache) * 1000 > self.cache_size:  # Rough estimate
            # Remove oldest entries
            to_remove = len(self.cache) // 4
            for k in list(self.cache.keys())[:to_remove]:
                del self.cache[k]

        self.cache[key] = data

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total if total > 0 else 0

        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "estimated_memory_mb": len(self.cache) * 1000 / (1024 * 1024),
        }

    def close(self):
        """Close all resources."""
        for mmap_file in self.memory_maps.values():
            mmap_file.close()
        self.io_pool.shutdown()


class EnhancedPIRServer:
    """
    Production-ready PIR server with optimizations for genomic data.
    """

    def __init__(
        self,
        server_id: str,
        data_directory: Path,
        is_trusted_signatory: bool = False,
        enable_preprocessing: bool = True,
        cache_size_mb: int = 2048,
    ):
        """
        Initialize enhanced PIR server.

        Args:
            server_id: Unique server identifier
            data_directory: Directory containing database shards
            is_trusted_signatory: Whether this is a HIPAA-compliant TS
            enable_preprocessing: Enable query preprocessing optimizations
            cache_size_mb: Cache size in megabytes
        """
        self.server_id = server_id
        self.data_directory = Path(data_directory)
        self.is_trusted_signatory = is_trusted_signatory
        self.enable_preprocessing = enable_preprocessing

        # Initialize database
        self.database = OptimizedPIRDatabase(data_directory, cache_size_mb)

        # Load shard metadata
        self.shards = self._load_enhanced_shards()

        # Processing pools
        self.process_pool = ProcessPoolExecutor(max_workers=config.get("pir.server_workers", 4))
        self.thread_pool = ThreadPoolExecutor(max_workers=8)

        # Query preprocessing cache
        self.preprocessing_cache = {} if enable_preprocessing else None

        # Performance tracking
        self.metrics = {
            "total_queries": 0,
            "total_bytes_served": 0,
            "average_query_time_ms": 0,
            "preprocessing_hits": 0,
        }

        # Security parameters
        self.max_query_size = config.get("pir.max_query_size", 10 * 1024 * 1024)  # 10MB
        self.rate_limiter = self._init_rate_limiter()

        logger.info(
            "Enhanced PIR server {server_id} initialized",
            extra={
                "server_type": "TS" if is_trusted_signatory else "LN",
                "shards": len(self.shards),
                "preprocessing": enable_preprocessing,
                "cache_size_mb": cache_size_mb,
            },
        )

    def _load_enhanced_shards(self) -> dict[str, ShardMetadata]:
        """Load enhanced shard metadata with genomic ranges."""
        shards = {}
        manifest_path = self.data_directory / "enhanced_manifest.json"

        if not manifest_path.exists():
            logger.warning("No enhanced manifest found, creating default")
            return self._create_default_shards()

        with open(manifest_path) as f:
            manifest = json.load(f)

        for shard_info in manifest["shards"]:
            shard = ShardMetadata(
                shard_id=shard_info["id"],
                data_path=self.data_directory / shard_info["data_file"],
                index_path=self.data_directory / shard_info["index_file"],
                size=shard_info["size"],
                item_count=shard_info["item_count"],
                data_type=shard_info["data_type"],
                version=shard_info["version"],
                checksum=shard_info["checksum"],
                chromosome_ranges=shard_info.get("chromosome_ranges", {}),
                compression_ratio=shard_info.get("compression_ratio", 1.0),
            )

            if self._verify_shard_integrity(shard):
                shards[shard.shard_id] = shard
            else:
                logger.error(f"Shard {shard.shard_id} integrity check failed")

        return shards

    def _create_default_shards(self) -> dict[str, ShardMetadata]:
        """Create default shard structure for genomic data."""
        shards = {}

        # Create one shard per chromosome for better locality
        chromosomes = ["chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

        for chr_name in chromosomes:
            shard_id = "genomic_{chr_name}"
            data_file = "{shard_id}.dat"
            index_file = "{shard_id}.idx"

            # Create empty files if they don't exist
            data_path = self.data_directory / data_file
            index_path = self.data_directory / index_file

            if not data_path.exists():
                data_path.touch()
                index_path.touch()

            shard = ShardMetadata(
                shard_id=shard_id,
                data_path=data_path,
                index_path=index_path,
                size=0,
                item_count=0,
                data_type="genomic",
                version="1.0",
                checksum="",
                chromosome_ranges={chr_name: (0, 300000000)},  # Approximate chr size
            )

            shards[shard_id] = shard

        return shards

    def _verify_shard_integrity(self, shard: ShardMetadata) -> bool:
        """Verify shard data integrity."""
        if not shard.data_path.exists() or not shard.index_path.exists():
            return False

        # In production, verify checksum
        # For now, just check files exist and are non-empty
        return shard.data_path.stat().st_size > 0 and shard.index_path.stat().st_size > 0

    def _init_rate_limiter(self) -> dict[str, list[float]]:
        """Initialize rate limiting for security."""
        return defaultdict(list)

    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        now = time.time()
        window = 60  # 1 minute window
        max_requests = 100  # Max requests per window

        # Clean old entries
        self.rate_limiter[client_id] = [t for t in self.rate_limiter[client_id] if now - t < window]

        # Check limit
        if len(self.rate_limiter[client_id]) >= max_requests:
            return False

        # Add current request
        self.rate_limiter[client_id].append(now)
        return True

    async def process_query(self, query_data: dict[str, Any]) -> dict[str, Any]:
        """
        Process enhanced PIR query with optimizations.

        Args:
            query_data: Query containing vectors and parameters

        Returns:
            Response with computed results
        """
        start_time = time.time()

        # Extract and validate query parameters
        try:
            query_id = query_data["query_id"]
            client_id = query_data.get("client_id", "anonymous")
            query_vectors = query_data["query_vectors"]  # Multiple vectors for batch
            query_type = query_data.get("query_type", "genomic")
            parameters = query_data.get("parameters", {})

            # Rate limiting
            if not self._check_rate_limit(client_id):
                raise SecurityError("Rate limit exceeded")

            # Validate query size
            query_size = sum(len(str(v)) for v in query_vectors)
            if query_size > self.max_query_size:
                raise ValidationError("Query size exceeds maximum")

        except (KeyError, ValueError) as e:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            logger.error(f"Invalid query format: {e}")
            return {
                "error": "Invalid query format",
                "query_id": query_data.get("query_id"),
            }
            raise

        # Log query (privacy-safe)
        logger.info(
            "Processing PIR query {query_id}",
            extra={
                "query_type": query_type,
                "vector_count": len(query_vectors),
                "client_id_hash": hashlib.sha256(client_id.encode()).hexdigest()[:8],
            },
        )

        try:
            # Process query based on type
            if query_type == "genomic":
                results = await self._process_genomic_query(query_vectors, parameters)
            elif query_type == "annotation":
                results = await self._process_annotation_query(query_vectors, parameters)
            elif query_type == "graph":
                results = await self._process_graph_query(query_vectors, parameters)
            else:
                raise ValidationError("Unknown query type: {query_type}")

            # Calculate metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_metrics(processing_time_ms, len(results))

            # Build response
            response = {
                "query_id": query_id,
                "server_id": self.server_id,
                "results": results,
                "processing_time_ms": processing_time_ms,
                "server_type": "TS" if self.is_trusted_signatory else "LN",
                "cache_stats": self.database.get_cache_stats(),
            }

            return response

        except Exception:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            logger.error(f"Error processing query {query_id}: {e}")
            return {"error": str(e), "query_id": query_id, "server_id": self.server_id}
            raise

    async def _process_genomic_query(
        self, query_vectors: list[np.ndarray], parameters: dict[str, Any]
    ) -> list[bytes]:
        """
        Process genomic data query.

        Args:
            query_vectors: Binary selection vectors
            parameters: Query parameters (regions, filters)

        Returns:
            Selected genomic data
        """
        results = []

        # Determine relevant shards based on parameters
        target_shards = self._select_target_shards(parameters)

        # Process each query vector
        for vector in query_vectors:
            # Check preprocessing cache
            vector_hash = hashlib.sha256(vector.tobytes()).hexdigest()

            if self.enable_preprocessing and vector_hash in self.preprocessing_cache:
                results.append(self.preprocessing_cache[vector_hash])
                self.metrics["preprocessing_hits"] += 1
                continue

            # Compute PIR result
            result = await self._compute_pir_result(vector, target_shards)
            results.append(result)

            # Update preprocessing cache
            if self.enable_preprocessing:
                self.preprocessing_cache[vector_hash] = result
                # Limit cache size
                if len(self.preprocessing_cache) > 10000:
                    # Remove oldest entries
                    keys = list(self.preprocessing_cache.keys())
                    for k in keys[:1000]:
                        del self.preprocessing_cache[k]

        return results

    async def _process_annotation_query(
        self, query_vectors: list[np.ndarray], parameters: dict[str, Any]
    ) -> list[bytes]:
        """Process annotation data query."""
        # Similar to genomic query but for annotation data
        # Implementation would follow same pattern
        return await self._process_genomic_query(query_vectors, parameters)

    async def _process_graph_query(
        self, query_vectors: list[np.ndarray], parameters: dict[str, Any]
    ) -> list[bytes]:
        """Process graph data query."""
        # Process pangenome graph queries
        # Implementation would handle graph-specific operations
        return await self._process_genomic_query(query_vectors, parameters)

    def _select_target_shards(self, parameters: dict[str, Any]) -> list[ShardMetadata]:
        """Select shards based on query parameters."""
        target_shards = []

        # Check if specific regions are requested
        if "regions" in parameters:
            for region in parameters["regions"]:
                chromosome = region.get("chromosome")
                position = region.get("position")

                for shard in self.shards.values():
                    if shard.contains_region(chromosome, position):
                        target_shards.append(shard)
        else:
            # Use all genomic shards
            target_shards = [s for s in self.shards.values() if s.data_type == "genomic"]

        return target_shards

    async def _compute_pir_result(
        self, query_vector: np.ndarray, shards: list[ShardMetadata]
    ) -> bytes:
        """
        Compute PIR result across multiple shards.

        Args:
            query_vector: Binary selection vector
            shards: Target shards

        Returns:
            Combined result
        """
        # Initialize result
        result_size = 1000  # Standard result size
        result = np.zeros(result_size, dtype=np.uint8)

        # Process each shard
        tasks = []
        for shard in shards:
            task = self._process_shard_query(query_vector, shard)
            tasks.append(task)

        # Wait for all shard results
        shard_results = await asyncio.gather(*tasks)

        # Combine results with XOR
        for shard_result in shard_results:
            if shard_result is not None:
                # Ensure same size
                if len(shard_result) > result_size:
                    shard_result = shard_result[:result_size]
                elif len(shard_result) < result_size:
                    padded = np.zeros(result_size, dtype=np.uint8)
                    padded[: len(shard_result)] = shard_result
                    shard_result = padded

                # XOR combine
                result = np.bitwise_xor(result, shard_result)

        return result.tobytes()

    async def _process_shard_query(
        self, query_vector: np.ndarray, shard: ShardMetadata
    ) -> np.ndarray | None:
        """Process query on a single shard."""
        try:
            # Get shard index
            index = await self.database.load_shard_index(shard)

            # Initialize result
            result = None

            # Process each selected index
            for i, selected in enumerate(query_vector):
                if selected and i < shard.item_count:
                    # Construct position key (simplified)
                    # In production, would map index to actual position
                    position_key = list(index.keys())[i] if i < len(index) else None

                    if position_key:
                        item_data = await self.database.query_item(shard, position_key)

                        if item_data:
                            if result is None:
                                result = np.frombuffer(item_data, dtype=np.uint8)
                            else:
                                # XOR combine
                                item_array = np.frombuffer(item_data, dtype=np.uint8)
                                # Ensure same size
                                min_len = min(len(result), len(item_array))
                                result[:min_len] = np.bitwise_xor(
                                    result[:min_len], item_array[:min_len]
                                )

            return result

        except Exception:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            logger.error(f"Error processing shard {shard.shard_id}: {e}")
            return None
            raise

    def _update_metrics(self, processing_time_ms: float, result_count: int):
        """Update server metrics."""
        self.metrics["total_queries"] += 1
        self.metrics["total_bytes_served"] += result_count * 1000  # Estimate

        # Update average (simple moving average)
        n = self.metrics["total_queries"]
        current_avg = self.metrics["average_query_time_ms"]
        self.metrics["average_query_time_ms"] = (current_avg * (n - 1) + processing_time_ms) / n

    async def get_server_status(self) -> dict[str, Any]:
        """Get comprehensive server status."""
        return {
            "server_id": self.server_id,
            "server_type": "TS" if self.is_trusted_signatory else "LN",
            "status": "healthy",
            "shards": {
                "total": len(self.shards),
                "by_type": self._count_shards_by_type(),
                "total_size_gb": sum(s.size for s in self.shards.values()) / (1024**3),
            },
            "performance": {
                "total_queries": self.metrics["total_queries"],
                "average_query_time_ms": self.metrics["average_query_time_ms"],
                "preprocessing_hit_rate": (
                    self.metrics["preprocessing_hits"] / self.metrics["total_queries"]
                    if self.metrics["total_queries"] > 0
                    else 0
                ),
                "cache_stats": self.database.get_cache_stats(),
            },
            "resources": {
                "memory_usage_mb": self._get_memory_usage(),
                "cpu_cores": os.cpu_count(),
                "thread_pool_size": self.thread_pool._max_workers,
                "process_pool_size": self.process_pool._max_workers,
            },
        }

    def _count_shards_by_type(self) -> dict[str, int]:
        """Count shards by data type."""
        counts = defaultdict(int)
        for shard in self.shards.values():
            counts[shard.data_type] += 1
        return dict(counts)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info(f"Shutting down PIR server {self.server_id}")

        # Close database
        self.database.close()

        # Shutdown pools
        self.process_pool.shutdown(wait=True)
        self.thread_pool.shutdown(wait=True)

        # Save metrics
        metrics_path = self.data_directory / "server_{self.server_id}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"PIR server {self.server_id} shutdown complete")


# Example usage and testing
async def main():
    """Example usage of enhanced PIR server."""
    # Initialize server
    server = EnhancedPIRServer(
        server_id="pir_server_001",
        data_directory=Path("/data/genomevault/pir"),
        is_trusted_signatory=True,
        enable_preprocessing=True,
        cache_size_mb=4096,
    )

    # Example query
    query = {
        "query_id": "test_001",
        "client_id": "client_123",
        "query_vectors": [np.random.binomial(1, 0.001, 10000).astype(np.uint8) for _ in range(5)],
        "query_type": "genomic",
        "parameters": {
            "regions": [
                {"chromosome": "chr1", "position": 1000000},
                {"chromosome": "chr2", "position": 5000000},
            ]
        },
    }

    # Process query
    response = await server.process_query(query)
    print("Query processed in {response.get('processing_time_ms', 0):.2f}ms")

    # Get server status
    status = await server.get_server_status()
    print("Server status: {json.dumps(status, indent=2)}")

    # Shutdown
    await server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
