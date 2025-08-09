"""
Shard manager for distributed PIR database.
Handles data distribution, updates, and integrity verification.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from genomevault.utils.config import get_config

config = get_config()
from genomevault.utils.logging import get_logger, logger, performance_logger

logger = get_logger(__name__)


@dataclass
class ShardMetadata:
    """Metadata for a database shard."""

    shard_id: str
    shard_index: int
    data_type: str
    version: str
    created_timestamp: float
    size_bytes: int
    item_count: int
    checksum: str
    genomic_regions: list[dict] | None = None
    populations: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.shard_id,
            "index": self.shard_index,
            "data_type": self.data_type,
            "version": self.version,
            "created": self.created_timestamp,
            "size": self.size_bytes,
            "items": self.item_count,
            "checksum": self.checksum,
            "regions": self.genomic_regions,
            "populations": self.populations,
        }


@dataclass
class ShardDistribution:
    """Distribution strategy for shards across servers."""

    strategy: str  # 'replicated', 'striped', 'hybrid'
    replication_factor: int
    server_assignments: dict[str, list[str]] = field(default_factory=dict)

    def assign_shard(self, shard_id: str, server_ids: list[str]) -> None:
        """Assign shard to servers."""
        self.server_assignments[shard_id] = server_ids

    def get_servers_for_shard(self, shard_id: str) -> list[str]:
        """Get servers hosting a shard."""
        return self.server_assignments.get(shard_id, [])


class ShardManager:
    """
    Manages database sharding for PIR system.
    Handles creation, distribution, and maintenance of data shards.
    """

    def __init__(self, data_directory: Path, num_shards: int = 10):
        """
        Initialize shard manager.

        Args:
            data_directory: Base directory for shard storage
            num_shards: Number of shards to create
        """
        self.data_directory = Path(data_directory)
        self.num_shards = num_shards

        # Create directory if needed
        self.data_directory.mkdir(parents=True, exist_ok=True)

        # Shard metadata
        self.shards: dict[str, ShardMetadata] = {}
        self.shard_distribution = ShardDistribution(strategy="hybrid", replication_factor=3)

        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Lock for thread safety
        self.lock = threading.Lock()

        # Load existing shards
        self._load_shard_metadata()

        logger.info("ShardManager initialized with len(self.shards) shards")

    def _load_shard_metadata(self) -> None:
        """Load shard metadata from manifest."""
        manifest_path = self.data_directory / "shard_manifest.json"

        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

            for shard_data in manifest.get("shards", []):
                metadata = ShardMetadata(
                    shard_id=shard_data["id"],
                    shard_index=shard_data["index"],
                    data_type=shard_data["data_type"],
                    version=shard_data["version"],
                    created_timestamp=shard_data["created"],
                    size_bytes=shard_data["size"],
                    item_count=shard_data["items"],
                    checksum=shard_data["checksum"],
                    genomic_regions=shard_data.get("regions"),
                    populations=shard_data.get("populations"),
                )
                self.shards[metadata.shard_id] = metadata

            # Load distribution
            if "distribution" in manifest:
                dist = manifest["distribution"]
                self.shard_distribution = ShardDistribution(
                    strategy=dist["strategy"],
                    replication_factor=dist["replication_factor"],
                    server_assignments=dist.get("assignments", {}),
                )

    def _save_shard_metadata(self) -> None:
        """Save shard metadata to manifest."""
        manifest = {
            "version": "1.0",
            "created": time.time(),
            "shards": [
                {**shard.to_dict(), "filename": "shard_{shard.shard_index:04d}.dat"}
                for shard in self.shards.values()
            ],
            "distribution": {
                "strategy": self.shard_distribution.strategy,
                "replication_factor": self.shard_distribution.replication_factor,
                "assignments": self.shard_distribution.server_assignments,
            },
        }

        manifest_path = self.data_directory / "shard_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @performance_logger.log_operation("create_shards")
    def create_shards_from_data(self, data_source: Path, data_type: str = "genomic") -> list[str]:
        """
        Create shards from source data.

        Args:
            data_source: Path to source data
            data_type: Type of data being sharded

        Returns:
            List of created shard IDs
        """
        logger.info("Creating self.num_shards shards from data_source")

        # Read source data
        with open(data_source, "rb") as f:
            data = f.read()

        total_size = len(data)
        shard_size = total_size // self.num_shards

        created_shards = []

        # Create shards in parallel
        futures = []
        for i in range(self.num_shards):
            start_idx = i * shard_size
            end_idx = start_idx + shard_size if i < self.num_shards - 1 else total_size

            shard_data = data[start_idx:end_idx]

            future = self.executor.submit(
                self._create_single_shard,
                shard_index=i,
                shard_data=shard_data,
                data_type=data_type,
            )
            futures.append(future)

        # Wait for all shards to be created
        for future in futures:
            shard_id = future.result()
            if shard_id:
                created_shards.append(shard_id)

        # Save metadata
        with self.lock:
            self._save_shard_metadata()

        logger.info("Created len(created_shards) shards")
        return created_shards

    def _create_single_shard(
        self, shard_index: int, shard_data: bytes, data_type: str
    ) -> str | None:
        """
        Create a single shard.

        Args:
            shard_index: Index of the shard
            shard_data: Data for the shard
            data_type: Type of data

        Returns:
            Shard ID if successful
        """
        try:
            # Generate shard ID
            shard_id = "{data_type}_{shard_index:04d}_{int(time.time())}"

            # Calculate checksum
            checksum = hashlib.sha256(shard_data).hexdigest()

            # Determine item count based on data type
            if data_type == "genomic":
                item_size = 100
            elif data_type == "annotation":
                item_size = 50
            else:
                item_size = 200

            item_count = len(shard_data) // item_size

            # Write shard data
            shard_path = self.data_directory / "shard_{shard_index:04d}.dat"
            with open(shard_path, "wb") as f:
                f.write(shard_data)

            # Create metadata
            metadata = ShardMetadata(
                shard_id=shard_id,
                shard_index=shard_index,
                data_type=data_type,
                version="1.0",
                created_timestamp=time.time(),
                size_bytes=len(shard_data),
                item_count=item_count,
                checksum=checksum,
            )

            # Store metadata
            with self.lock:
                self.shards[shard_id] = metadata

            logger.info("Created shard shard_id with item_count items")
            return shard_id

        except Exception:
            logger.exception("Unhandled exception")
            logger.error("Error creating shard shard_index: e")
            return None
            raise RuntimeError("Unspecified error")

    def distribute_shards(self, server_list: list[str]) -> ShardDistribution:
        """
        Distribute shards across servers.

        Args:
            server_list: List of available server IDs

        Returns:
            Distribution strategy
        """
        if len(server_list) < self.shard_distribution.replication_factor:
            raise ValueError(
                "Insufficient servers: {len(server_list)} < {self.shard_distribution.replication_factor}"
            )

        # Clear existing assignments
        self.shard_distribution.server_assignments.clear()

        # Distribute shards
        for shard_id in self.shards:
            # Select servers for this shard
            if self.shard_distribution.strategy == "replicated":
                # All servers get all shards
                assigned_servers = server_list[: self.shard_distribution.replication_factor]

            elif self.shard_distribution.strategy == "striped":
                # Round-robin distribution
                shard_idx = list(self.shards.keys()).index(shard_id)
                start_idx = shard_idx % len(server_list)
                assigned_servers = []
                for i in range(self.shard_distribution.replication_factor):
                    server_idx = (start_idx + i) % len(server_list)
                    assigned_servers.append(server_list[server_idx])

            else:  # hybrid
                # Mix of replication and striping
                # First 2 replicas on TS servers, additional on LN
                ts_servers = [s for s in server_list if s.startswith("ts")]
                ln_servers = [s for s in server_list if s.startswith("ln")]

                assigned_servers = ts_servers[:2]
                if len(assigned_servers) < self.shard_distribution.replication_factor:
                    needed = self.shard_distribution.replication_factor - len(assigned_servers)
                    assigned_servers.extend(ln_servers[:needed])

            self.shard_distribution.assign_shard(shard_id, assigned_servers)

        # Save distribution
        with self.lock:
            self._save_shard_metadata()

        logger.info("Distributed len(self.shards) shards across len(server_list) servers")
        return self.shard_distribution

    def verify_shard_integrity(self, shard_id: str) -> bool:
        """
        Verify integrity of a shard.

        Args:
            shard_id: Shard to verify

        Returns:
            True if integrity check passes
        """
        if shard_id not in self.shards:
            logger.error("Unknown shard: shard_id")
            return False

        metadata = self.shards[shard_id]
        shard_path = self.data_directory / "shard_{metadata.shard_index:04d}.dat"

        if not shard_path.exists():
            logger.error("Shard file missing: shard_path")
            return False

        # Calculate checksum
        with open(shard_path, "rb") as f:
            data = f.read()
            checksum = hashlib.sha256(data).hexdigest()

        if checksum != metadata.checksum:
            logger.error("Shard shard_id checksum mismatch")
            return False

        return True

    def update_shard(self, shard_id: str, new_data: bytes) -> bool:
        """
        Update a shard with new data.

        Args:
            shard_id: Shard to update
            new_data: New data for the shard

        Returns:
            Success status
        """
        if shard_id not in self.shards:
            logger.error("Unknown shard: shard_id")
            return False

        metadata = self.shards[shard_id]
        shard_path = self.data_directory / "shard_{metadata.shard_index:04d}.dat"

        # Backup existing shard
        backup_path = shard_path.with_suffix(".bak")
        shutil.copy2(shard_path, backup_path)

        try:
            # Write new data
            with open(shard_path, "wb") as f:
                f.write(new_data)

            # Update metadata
            with self.lock:
                metadata.checksum = hashlib.sha256(new_data).hexdigest()
                metadata.size_bytes = len(new_data)
                metadata.version = "{float(metadata.version) + 0.1:.1f}"

                # Recalculate item count
                if metadata.data_type == "genomic":
                    item_size = 100
                elif metadata.data_type == "annotation":
                    item_size = 50
                else:
                    item_size = 200

                metadata.item_count = len(new_data) // item_size

                # Save metadata
                self._save_shard_metadata()

            # Remove backup
            backup_path.unlink()

            logger.info("Updated shard shard_id to version metadata.version")
            return True

        except Exception:
            logger.exception("Unhandled exception")
            logger.error("Error updating shard shard_id: e")

            # Restore backup
            if backup_path.exists():
                shutil.copy2(backup_path, shard_path)
                backup_path.unlink()

            return False
            raise RuntimeError("Unspecified error")

    def get_shard_statistics(self) -> dict[str, Any]:
        """
        Get statistics about shards.

        Returns:
            Shard statistics
        """
        total_size = sum(s.size_bytes for s in self.shards.values())
        total_items = sum(s.item_count for s in self.shards.values())

        by_type = {}
        for shard in self.shards.values():
            if shard.data_type not in by_type:
                by_type[shard.data_type] = {"count": 0, "size": 0, "items": 0}
            by_type[shard.data_type]["count"] += 1
            by_type[shard.data_type]["size"] += shard.size_bytes
            by_type[shard.data_type]["items"] += shard.item_count

        return {
            "total_shards": len(self.shards),
            "total_size_bytes": total_size,
            "total_items": total_items,
            "by_type": by_type,
            "distribution": {
                "strategy": self.shard_distribution.strategy,
                "replication_factor": self.shard_distribution.replication_factor,
                "servers_used": len(
                    {
                        server
                        for servers in self.shard_distribution.server_assignments.values()
                        for server in servers
                    }
                ),
            },
        }

    def optimize_distribution(self, server_stats: dict[str, dict]) -> ShardDistribution:
        """
        Optimize shard distribution based on server performance.

        Args:
            server_stats: Performance statistics for each server

        Returns:
            Optimized distribution
        """
        # Sort servers by performance (lower latency is better)
        sorted_servers = sorted(
            server_stats.items(), key=lambda x: x[1].get("avg_latency_ms", float("inf"))
        )

        # Rebalance shards
        new_distribution = ShardDistribution(
            strategy="optimized",
            replication_factor=self.shard_distribution.replication_factor,
        )

        # Assign most accessed shards to fastest servers
        # This is a simplified optimization
        for i, (shard_id, metadata) in enumerate(self.shards.items()):
            # Use round-robin with bias towards faster servers
            assigned_servers = []
            for j in range(self.shard_distribution.replication_factor):
                server_idx = (i + j) % len(sorted_servers)
                assigned_servers.append(sorted_servers[server_idx][0])

            new_distribution.assign_shard(shard_id, assigned_servers)

        logger.info("Optimized shard distribution based on server performance")
        return new_distribution

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.executor.shutdown(wait=True)


# Example usage
if __name__ == "__main__":
    import tempfile

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = ShardManager(Path(temp_dir), num_shards=5)

        # Create test data
        test_data = b"A" * 100000  # 100KB of test data
        test_file = Path(temp_dir) / "test_data.bin"
        test_file.write_bytes(test_data)

        # Create shards
        shard_ids = manager.create_shards_from_data(test_file, data_type="genomic")
        logger.info("Created {len(shard_ids)} shards")

        # Distribute shards
        servers = ["ts1", "ts2", "ln1", "ln2", "ln3"]
        distribution = manager.distribute_shards(servers)

        # Show distribution
        for shard_id, servers in distribution.server_assignments.items():
            logger.info("Shard {shard_id} -> {servers}")

        # Get statistics
        stats = manager.get_shard_statistics()
        logger.info("\nStatistics: {json.dumps(stats, indent=2)}")
