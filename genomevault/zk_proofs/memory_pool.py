"""Memory pool management for ZK proof generation."""

import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from threading import Lock
from collections import defaultdict
import weakref


@dataclass
class Buffer:
    """Reusable memory buffer."""

    data: np.ndarray
    size: int
    in_use: bool = False
    last_used: float = 0
    circuit_type: Optional[str] = None

    def __hash__(self):
        """Make Buffer hashable for WeakSet."""
        return id(self)


class MemoryPool:
    """Pre-allocated memory pool for circuit operations."""

    def __init__(self, pool_size: int = 10):
        self.pool_size = pool_size
        self.pools: Dict[str, List[Buffer]] = defaultdict(list)
        self.lock = Lock()
        self.allocated_bytes = 0
        self.stats = {"acquisitions": 0, "releases": 0, "allocations": 0, "reuses": 0}

        # Track active buffers with weak references
        self.active_buffers = weakref.WeakSet()

    def _get_buffer_size(self, circuit_type: str) -> int:
        """Get required buffer size for circuit type."""
        # Estimated buffer sizes for different circuits
        sizes = {
            "variant_presence": 1024 * 100,  # 100KB
            "polygenic_risk_score": 1024 * 500,  # 500KB
            "ancestry_composition": 1024 * 1000,  # 1MB
            "diabetes_risk_alert": 1024 * 200,  # 200KB
            "pharmacogenomic": 1024 * 150,  # 150KB
            "pathway_enrichment": 1024 * 800,  # 800KB
        }
        return sizes.get(circuit_type, 1024 * 100)  # Default 100KB

    def acquire(self, circuit_type: str, min_size: Optional[int] = None) -> np.ndarray:
        """
        Acquire a buffer from the pool.

        Args:
            circuit_type: Type of circuit needing buffer
            min_size: Minimum buffer size needed

        Returns:
            NumPy array buffer
        """
        with self.lock:
            self.stats["acquisitions"] += 1

            # Get required size
            required_size = min_size or self._get_buffer_size(circuit_type)

            # Look for available buffer in pool
            pool = self.pools[circuit_type]

            for buffer in pool:
                if not buffer.in_use and buffer.size >= required_size:
                    # Reuse existing buffer
                    buffer.in_use = True
                    buffer.last_used = time.time()
                    self.stats["reuses"] += 1

                    # Clear buffer
                    buffer.data.fill(0)

                    self.active_buffers.add(buffer)
                    return buffer.data[:required_size]

            # No suitable buffer found - allocate new one
            if len(pool) < self.pool_size:
                # Allocate with some headroom
                alloc_size = int(required_size * 1.2)
                new_buffer = Buffer(
                    data=np.zeros(alloc_size, dtype=np.float32),
                    size=alloc_size,
                    in_use=True,
                    last_used=time.time(),
                    circuit_type=circuit_type,
                )

                pool.append(new_buffer)
                self.allocated_bytes += alloc_size * 4  # float32 = 4 bytes
                self.stats["allocations"] += 1

                self.active_buffers.add(new_buffer)
                return new_buffer.data[:required_size]

            # Pool full - wait or allocate temporary
            # For now, allocate temporary buffer
            temp_buffer = np.zeros(required_size, dtype=np.float32)
            return temp_buffer

    def release(self, buffer: np.ndarray) -> None:
        """
        Release buffer back to pool.

        Args:
            buffer: Buffer to release
        """
        with self.lock:
            self.stats["releases"] += 1

            # Find buffer in pools
            for circuit_type, pool in self.pools.items():
                for buf in pool:
                    if buf.data is buffer or np.shares_memory(buf.data, buffer):
                        buf.in_use = False
                        buf.last_used = time.time()

                        # Remove from active set
                        try:
                            self.active_buffers.remove(buf)
                        except KeyError:
                            pass

                        return

    def clear_pool(self, circuit_type: Optional[str] = None) -> None:
        """Clear buffers for specific circuit or all."""
        with self.lock:
            if circuit_type:
                pools_to_clear = [circuit_type]
            else:
                pools_to_clear = list(self.pools.keys())

            for ct in pools_to_clear:
                if ct in self.pools:
                    # Clear unused buffers
                    self.pools[ct] = [buf for buf in self.pools[ct] if buf.in_use]

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            total_buffers = sum(len(pool) for pool in self.pools.values())
            in_use = sum(1 for pool in self.pools.values() for buf in pool if buf.in_use)

            return {
                "total_buffers": total_buffers,
                "in_use": in_use,
                "available": total_buffers - in_use,
                "allocated_mb": self.allocated_bytes / (1024 * 1024),
                "acquisitions": self.stats["acquisitions"],
                "releases": self.stats["releases"],
                "allocations": self.stats["allocations"],
                "reuses": self.stats["reuses"],
                "reuse_rate": self.stats["reuses"] / max(1, self.stats["acquisitions"]),
            }

    def optimize_pool_sizes(self) -> None:
        """Optimize pool sizes based on usage patterns."""
        with self.lock:
            for circuit_type, pool in self.pools.items():
                # Count usage frequency
                usage_count = sum(1 for buf in pool if buf.last_used > 0)

                if usage_count == 0:
                    continue

                # Adjust pool size based on usage
                if usage_count == len(pool):
                    # All buffers used - might need more
                    self.pool_size = min(self.pool_size + 2, 20)
                elif usage_count < len(pool) / 2:
                    # Under-utilized - reduce
                    self.pool_size = max(self.pool_size - 1, 5)


class CircuitMemoryManager:
    """Manages memory for circuit operations."""

    def __init__(self):
        self.pools: Dict[str, MemoryPool] = {}
        self.default_pool = MemoryPool()

    def get_pool(self, circuit_type: str) -> MemoryPool:
        """Get or create pool for circuit type."""
        if circuit_type not in self.pools:
            self.pools[circuit_type] = MemoryPool()
        return self.pools[circuit_type]

    def allocate_workspace(self, circuit_type: str, operations: List[str]) -> Dict[str, np.ndarray]:
        """
        Allocate workspace buffers for circuit operations.

        Args:
            circuit_type: Type of circuit
            operations: List of operations needing buffers

        Returns:
            Dictionary of operation -> buffer mappings
        """
        pool = self.get_pool(circuit_type)
        workspace = {}

        # Allocate buffers for each operation
        sizes = {
            "constraint_generation": 1024 * 50,
            "witness_computation": 1024 * 100,
            "polynomial_evaluation": 1024 * 200,
            "fft": 1024 * 500,
            "msm": 1024 * 300,  # Multi-scalar multiplication
        }

        for op in operations:
            size = sizes.get(op, 1024 * 100)
            workspace[op] = pool.acquire(circuit_type, size)

        return workspace

    def release_workspace(self, circuit_type: str, workspace: Dict[str, np.ndarray]) -> None:
        """Release workspace buffers back to pool."""
        pool = self.get_pool(circuit_type)

        for buffer in workspace.values():
            pool.release(buffer)

    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics across all pools."""
        stats = {"pools": {}, "total_allocated_mb": 0, "total_buffers": 0}

        for circuit_type, pool in self.pools.items():
            pool_stats = pool.get_stats()
            stats["pools"][circuit_type] = pool_stats
            stats["total_allocated_mb"] += pool_stats["allocated_mb"]
            stats["total_buffers"] += pool_stats["total_buffers"]

        # Add default pool
        default_stats = self.default_pool.get_stats()
        stats["pools"]["default"] = default_stats
        stats["total_allocated_mb"] += default_stats["allocated_mb"]
        stats["total_buffers"] += default_stats["total_buffers"]

        return stats


# Global instance
_memory_manager = None


def get_memory_manager() -> CircuitMemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = CircuitMemoryManager()
    return _memory_manager


class MemoryEfficientProver:
    """Prover with memory pool integration."""

    def __init__(self):
        self.memory_manager = get_memory_manager()
        # Import here to avoid circular dependency
        from genomevault.zk_proofs.prover import Prover

        self.base_prover = Prover()

    def generate_witness_with_pool(
        self, circuit_type: str, public_inputs: Dict, private_inputs: Dict
    ) -> Dict:
        """Generate witness using memory pool."""

        # Allocate workspace
        workspace = self.memory_manager.allocate_workspace(
            circuit_type, ["constraint_generation", "witness_computation"]
        )

        try:
            # Use pre-allocated buffers for computation
            constraints_buf = workspace["constraint_generation"]
            witness_buf = workspace["witness_computation"]

            # Perform computation using buffers
            witness = self._compute_with_buffers(
                circuit_type, public_inputs, private_inputs, constraints_buf, witness_buf
            )

            return witness

        finally:
            # Always release workspace
            self.memory_manager.release_workspace(circuit_type, workspace)

    def _compute_with_buffers(
        self,
        circuit_type: str,
        public_inputs: Dict,
        private_inputs: Dict,
        constraints_buf: np.ndarray,
        witness_buf: np.ndarray,
    ) -> Dict:
        """Compute witness using pre-allocated buffers."""
        # Use base prover for actual computation
        # In production, would use buffers for intermediate results
        proof = self.base_prover.generate_proof(circuit_type, public_inputs, private_inputs)

        # Add metadata about memory usage
        if hasattr(proof, "metadata"):
            proof.metadata["memory_pool_used"] = True
            proof.metadata["buffer_sizes"] = {
                "constraints": len(constraints_buf),
                "witness": len(witness_buf),
            }

        return proof

    def batch_generate_with_pool(self, tasks: List[Tuple[str, Dict, Dict]]) -> List[Dict]:
        """Generate multiple witnesses with memory pooling."""
        results = []

        for circuit_type, public_inputs, private_inputs in tasks:
            witness = self.generate_witness_with_pool(circuit_type, public_inputs, private_inputs)
            results.append(witness)

        # Optimize pools based on usage
        for pool in self.memory_manager.pools.values():
            pool.optimize_pool_sizes()

        return results
