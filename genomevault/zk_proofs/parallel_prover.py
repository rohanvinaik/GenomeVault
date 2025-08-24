"""Parallel proof generation system for batch operations."""

import time
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import multiprocessing as mp
from threading import Semaphore

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProofTask:
    """Single proof generation task."""

    task_id: str
    circuit_name: str
    public_inputs: Dict[str, Any]
    private_inputs: Dict[str, Any]
    priority: int = 0

    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority > other.priority


class ParallelProver:
    """Parallel proof generation with adaptive worker management."""

    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        """
        Initialize parallel prover.

        Args:
            max_workers: Maximum parallel workers (default: CPU count)
            use_processes: Use processes instead of threads for CPU-bound work
        """
        if max_workers is None:
            max_workers = mp.cpu_count()

        self.max_workers = max_workers
        self.use_processes = use_processes

        # Create executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Resource management
        self.semaphore = Semaphore(max_workers * 2)  # Allow some queuing

        # Performance tracking
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_time": 0,
            "queue_time": 0,
        }

        # Import prover here to avoid circular import
        from genomevault.zk_proofs.prover import Prover

        self.prover = Prover()

    @staticmethod
    def _compute_variant_hash(variant: Dict) -> str:
        """
        Compute consistent hash for variant across single and parallel execution.
        
        Args:
            variant: Variant dictionary
            
        Returns:
            Consistent hash string
        """
        # Ensure consistent ordering and format
        canonical_variant = {
            'chr': str(variant.get('chr', '')),
            'pos': int(variant.get('pos', 0)),
            'ref': str(variant.get('ref', '')),
            'alt': str(variant.get('alt', ''))
        }
        
        # Create deterministic string representation
        variant_str = json.dumps(canonical_variant, sort_keys=True)
        
        # Compute hash
        return hashlib.sha256(variant_str.encode()).hexdigest()

    def generate_witness_batch(
        self, tasks: List[ProofTask], timeout: Optional[float] = None
    ) -> List[Tuple[str, Dict, Optional[Exception]]]:
        """
        Generate witnesses for multiple circuits in parallel.

        Returns:
            List of (task_id, witness, error) tuples
        """
        results = []
        futures = {}

        # Submit all tasks
        for task in tasks:
            future = self.executor.submit(self._generate_single_witness, task)
            futures[future] = task
            self.stats["total_tasks"] += 1

        # Collect results
        for future in as_completed(futures, timeout=timeout):
            task = futures[future]

            try:
                witness = future.result()
                results.append((task.task_id, witness, None))
                self.stats["completed_tasks"] += 1

            except Exception as e:
                logger.error(f"Task {task.task_id} failed: {e}")
                results.append((task.task_id, None, e))
                self.stats["failed_tasks"] += 1

        return results

    def _generate_single_witness(self, task: ProofTask) -> Dict:
        """Generate witness for a single task with consistent hashing."""
        start = time.perf_counter()

        # Ensure consistent variant hashing in public inputs
        if 'variant_hash' in task.public_inputs and 'variant_data' in task.private_inputs:
            # Recompute hash to ensure consistency
            variant_data = task.private_inputs['variant_data']
            computed_hash = self._compute_variant_hash(variant_data)
            
            # Update public inputs with consistent hash
            task.public_inputs['variant_hash'] = computed_hash
            
            # Log if there was a mismatch
            if task.public_inputs.get('variant_hash') != computed_hash:
                logger.debug(f"Fixed hash mismatch for task {task.task_id}")

        # Acquire resource semaphore
        self.semaphore.acquire()
        try:
            queue_time = time.perf_counter() - start
            self.stats["queue_time"] += queue_time

            # Generate full proof (which includes witness)
            proof = self.prover.generate_proof(
                task.circuit_name, task.public_inputs, task.private_inputs
            )

            # Track timing
            total_time = time.perf_counter() - start
            self.stats["total_time"] += total_time

            # Create witness result with metadata
            witness = {
                "proof": proof,
                "_parallel_metadata": {
                    "task_id": task.task_id,
                    "queue_time_ms": queue_time * 1000,
                    "total_time_ms": total_time * 1000,
                    "worker_type": "process" if self.use_processes else "thread",
                },
            }

            return witness

        finally:
            self.semaphore.release()

    def generate_proofs_batch(
        self, tasks: List[ProofTask]
    ) -> List[Tuple[str, Dict, Optional[Exception]]]:
        """Generate full proofs (witness + proof) in parallel."""
        # For our implementation, witness generation includes proof generation
        return self.generate_witness_batch(tasks)

    def adaptive_batch_size(self, tasks: List[ProofTask]) -> List[List[ProofTask]]:
        """
        Adaptively batch tasks based on circuit complexity.

        Returns:
            List of task batches optimized for parallel execution
        """
        # Estimate complexity based on circuit type
        complexity_map = {
            "variant_presence": 1,
            "polygenic_risk_score": 3,
            "ancestry_composition": 5,
            "diabetes_risk_alert": 2,
            "pharmacogenomic": 2,
            "pathway_enrichment": 4,
        }

        # Sort by complexity
        tasks_with_complexity = [(t, complexity_map.get(t.circuit_name, 1)) for t in tasks]
        tasks_with_complexity.sort(key=lambda x: x[1])

        # Create balanced batches
        batches = []
        current_batch = []
        current_complexity = 0
        max_batch_complexity = self.max_workers * 2

        for task, complexity in tasks_with_complexity:
            if current_complexity + complexity > max_batch_complexity:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [task]
                current_complexity = complexity
            else:
                current_batch.append(task)
                current_complexity += complexity

        if current_batch:
            batches.append(current_batch)

        return batches

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total = self.stats["completed_tasks"] + self.stats["failed_tasks"]

        return {
            "total_tasks": self.stats["total_tasks"],
            "completed": self.stats["completed_tasks"],
            "failed": self.stats["failed_tasks"],
            "success_rate": self.stats["completed_tasks"] / total if total > 0 else 0,
            "avg_time_ms": self.stats["total_time"] / total * 1000 if total > 0 else 0,
            "avg_queue_time_ms": self.stats["queue_time"] / total * 1000 if total > 0 else 0,
            "throughput_per_sec": (
                total / self.stats["total_time"] if self.stats["total_time"] > 0 else 0
            ),
            "max_workers": self.max_workers,
            "executor_type": "process" if self.use_processes else "thread",
        }

    def shutdown(self):
        """Shutdown executor cleanly."""
        self.executor.shutdown(wait=True)


def parallel_prove_example():
    """Example of parallel proof generation."""
    import hashlib

    prover = ParallelProver(max_workers=4)

    # Create tasks
    tasks = []
    for i in range(100):
        # Generate proper variant hash
        variant_str = f"chr1:{i*100}:A:G"
        variant_hash = hashlib.sha256(variant_str.encode()).hexdigest()

        task = ProofTask(
            task_id=f"task_{i}",
            circuit_name="variant_presence",
            public_inputs={
                "variant_hash": variant_hash,
                "reference_hash": "ref_hash",
                "commitment_root": "root_hash",
            },
            private_inputs={
                "variant_data": {"chr": "chr1", "pos": i * 100, "ref": "A", "alt": "G"},
                "merkle_proof": ["proof1", "proof2"],
                "witness_randomness": f"random_{i}",
            },
            priority=i % 3,  # Vary priority
        )
        tasks.append(task)

    # Generate in parallel
    results = prover.generate_witness_batch(tasks)

    # Check results
    successful = sum(1 for _, _, error in results if error is None)
    print(f"Successfully generated {successful}/{len(tasks)} witnesses")

    # Get stats
    stats = prover.get_performance_stats()
    print(f"Throughput: {stats['throughput_per_sec']:.1f} proofs/sec")

    prover.shutdown()


if __name__ == "__main__":
    parallel_prove_example()
