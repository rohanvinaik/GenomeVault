"""
Batched PIR Query Builder for HDC Error-Tuned Queries
Implements repeat-aware PIR batching with median aggregation for uncertainty tuning
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import numpy as np
import torch

from genomevault.hypervector.error_handling import ErrorBudget
from genomevault.utils.logging import get_logger

from .pir_client import PIRClient, PIRQuery
from .query_builder import GenomicQuery, PIRQueryBuilder, QueryResult, QueryType

logger = get_logger(__name__)


class AggregationMethod(Enum):
    """Methods for aggregating repeat queries"""

    MEDIAN = "median"
    MEAN = "mean"
    MAJORITY_VOTE = "majority_vote"


@dataclass
class BatchedQuery:
    """A batch of queries for repeat execution"""

    base_query: GenomicQuery
    queries: List[PIRQuery]
    aggregation_method: AggregationMethod
    error_threshold: float
    budget: ErrorBudget


@dataclass
class BatchedQueryResult:
    """Result of a batched query execution"""

    query: BatchedQuery
    results: List[Any]
    aggregated_result: Any
    metadata: Dict[str, Any]
    pir_queries_used: int
    computation_time_ms: float
    median_error: float
    proof_metadata: Dict[str, Any]


class BatchedPIRQueryBuilder(PIRQueryBuilder):
    """
    Enhanced PIR query builder with repeat-aware batching
    Integrates with HDC error budget system for uncertainty tuning
    """

    def __init__(self, pir_client: PIRClient, index_mapping: Dict[str, Dict[str, int]]):
        super().__init__(pir_client, index_mapping)
        self.batch_cache: Dict[str, BatchedQueryResult] = {}
        self.batch_cache_size = 50

    def build_repeat_batch(self, budget: ErrorBudget, query: GenomicQuery) -> BatchedQuery:
        """
        Build a batch of k queries with different seeds for median aggregation

        Args:
            budget: Error budget configuration with repeat count
            query: Base genomic query to repeat

        Returns:
            BatchedQuery object with k PIR queries
        """
        queries = []
        base_params = query.parameters.copy()

        for seed_idx in range(budget.repeats):
            # Generate deterministic seed for this repeat
            repeat_seed = self._generate_repeat_seed(
                query.get_cache_key(), seed_idx, budget.dimension
            )

            # Build query variant with seed
            seeded_params = base_params.copy()
            seeded_params["_repeat_seed"] = repeat_seed
            seeded_params["_repeat_idx"] = seed_idx

            # Create PIR query for this repeat
            # The seed affects the masking/blinding of the query
            pir_query = self._build_seeded_pir_query(query, seeded_params, repeat_seed)
            queries.append(pir_query)

        return BatchedQuery(
            base_query=query,
            queries=queries,
            aggregation_method=AggregationMethod.MEDIAN,
            error_threshold=budget.epsilon,
            budget=budget,
        )

    def _generate_repeat_seed(self, base_key: str, repeat_idx: int, dimension: int) -> int:
        """Generate deterministic seed for a repeat query"""
        seed_data = f"{base_key}:{repeat_idx}:{dimension}"
        seed_hash = hashlib.sha256(seed_data.encode()).digest()
        return int.from_bytes(seed_hash[:4], byteorder="big")

    def _build_seeded_pir_query(
        self, base_query: GenomicQuery, seeded_params: Dict, seed: int
    ) -> PIRQuery:
        """Build a PIR query with deterministic seeding"""
        # This depends on the query type
        if base_query.query_type == QueryType.VARIANT_LOOKUP:
            return self._build_seeded_variant_query(seeded_params, seed)
        elif base_query.query_type == QueryType.REGION_SCAN:
            return self._build_seeded_region_query(seeded_params, seed)
        else:
            # For other types, create basic PIR query
            # In practice, each type would have its own seeding logic
            return PIRQuery(
                indices=[0],  # Placeholder
                seed=seed,
                metadata={"query_type": base_query.query_type.value},
            )

    def _build_seeded_variant_query(self, params: Dict, seed: int) -> PIRQuery:
        """Build seeded variant lookup query"""
        # Get variant key
        var_key = f"{params['chromosome']}:{params['position']}"
        if "ref_allele" in params and "alt_allele" in params:
            var_key += f":{params['ref_allele']}:{params['alt_allele']}"

        # Get database index
        if var_key not in self.index_mapping["variants"]:
            return PIRQuery(indices=[], seed=seed, metadata={"found": False})

        db_index = self.index_mapping["variants"][var_key]

        # Create PIR query with seed for masking
        return self.pir_client.create_query(db_index, seed=seed)

    def _build_seeded_region_query(self, params: Dict, seed: int) -> PIRQuery:
        """Build seeded region scan query"""
        # Find all indices in region
        indices = []
        for pos in range(params["start"], params["end"] + 1):
            var_key = f"{params['chromosome']}:{pos}"
            if var_key in self.index_mapping["positions"]:
                indices.extend(self.index_mapping["positions"][var_key])

        # Create batched PIR query with seed
        return PIRQuery(
            indices=indices,
            seed=seed,
            metadata={
                "region_size": params["end"] - params["start"],
                "query_type": "region_scan",
            },
        )

    async def execute_batched_query(self, batched_query: BatchedQuery) -> BatchedQueryResult:
        """
        Execute a batch of repeat queries and aggregate results

        Args:
            batched_query: Batch of queries to execute

        Returns:
            Aggregated result with confidence metrics
        """
        start_time = time.time()

        # Check cache
        cache_key = f"{batched_query.base_query.get_cache_key()}:k{batched_query.budget.repeats}"
        if cache_key in self.batch_cache:
            logger.info("Batched query result found in cache")
            return self.batch_cache[cache_key]

        # Execute all repeat queries
        results = []
        tasks = []

        for pir_query in batched_query.queries:
            task = self._execute_single_repeat(pir_query)
            tasks.append(task)

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        # Aggregate results
        aggregated_result, median_error = self._aggregate_results(
            results, batched_query.aggregation_method
        )

        # Generate proof metadata
        proof_metadata = self._generate_proof_metadata(
            batched_query, results, aggregated_result, median_error
        )

        computation_time = (time.time() - start_time) * 1000

        result = BatchedQueryResult(
            query=batched_query,
            results=results,
            aggregated_result=aggregated_result,
            metadata={
                "repeats_executed": len(results),
                "aggregation_method": batched_query.aggregation_method.value,
                "error_within_bound": median_error <= batched_query.error_threshold,
            },
            pir_queries_used=len(batched_query.queries),
            computation_time_ms=computation_time,
            median_error=median_error,
            proof_metadata=proof_metadata,
        )

        # Cache result
        self._add_to_batch_cache(cache_key, result)

        return result

    async def execute_streaming_batch(
        self, batched_query: BatchedQuery
    ) -> AsyncIterator[Tuple[int, Any]]:
        """
        Stream results as they complete for real-time progress updates

        Args:
            batched_query: Batch of queries to execute

        Yields:
            Tuples of (repeat_idx, result) as queries complete
        """
        tasks = []
        for idx, pir_query in enumerate(batched_query.queries):
            task = asyncio.create_task(self._execute_indexed_repeat(idx, pir_query))
            tasks.append(task)

        # Yield results as they complete
        accumulated_results = []
        for coro in asyncio.as_completed(tasks):
            idx, result = await coro
            accumulated_results.append(result)
            yield (idx, result)

            # Check for early termination if we have enough good results
            if self._can_terminate_early(batched_query, accumulated_results):
                logger.info(f"Early termination after {len(accumulated_results)} results")
                # Cancel remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                break

    async def _execute_single_repeat(self, pir_query: PIRQuery) -> Any:
        """Execute a single repeat query"""
        if not pir_query.indices:
            return None

        # Execute PIR query
        result_data = await self.pir_client.execute_query(pir_query)

        # Decode response based on query type
        query_type = pir_query.metadata.get("query_type", "genomic")
        decoded = self.pir_client.decode_response(result_data, query_type)

        return decoded

    async def _execute_indexed_repeat(self, idx: int, pir_query: PIRQuery) -> Tuple[int, Any]:
        """Execute a repeat query with its index"""
        result = await self._execute_single_repeat(pir_query)
        return (idx, result)

    def _aggregate_results(
        self, results: List[Any], method: AggregationMethod
    ) -> Tuple[Any, float]:
        """
        Aggregate multiple query results using specified method

        Returns:
            Tuple of (aggregated_result, median_error)
        """
        # Filter out None results
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            return None, float("inf")

        if method == AggregationMethod.MEDIAN:
            # For numeric results
            if isinstance(valid_results[0], (int, float)):
                values = np.array(valid_results)
                median_val = np.median(values)
                median_error = np.median(np.abs(values - median_val))
                return float(median_val), float(median_error)

            # For vector results (e.g., hypervectors)
            elif torch.is_tensor(valid_results[0]):
                stacked = torch.stack(valid_results)
                median_vec = torch.median(stacked, dim=0)[0]
                errors = torch.abs(stacked - median_vec)
                median_error = torch.median(errors).item()
                return median_vec, median_error

            # For complex results (dicts, lists)
            else:
                # For now, return the most common result
                from collections import Counter

                result_strs = [str(r) for r in valid_results]
                most_common = Counter(result_strs).most_common(1)[0][0]
                # Find the original result
                for r in valid_results:
                    if str(r) == most_common:
                        return r, 0.0

        elif method == AggregationMethod.MEAN:
            if isinstance(valid_results[0], (int, float)):
                mean_val = np.mean(valid_results)
                std_dev = np.std(valid_results)
                return float(mean_val), float(std_dev)

        # Default: return first valid result
        return valid_results[0], 0.0

    def _can_terminate_early(
        self, batched_query: BatchedQuery, accumulated_results: List[Any]
    ) -> bool:
        """
        Check if we can terminate early based on accumulated results

        Early termination criteria:
        - Have at least 60% of planned repeats
        - Median error is well below threshold
        - Results are converging
        """
        if len(accumulated_results) < int(0.6 * batched_query.budget.repeats):
            return False

        # Check convergence
        if len(accumulated_results) >= 3:
            _, current_error = self._aggregate_results(
                accumulated_results, batched_query.aggregation_method
            )

            # If error is already 50% below threshold, we can stop
            if current_error < 0.5 * batched_query.error_threshold:
                return True

        return False

    def _generate_proof_metadata(
        self,
        batched_query: BatchedQuery,
        results: List[Any],
        aggregated_result: Any,
        median_error: float,
    ) -> Dict[str, Any]:
        """Generate metadata for proof generation"""
        return {
            "query_type": batched_query.base_query.query_type.value,
            "repeats_executed": len(results),
            "dimension": batched_query.budget.dimension,
            "ecc_enabled": batched_query.budget.ecc_enabled,
            "epsilon": batched_query.budget.epsilon,
            "delta_exp": batched_query.budget.delta_exp,
            "median_error": median_error,
            "error_within_bound": median_error <= batched_query.error_threshold,
            "aggregation_method": batched_query.aggregation_method.value,
            "timestamp": time.time(),
        }

    def _add_to_batch_cache(self, key: str, result: BatchedQueryResult):
        """Add result to batch cache with LRU eviction"""
        if len(self.batch_cache) >= self.batch_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.batch_cache))
            del self.batch_cache[oldest_key]

        self.batch_cache[key] = result

    async def query_with_error_budget(
        self, query: GenomicQuery, budget: ErrorBudget
    ) -> BatchedQueryResult:
        """
        High-level API: Execute query with specified error budget

        Args:
            query: Genomic query to execute
            budget: Error budget configuration

        Returns:
            Batched query result with confidence metrics
        """
        # Build repeat batch
        batched_query = self.build_repeat_batch(budget, query)

        # Execute batch
        result = await self.execute_batched_query(batched_query)

        logger.info(
            f"Executed batched query: {budget.repeats} repeats, "
            f"median error: {result.median_error:.4f}, "
            f"within bound: {result.metadata['error_within_bound']}"
        )

        return result

    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get statistics about batched queries"""
        stats = super().get_query_statistics()

        # Add batch-specific stats
        batch_stats = {
            "batch_cache_size": len(self.batch_cache),
            "avg_repeats": 0,
            "avg_median_error": 0,
            "error_bound_success_rate": 0,
        }

        if self.batch_cache:
            total_repeats = sum(r.query.budget.repeats for r in self.batch_cache.values())
            total_errors = sum(r.median_error for r in self.batch_cache.values())
            successes = sum(
                1 for r in self.batch_cache.values() if r.metadata["error_within_bound"]
            )

            batch_stats["avg_repeats"] = total_repeats / len(self.batch_cache)
            batch_stats["avg_median_error"] = total_errors / len(self.batch_cache)
            batch_stats["error_bound_success_rate"] = successes / len(self.batch_cache)

        stats["batch_statistics"] = batch_stats
        return stats


# Module exports
__all__ = [
    "BatchedPIRQueryBuilder",
    "BatchedQuery",
    "BatchedQueryResult",
    "AggregationMethod",
]
