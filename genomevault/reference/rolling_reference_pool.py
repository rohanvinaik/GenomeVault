"""
Rolling Reference Pool with Entropy-Based Rotation

Implements dynamic reference pool management to prevent information leakage over time.

Key Concepts:
- Each query leaks ~7 bits of information
- Initial entropy = log2(C(N,k)) + 260 bits (pool selection + alignment randomization)
- Update pool when remaining entropy < 128 bits threshold
- Multiple update strategies: entropy-based, query-count, time-based
- Forward secrecy: old pool compromise doesn't affect new pool

Usage:
    from genomevault.reference import RollingReferencePool

    # Initialize rolling pool
    pool = RollingReferencePool(
        initial_pool=[ref1_vcf, ref2_vcf, ref3_vcf],
        genome_database=Path("data/genome_pool/"),
        k_min=3,
        k_max=10,
        entropy_threshold=128.0,
        update_strategy="entropy"
    )

    # Process queries
    for query in user_queries:
        # Align query to current pool
        result = aligner.align_query_to_pool(query, pool.get_current_pool())

        # Record query and check for update
        pool.record_query(query.id, information_leakage=7.0)

        # Pool automatically updates when entropy < 128 bits
"""

import logging
import random
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np

# Try to import scipy for combinatorial calculations
try:
    from scipy.special import comb
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    import math
    def comb(n, k):
        """Fallback binomial coefficient calculation."""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)  # Optimization
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result

logger = logging.getLogger(__name__)


class UpdateStrategy(Enum):
    """Strategy for determining when to update reference pool."""
    ENTROPY = "entropy"  # Update when entropy drops below threshold
    QUERY_COUNT = "query_count"  # Update after fixed number of queries
    TIME = "time"  # Update after fixed time period
    HYBRID = "hybrid"  # Combination of entropy + time


class PoolUpdateMethod(Enum):
    """Method for updating reference pool."""
    ADD_NEW = "add_new"  # Add new genome (increases k)
    REPLACE_OLDEST = "replace_oldest"  # LRU eviction (maintains k)
    REPLACE_RANDOM = "replace_random"  # Random eviction (maintains k)
    SHUFFLE = "shuffle"  # Reorder existing pool
    FULL_REFRESH = "full_refresh"  # Replace entire pool


@dataclass
class QueryRecord:
    """Record of a query processed against the reference pool."""
    query_id: str
    timestamp: datetime
    information_leakage: float  # Bits of information leaked
    pool_version: int  # Which version of pool was used
    pool_size: int  # k value at time of query

    def age_days(self) -> float:
        """Calculate age of this query in days."""
        return (datetime.now() - self.timestamp).total_seconds() / 86400


@dataclass
class GenomeReference:
    """Represents a reference genome in the pool."""
    path: Path
    genome_id: str
    added_timestamp: datetime
    last_used: datetime
    query_count: int = 0
    metadata: Dict = field(default_factory=dict)

    def update_usage(self):
        """Update usage statistics."""
        self.last_used = datetime.now()
        self.query_count += 1

    def age_days(self) -> float:
        """Calculate age since addition."""
        return (datetime.now() - self.added_timestamp).total_seconds() / 86400


@dataclass
class PoolStatistics:
    """Statistics for the reference pool."""
    pool_version: int
    current_k: int
    initial_entropy: float
    remaining_entropy: float
    total_queries: int
    total_information_leaked: float
    queries_until_update: int
    last_update: datetime
    update_history: List[Dict] = field(default_factory=list)


class RollingReferencePool:
    """
    Dynamic reference pool with automatic rotation based on entropy decay.

    Features:
    - Tracks information leakage per query (~7 bits)
    - Updates pool when entropy drops below threshold (128 bits)
    - Multiple update strategies: time-based, query-count, entropy-based
    - Forward secrecy: old pool compromise doesn't affect new pool
    """

    # Information leakage constants
    DEFAULT_QUERY_LEAKAGE = 7.0  # bits per query
    ALIGNMENT_RANDOMIZATION_ENTROPY = 260.0  # bits from user randomization

    def __init__(
        self,
        initial_pool: List[Path],
        genome_database: Optional[Path] = None,
        k_min: int = 3,
        k_max: int = 10,
        entropy_threshold: float = 128.0,
        update_strategy: str = "entropy",
        update_method: str = "add_new",
        auto_update: bool = True
    ):
        """
        Initialize rolling reference pool.

        Args:
            initial_pool: Initial k reference VCF/FASTQ paths
            genome_database: Directory containing genome pool for rotation
            k_min: Minimum anonymity set size (PoC: 3, production: 10)
            k_max: Maximum pool size
            entropy_threshold: Update when entropy drops below this (bits)
            update_strategy: "entropy", "query_count", "time", or "hybrid"
            update_method: "add_new", "replace_oldest", "shuffle", etc.
            auto_update: Automatically update pool when threshold reached
        """
        self.k_min = k_min
        self.k_max = k_max
        self.entropy_threshold = entropy_threshold
        self.auto_update = auto_update

        # Validate strategy and method
        try:
            self.strategy = UpdateStrategy(update_strategy)
        except ValueError:
            raise ValueError(f"Unknown strategy: {update_strategy}. Must be one of {[s.value for s in UpdateStrategy]}")

        try:
            self.update_method = PoolUpdateMethod(update_method)
        except ValueError:
            raise ValueError(f"Unknown method: {update_method}. Must be one of {[m.value for m in PoolUpdateMethod]}")

        # Initialize pool
        self.pool: List[GenomeReference] = []
        now = datetime.now()
        for i, path in enumerate(initial_pool):
            genome = GenomeReference(
                path=path,
                genome_id=f"genome_{i}_{path.stem}",
                added_timestamp=now,
                last_used=now
            )
            self.pool.append(genome)

        if len(self.pool) < self.k_min:
            raise ValueError(f"Initial pool size {len(self.pool)} < k_min {self.k_min}")

        # Genome database for rotation
        self.genome_db_path = genome_database
        self.available_genomes: List[Path] = []
        if genome_database:
            self.available_genomes = self._load_genome_database(genome_database)

        # Query tracking
        self.query_history: List[QueryRecord] = []
        self.pool_version = 1
        self.creation_timestamp = now
        self.last_update_timestamp = now

        # Calculate initial entropy
        self.initial_entropy = self._compute_initial_entropy()

        logger.info("="*70)
        logger.info("ROLLING REFERENCE POOL INITIALIZED")
        logger.info("="*70)
        logger.info(f"  Initial pool size (k): {len(self.pool)}")
        logger.info(f"  k_min: {self.k_min}, k_max: {self.k_max}")
        logger.info(f"  Available genomes: {len(self.available_genomes)}")
        logger.info(f"  Initial entropy: {self.initial_entropy:.1f} bits")
        logger.info(f"  Entropy threshold: {self.entropy_threshold:.1f} bits")
        logger.info(f"  Update strategy: {self.strategy.value}")
        logger.info(f"  Update method: {self.update_method.value}")
        logger.info(f"  Auto-update: {self.auto_update}")
        logger.info("="*70)

    def _load_genome_database(self, database_path: Path) -> List[Path]:
        """
        Load available genomes from database directory.

        Args:
            database_path: Path to directory containing genomes

        Returns:
            List of genome file paths
        """
        if not database_path.exists():
            logger.warning(f"Genome database not found: {database_path}")
            return []

        # Find VCF and FASTQ files
        genomes = []
        for pattern in ['*.vcf', '*.vcf.gz', '*.fastq', '*.fastq.gz', '*.fq', '*.fq.gz']:
            genomes.extend(database_path.glob(pattern))

        # Exclude files already in pool
        pool_paths = {g.path for g in self.pool}
        genomes = [g for g in genomes if g not in pool_paths]

        logger.info(f"Loaded {len(genomes)} genomes from database: {database_path}")
        return sorted(genomes)

    def _compute_initial_entropy(self) -> float:
        """
        Calculate initial pool entropy.

        H(pool) = log2(C(N, k)) + 260

        Where:
        - C(N, k) = pool selection entropy (binomial coefficient)
        - 260 = alignment randomization entropy

        Returns:
            Total initial entropy in bits
        """
        N = len(self.available_genomes) + len(self.pool)
        k = len(self.pool)

        if N < k:
            # Not enough genomes for meaningful selection entropy
            pool_selection_entropy = 0.0
        else:
            # Calculate binomial coefficient entropy
            try:
                pool_selection_entropy = np.log2(comb(N, k, exact=False))
            except (ValueError, OverflowError):
                # Fallback to Stirling's approximation for large N, k
                pool_selection_entropy = (
                    N * np.log2(N) -
                    k * np.log2(k) -
                    (N - k) * np.log2(N - k)
                )

        # Total entropy = pool selection + alignment randomization
        total = pool_selection_entropy + self.ALIGNMENT_RANDOMIZATION_ENTROPY

        logger.debug(f"Initial pool entropy: {total:.1f} bits")
        logger.debug(f"  Pool selection: {pool_selection_entropy:.1f} bits (N={N}, k={k})")
        logger.debug(f"  Alignment randomization: {self.ALIGNMENT_RANDOMIZATION_ENTROPY:.1f} bits")

        return total

    def compute_remaining_entropy(self) -> float:
        """
        Compute remaining entropy after query history.

        H(pool | queries) = H(pool) - I(pool; queries)

        Where I(pool; queries) is the mutual information (information leaked).

        Returns:
            Remaining entropy in bits
        """
        # Sum information leakage from all queries
        leaked_info = sum(
            query.information_leakage
            for query in self.query_history
        )

        remaining = self.initial_entropy - leaked_info

        # Ensure non-negative
        remaining = max(0.0, remaining)

        logger.debug(f"Remaining entropy: {remaining:.1f} bits")
        logger.debug(f"  Initial: {self.initial_entropy:.1f}")
        logger.debug(f"  Leaked: {leaked_info:.1f}")
        logger.debug(f"  Queries: {len(self.query_history)}")

        return remaining

    def compute_queries_until_update(self) -> int:
        """
        Calculate estimated number of queries until pool update.

        Returns:
            Number of queries until entropy drops below threshold
        """
        remaining = self.compute_remaining_entropy()
        buffer = remaining - self.entropy_threshold

        if buffer <= 0:
            return 0

        queries = int(buffer / self.DEFAULT_QUERY_LEAKAGE)
        return max(0, queries)

    def should_update_pool(self) -> Tuple[bool, str]:
        """
        Determine if pool needs updating based on strategy.

        Returns:
            Tuple of (should_update, reason)
        """
        if self.strategy == UpdateStrategy.ENTROPY:
            remaining = self.compute_remaining_entropy()
            if remaining < self.entropy_threshold:
                return True, f"Entropy {remaining:.1f} < threshold {self.entropy_threshold:.1f}"

        elif self.strategy == UpdateStrategy.QUERY_COUNT:
            # Calculate threshold based on entropy
            threshold_queries = int(
                (self.initial_entropy - self.entropy_threshold) / self.DEFAULT_QUERY_LEAKAGE
            )
            if len(self.query_history) >= threshold_queries:
                return True, f"Query count {len(self.query_history)} >= threshold {threshold_queries}"

        elif self.strategy == UpdateStrategy.TIME:
            # Update every 30 days
            days_elapsed = (datetime.now() - self.last_update_timestamp).days
            if days_elapsed >= 30:
                return True, f"Time elapsed {days_elapsed} days >= 30"

        elif self.strategy == UpdateStrategy.HYBRID:
            # Check both entropy and time
            remaining = self.compute_remaining_entropy()
            days_elapsed = (datetime.now() - self.last_update_timestamp).days

            if remaining < self.entropy_threshold:
                return True, f"Entropy {remaining:.1f} < threshold {self.entropy_threshold:.1f}"
            if days_elapsed >= 30:
                return True, f"Time elapsed {days_elapsed} days >= 30"

        return False, "No update needed"

    def _select_new_genome(self) -> GenomeReference:
        """
        Select a new genome from the database.

        Returns:
            GenomeReference for newly selected genome

        Raises:
            RuntimeError: If no genomes available
        """
        # Exclude genomes already in pool
        pool_paths = {g.path for g in self.pool}
        available = [g for g in self.available_genomes if g not in pool_paths]

        if not available:
            raise RuntimeError("No genomes available for rotation")

        # Randomly select
        selected_path = random.choice(available)

        genome = GenomeReference(
            path=selected_path,
            genome_id=f"genome_{self.pool_version}_{selected_path.stem}",
            added_timestamp=datetime.now(),
            last_used=datetime.now()
        )

        logger.info(f"Selected new genome: {genome.genome_id} ({selected_path.name})")
        return genome

    def update_pool(
        self,
        method: Optional[str] = None,
        force: bool = False
    ) -> Dict:
        """
        Execute pool update.

        Args:
            method: Override default update method
            force: Force update even if not needed

        Returns:
            Dict with update statistics

        Raises:
            ValueError: If method is invalid
        """
        should_update, reason = self.should_update_pool()

        if not should_update and not force:
            logger.info("Pool update not needed")
            return {'updated': False, 'reason': reason}

        logger.info("="*70)
        logger.info("UPDATING REFERENCE POOL")
        logger.info("="*70)
        logger.info(f"  Reason: {reason}")

        # Use provided method or default
        if method:
            try:
                update_method = PoolUpdateMethod(method)
            except ValueError:
                raise ValueError(f"Unknown method: {method}")
        else:
            update_method = self.update_method

        old_pool_size = len(self.pool)
        old_entropy = self.compute_remaining_entropy()

        # Execute update strategy
        if update_method == PoolUpdateMethod.ADD_NEW:
            if len(self.pool) < self.k_max:
                new_genome = self._select_new_genome()
                self.pool.append(new_genome)
                logger.info(f"  Added new genome: {new_genome.genome_id}")
                logger.info(f"  Pool size: {len(self.pool)} (was {old_pool_size})")
            else:
                logger.warning(f"  Pool at maximum size ({self.k_max}), cannot add")

        elif update_method == PoolUpdateMethod.REPLACE_OLDEST:
            # LRU eviction
            oldest = min(self.pool, key=lambda g: g.last_used)
            self.pool.remove(oldest)
            logger.info(f"  Removed oldest: {oldest.genome_id} (last used: {oldest.last_used})")

            new_genome = self._select_new_genome()
            self.pool.append(new_genome)
            logger.info(f"  Added: {new_genome.genome_id}")

        elif update_method == PoolUpdateMethod.REPLACE_RANDOM:
            # Random eviction
            removed = random.choice(self.pool)
            self.pool.remove(removed)
            logger.info(f"  Removed random: {removed.genome_id}")

            new_genome = self._select_new_genome()
            self.pool.append(new_genome)
            logger.info(f"  Added: {new_genome.genome_id}")

        elif update_method == PoolUpdateMethod.SHUFFLE:
            random.shuffle(self.pool)
            logger.info(f"  Shuffled pool order")

        elif update_method == PoolUpdateMethod.FULL_REFRESH:
            # Replace entire pool
            k = len(self.pool)
            self.pool = []
            for _ in range(k):
                new_genome = self._select_new_genome()
                self.pool.append(new_genome)
            logger.info(f"  Replaced entire pool (k={k})")

        # Reset query history (forward secrecy)
        old_query_count = len(self.query_history)
        self.query_history = []
        self.pool_version += 1
        self.last_update_timestamp = datetime.now()

        # Recalculate entropy
        self.initial_entropy = self._compute_initial_entropy()
        new_entropy = self.compute_remaining_entropy()

        logger.info(f"  Pool version: {self.pool_version}")
        logger.info(f"  Queries cleared: {old_query_count}")
        logger.info(f"  Old entropy: {old_entropy:.1f} bits")
        logger.info(f"  New entropy: {new_entropy:.1f} bits")
        logger.info(f"  Entropy gain: {new_entropy - old_entropy:.1f} bits")
        logger.info("="*70)

        # Record update in history
        update_record = {
            'timestamp': self.last_update_timestamp.isoformat(),
            'version': self.pool_version,
            'method': update_method.value,
            'reason': reason,
            'old_pool_size': old_pool_size,
            'new_pool_size': len(self.pool),
            'old_entropy': old_entropy,
            'new_entropy': new_entropy,
            'queries_processed': old_query_count
        }

        return {
            'updated': True,
            'reason': reason,
            'method': update_method.value,
            'pool_version': self.pool_version,
            'pool_size': len(self.pool),
            'entropy': new_entropy,
            'update_record': update_record
        }

    def record_query(
        self,
        query_id: str,
        information_leakage: Optional[float] = None
    ) -> bool:
        """
        Record a query and its information leakage.

        Args:
            query_id: Unique query identifier
            information_leakage: Bits leaked (default: 7.0)

        Returns:
            True if pool was updated, False otherwise
        """
        if information_leakage is None:
            information_leakage = self.DEFAULT_QUERY_LEAKAGE

        # Create query record
        record = QueryRecord(
            query_id=query_id,
            timestamp=datetime.now(),
            information_leakage=information_leakage,
            pool_version=self.pool_version,
            pool_size=len(self.pool)
        )
        self.query_history.append(record)

        # Update usage statistics for all pool members
        for genome in self.pool:
            genome.update_usage()

        logger.debug(f"Recorded query: {query_id} (leaked: {information_leakage:.1f} bits)")

        # Check if update needed
        should_update, reason = self.should_update_pool()
        if should_update and self.auto_update:
            logger.info(f"Entropy threshold reached. Triggering pool update...")
            logger.info(f"  Reason: {reason}")
            self.update_pool()
            return True

        return False

    def get_current_pool(self) -> List[Path]:
        """
        Get current reference pool paths.

        Returns:
            List of paths to reference genomes
        """
        return [genome.path for genome in self.pool]

    def get_statistics(self) -> PoolStatistics:
        """
        Get current pool statistics.

        Returns:
            PoolStatistics object with comprehensive metrics
        """
        remaining_entropy = self.compute_remaining_entropy()
        total_leaked = sum(q.information_leakage for q in self.query_history)
        queries_until_update = self.compute_queries_until_update()

        stats = PoolStatistics(
            pool_version=self.pool_version,
            current_k=len(self.pool),
            initial_entropy=self.initial_entropy,
            remaining_entropy=remaining_entropy,
            total_queries=len(self.query_history),
            total_information_leaked=total_leaked,
            queries_until_update=queries_until_update,
            last_update=self.last_update_timestamp
        )

        return stats

    def print_statistics(self):
        """Print comprehensive pool statistics."""
        stats = self.get_statistics()

        logger.info("="*70)
        logger.info("ROLLING REFERENCE POOL STATISTICS")
        logger.info("="*70)
        logger.info(f"Pool Information:")
        logger.info(f"  Version: {stats.pool_version}")
        logger.info(f"  Current k: {stats.current_k}")
        logger.info(f"  Available genomes: {len(self.available_genomes)}")
        logger.info("")
        logger.info(f"Entropy:")
        logger.info(f"  Initial: {stats.initial_entropy:.1f} bits")
        logger.info(f"  Remaining: {stats.remaining_entropy:.1f} bits")
        logger.info(f"  Leaked: {stats.total_information_leaked:.1f} bits")
        logger.info(f"  Threshold: {self.entropy_threshold:.1f} bits")
        logger.info("")
        logger.info(f"Query History:")
        logger.info(f"  Total queries: {stats.total_queries}")
        logger.info(f"  Queries until update: {stats.queries_until_update}")
        logger.info(f"  Last update: {stats.last_update}")
        logger.info("")
        logger.info(f"Update Strategy:")
        logger.info(f"  Strategy: {self.strategy.value}")
        logger.info(f"  Method: {self.update_method.value}")
        logger.info(f"  Auto-update: {self.auto_update}")
        logger.info("="*70)

    def save_state(self, output_path: Path):
        """
        Save pool state to JSON file.

        Args:
            output_path: Path to save state
        """
        state = {
            'pool_version': self.pool_version,
            'creation_timestamp': self.creation_timestamp.isoformat(),
            'last_update_timestamp': self.last_update_timestamp.isoformat(),
            'k_min': self.k_min,
            'k_max': self.k_max,
            'entropy_threshold': self.entropy_threshold,
            'strategy': self.strategy.value,
            'update_method': self.update_method.value,
            'auto_update': self.auto_update,
            'initial_entropy': self.initial_entropy,
            'pool': [
                {
                    'genome_id': g.genome_id,
                    'path': str(g.path),
                    'added_timestamp': g.added_timestamp.isoformat(),
                    'last_used': g.last_used.isoformat(),
                    'query_count': g.query_count
                }
                for g in self.pool
            ],
            'query_history': [
                {
                    'query_id': q.query_id,
                    'timestamp': q.timestamp.isoformat(),
                    'information_leakage': q.information_leakage,
                    'pool_version': q.pool_version,
                    'pool_size': q.pool_size
                }
                for q in self.query_history
            ],
            'statistics': asdict(self.get_statistics())
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"Saved pool state to: {output_path}")

    @classmethod
    def load_state(cls, state_path: Path, genome_database: Path) -> 'RollingReferencePool':
        """
        Load pool state from JSON file.

        Args:
            state_path: Path to saved state
            genome_database: Path to genome database

        Returns:
            RollingReferencePool instance
        """
        with open(state_path, 'r') as f:
            state = json.load(f)

        # Reconstruct pool
        initial_pool = [Path(g['path']) for g in state['pool']]

        pool = cls(
            initial_pool=initial_pool,
            genome_database=genome_database,
            k_min=state['k_min'],
            k_max=state['k_max'],
            entropy_threshold=state['entropy_threshold'],
            update_strategy=state['strategy'],
            update_method=state['update_method'],
            auto_update=state['auto_update']
        )

        # Restore state
        pool.pool_version = state['pool_version']
        pool.creation_timestamp = datetime.fromisoformat(state['creation_timestamp'])
        pool.last_update_timestamp = datetime.fromisoformat(state['last_update_timestamp'])

        # Restore query history
        pool.query_history = [
            QueryRecord(
                query_id=q['query_id'],
                timestamp=datetime.fromisoformat(q['timestamp']),
                information_leakage=q['information_leakage'],
                pool_version=q['pool_version'],
                pool_size=q['pool_size']
            )
            for q in state['query_history']
        ]

        logger.info(f"Loaded pool state from: {state_path}")
        logger.info(f"  Version: {pool.pool_version}")
        logger.info(f"  Queries: {len(pool.query_history)}")

        return pool
