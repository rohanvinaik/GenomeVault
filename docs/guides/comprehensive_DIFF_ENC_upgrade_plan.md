# GenomeVault Differential Encoding: Comprehensive Upgrade Plan

## Executive Summary

This plan upgrades the differential encoding system from **21.67ms → <5ms** encoding time and **230K → 1M+ variants/sec** throughput by implementing:
1. Fast hash functions and caching (50-100× speedup on hot paths)
2. NumPy-based vectorized operations (2-5× speedup)
3. Interval tree indexing for position matching (10-50× speedup)
4. Reference genome caching with FM-index style approach
5. Parallel chunk processing across CPU cores (4-16× speedup)
6. GPU-accelerated batch feature computation (optional 10-50× speedup)

---

## Phase 1: Core Algorithm Optimizations (Days 1-3)

### 1.1 Fast K-mer Hashing (`sequence_alignment.py`)

**CURRENT BOTTLENECK:**
```python
def _hash_kmer(self, kmer: str) -> int:
    return int.from_bytes(
        hashlib.sha256(kmer.encode()).digest()[:8],
        byteorder='big'
    )
```

**PROBLEM:** SHA-256 is 100× slower than needed for non-cryptographic hashing

**UPGRADE:**
```python
# NEW: Fast hash function
def _hash_kmer(self, kmer: str) -> int:
    """Fast non-cryptographic hash using Python's builtin hash().
    
    Alternative: Use xxhash library for even better performance:
        import xxhash
        return xxhash.xxh64(kmer.encode()).intdigest()
    
    Performance: 50-100× faster than SHA-256
    """
    # Python's builtin hash is fast and well-distributed
    return hash(kmer) & 0x7FFFFFFFFFFFFFFF  # Keep positive

# OPTIONAL: For production, use xxhash
# pip install xxhash
# import xxhash
# 
# def _hash_kmer_xxhash(self, kmer: str) -> int:
#     return xxhash.xxh64(kmer.encode()).intdigest()
```

**IMPLEMENTATION NOTES:**
- Replace `_hash_kmer` in `KmerIndex` class
- Add benchmarking to verify speedup
- Consider `xxhash` for production (requires pip install)
- **Expected speedup: 50-100×** on k-mer operations

---

### 1.2 K-mer Extraction Caching (`sequence_alignment.py`)

**CURRENT BOTTLENECK:**
```python
def _extract_kmers(self, chromosome, position, ref, alt) -> Set[str]:
    kmers = set()
    if len(ref) >= self.k:
        for i in range(len(ref) - self.k + 1):
            kmers.add(ref[i:i+self.k])
    # Same for alt...
```

**PROBLEM:** Recomputes k-mers for same sequences repeatedly

**UPGRADE:**
```python
from functools import lru_cache

class KmerIndex:
    def __init__(self, k: int = 31, cache_size: int = 10000):
        self.k = k
        self.kmer_map: Dict[int, Set[Tuple[str, str, int]]] = defaultdict(set)
        self.reference_ids: Set[str] = set()
        self._cache_size = cache_size
    
    @lru_cache(maxsize=10000)
    def _extract_kmers_from_sequence(self, sequence: str) -> frozenset:
        """Cache k-mer extraction results.
        
        Uses frozenset for hashability in LRU cache.
        Pre-allocates set size for efficiency.
        """
        if len(sequence) < self.k:
            return frozenset()
        
        # Use list comprehension (faster than loop + add)
        kmers = [sequence[i:i+self.k] 
                 for i in range(len(sequence) - self.k + 1)]
        return frozenset(kmers)
    
    def _extract_kmers(self, chromosome, position, ref, alt) -> Set[str]:
        """Extract k-mers using cached results."""
        kmers = set()
        
        # Use cached function for both ref and alt
        if len(ref) >= self.k:
            kmers.update(self._extract_kmers_from_sequence(ref))
        if len(alt) >= self.k:
            kmers.update(self._extract_kmers_from_sequence(alt))
        
        return kmers
```

**IMPLEMENTATION NOTES:**
- Add `_extract_kmers_from_sequence` with `@lru_cache`
- Modify existing `_extract_kmers` to use cached version
- Tune `cache_size` based on typical variant patterns
- **Expected speedup: 2-3×** for repeated sequences

---

### 1.3 Interval Tree for Position Matching (`sequence_alignment.py`)

**CURRENT BOTTLENECK:**
```python
def _match_variants(self, query_variant, reference_variants, tolerance):
    for ref_var in reference_variants:  # O(n) linear scan
        if not self._fuzzy_match_position(...):
            continue
        # Check alleles...
```

**PROBLEM:** O(n²) complexity for fuzzy position matching

**UPGRADE:**
```python
# Add to imports
from intervaltree import IntervalTree
from typing import Optional

class VariantAligner:
    def __init__(
        self,
        snp_weight: float = 1.0,
        indel_weight: float = 0.8,
        genotype_weight: float = 0.3,
        position_tolerance: int = 10,
    ):
        # ... existing init ...
        self.position_tolerance = position_tolerance
        
        # NEW: Interval tree index (built per reference section)
        self._position_index: Optional[IntervalTree] = None
        self._indexed_variants: List[Variant] = []
    
    def _build_position_index(self, variants: List[Variant]) -> None:
        """Build interval tree for O(log n) position queries.
        
        Each variant is indexed in a range [pos - tolerance, pos + tolerance]
        to support fuzzy matching.
        """
        self._position_index = IntervalTree()
        self._indexed_variants = variants
        
        for i, var in enumerate(variants):
            # Store variant index as interval data
            self._position_index.addi(
                var.position - self.position_tolerance,
                var.position + self.position_tolerance + 1,  # +1 for exclusive end
                i  # Store index into variants list
            )
    
    def _match_variants_fast(
        self,
        query_variant: Variant,
        reference_variants: List[Variant],
        tolerance: int
    ) -> Optional[Variant]:
        """Fast variant matching using interval tree.
        
        Complexity: O(log n + k) where k = number of overlapping intervals
        vs O(n) for linear scan.
        """
        # Build index if not already built
        if self._position_index is None or self._indexed_variants != reference_variants:
            self._build_position_index(reference_variants)
        
        # Query interval tree for candidates
        query_start = query_variant.position - tolerance
        query_end = query_variant.position + tolerance + 1
        
        overlapping = self._position_index.overlap(query_start, query_end)
        
        # Check candidates for exact match
        for interval in overlapping:
            ref_var = reference_variants[interval.data]
            
            # Exact allele match
            if (query_variant.ref == ref_var.ref and
                query_variant.alt == ref_var.alt):
                return ref_var
            
            # For indels, check if same length change
            if self._is_indel(query_variant) and self._is_indel(ref_var):
                query_len_change = len(query_variant.alt) - len(query_variant.ref)
                ref_len_change = len(ref_var.alt) - len(ref_var.ref)
                
                if query_len_change == ref_len_change:
                    return ref_var
        
        return None
    
    def align_section(
        self,
        query_section: GenomeSection,
        reference_section: GenomeSection,
        reference_id: str
    ) -> AlignmentScore:
        """Compute alignment score using fast position matching."""
        score = AlignmentScore(reference_id=reference_id)
        
        # Build index once for entire section
        self._build_position_index(reference_section.variants)
        
        # Separate variants by type
        query_snps = [v for v in query_section.variants if self._is_snp(v)]
        query_indels = [v for v in query_section.variants if self._is_indel(v)]
        
        # Score SNPs with exact position matching
        for query_var in query_snps:
            match = self._match_variants_fast(
                query_var,
                reference_section.variants,
                tolerance=0  # Exact position for SNPs
            )
            
            if match:
                score.snp_matches += 1
                if query_var.genotype != match.genotype:
                    score.snp_mismatches += 1
            else:
                score.new_variants += 1
        
        # Score indels with position tolerance
        for query_var in query_indels:
            match = self._match_variants_fast(
                query_var,
                reference_section.variants,
                tolerance=self.position_tolerance
            )
            
            if match:
                score.indel_matches += 1
                if query_var.genotype != match.genotype:
                    score.indel_mismatches += 1
            else:
                score.new_variants += 1
        
        # Rest of scoring logic unchanged...
        # [compute rates, overall_score, confidence as before]
        
        return score
```

**DEPENDENCIES:**
```bash
pip install intervaltree
```

**IMPLEMENTATION NOTES:**
- Add `intervaltree` to requirements.txt
- Replace `_match_variants` with `_match_variants_fast`
- Build index once per reference section (not per query)
- **Expected speedup: 10-50×** for position matching

---

### 1.4 Vectorized Variant Operations (`differences.py`)

**CURRENT BOTTLENECK:**
```python
# Multiple dictionary lookups and iterations
exp_index = {variant_key(v): v for v in experimental_section.variants}
ref_index = {variant_key(v): v for v in reference_section.variants}

for exp_variant in experimental_section.variants:
    if variant_key(exp_variant) not in ref_index:
        # Process...
```

**PROBLEM:** Python loops and dictionaries are slower than vectorized NumPy operations

**UPGRADE:**
```python
import numpy as np
from numpy.lib import recfunctions as rfn

class VectorizedDifferenceComputer:
    """Vectorized variant difference computation using NumPy."""
    
    @staticmethod
    def variants_to_array(variants: List[Variant]) -> np.ndarray:
        """Convert variant list to structured NumPy array.
        
        Enables fast set operations and vectorized comparisons.
        """
        if not variants:
            return np.array([], dtype=[
                ('chr', 'U10'), ('pos', 'i8'), 
                ('ref', 'U100'), ('alt', 'U100'),
                ('gt', 'U10')
            ])
        
        # Create structured array
        data = np.array([
            (v.chromosome, v.position, v.ref, v.alt, v.genotype)
            for v in variants
        ], dtype=[
            ('chr', 'U10'), ('pos', 'i8'), 
            ('ref', 'U100'), ('alt', 'U100'),
            ('gt', 'U10')
        ])
        
        return data
    
    @staticmethod
    def compute_set_differences(
        exp_array: np.ndarray,
        ref_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute variant set differences using NumPy.
        
        Returns:
            new_mutations: Variants in exp but not in ref
            missing_variants: Variants in ref but not in exp
            common_variants: Variants in both (for genotype comparison)
        """
        # Create composite keys for set operations
        # Key = (chr, pos, ref, alt)
        exp_keys = rfn.structured_to_unstructured(
            exp_array[['chr', 'pos', 'ref', 'alt']]
        )
        ref_keys = rfn.structured_to_unstructured(
            ref_array[['chr', 'pos', 'ref', 'alt']]
        )
        
        # Find set differences (vectorized)
        # This is much faster than Python loops
        in_exp_not_ref = np.isin(exp_keys, ref_keys, invert=True)
        in_ref_not_exp = np.isin(ref_keys, exp_keys, invert=True)
        in_both = np.isin(exp_keys, ref_keys)
        
        new_mutations = exp_array[in_exp_not_ref]
        missing_variants = ref_array[in_ref_not_exp]
        common_variants = exp_array[in_both]
        
        return new_mutations, missing_variants, common_variants


def compute_variant_differences_vectorized(
    experimental_section: GenomeSection,
    reference_section: GenomeSection
) -> List[VariantDifference]:
    """Vectorized variant difference computation.
    
    Uses NumPy arrays for 2-5× speedup on large variant sets.
    Falls back to original implementation for small sets (<100 variants).
    """
    # For small variant sets, original algorithm is fine
    if (len(experimental_section.variants) < 100 or 
        len(reference_section.variants) < 100):
        return compute_variant_differences(experimental_section, reference_section)
    
    # Convert to NumPy arrays
    computer = VectorizedDifferenceComputer()
    exp_array = computer.variants_to_array(experimental_section.variants)
    ref_array = computer.variants_to_array(reference_section.variants)
    
    # Compute set differences (vectorized)
    new_muts, missing_vars, common_vars = computer.compute_set_differences(
        exp_array, ref_array
    )
    
    differences: List[VariantDifference] = []
    chromosome = experimental_section.chromosome
    
    # Process new mutations
    for row in new_muts:
        # Find original variant for metadata
        var = next(v for v in experimental_section.variants 
                   if v.position == row['pos'] and v.ref == row['ref'])
        
        differences.append(
            VariantDifference(
                difference_type=DifferenceType.NEW_MUTATION,
                chromosome=chromosome,
                position=int(row['pos']),
                exp_ref=str(row['ref']),
                exp_alt=str(row['alt']),
                exp_genotype=str(row['gt']),
                exp_quality=var.quality,
                functional_impact=get_functional_impact(var),
                metadata=var.info.copy() if var.info else {}
            )
        )
    
    # Process missing variants
    for row in missing_vars:
        var = next(v for v in reference_section.variants 
                   if v.position == row['pos'] and v.ref == row['ref'])
        
        differences.append(
            VariantDifference(
                difference_type=DifferenceType.MISSING,
                chromosome=chromosome,
                position=int(row['pos']),
                ref_ref=str(row['ref']),
                ref_alt=str(row['alt']),
                ref_genotype=str(row['gt']),
                ref_quality=var.quality,
                functional_impact=get_functional_impact(var),
                metadata=var.info.copy() if var.info else {}
            )
        )
    
    # Process genotype differences
    # Find matching positions with different genotypes
    for row in common_vars:
        # Find both variants
        exp_var = next(v for v in experimental_section.variants 
                       if v.position == row['pos'] and v.ref == row['ref'])
        ref_var = next(v for v in reference_section.variants 
                       if v.position == row['pos'] and v.ref == row['ref'])
        
        if exp_var.genotype != ref_var.genotype:
            combined_metadata = {}
            if exp_var.info:
                combined_metadata.update(exp_var.info)
            if ref_var.info:
                combined_metadata["ref_info"] = ref_var.info.copy()
            
            differences.append(
                VariantDifference(
                    difference_type=DifferenceType.GENOTYPE_DIFF,
                    chromosome=chromosome,
                    position=exp_var.position,
                    exp_ref=exp_var.ref,
                    exp_alt=exp_var.alt,
                    exp_genotype=exp_var.genotype,
                    exp_quality=exp_var.quality,
                    ref_ref=ref_var.ref,
                    ref_alt=ref_var.alt,
                    ref_genotype=ref_var.genotype,
                    ref_quality=ref_var.quality,
                    functional_impact=get_functional_impact(exp_var),
                    metadata=combined_metadata
                )
            )
    
    # Sort by position
    differences.sort(key=lambda d: d.position)
    
    return differences
```

**IMPLEMENTATION NOTES:**
- Add `VectorizedDifferenceComputer` class to `differences.py`
- Create new function `compute_variant_differences_vectorized`
- Keep original function as fallback for small sets
- **Expected speedup: 2-5×** for large variant sets (>1000 variants)

---

### 1.5 Memory-Efficient Data Structures (`differences.py`)

**CURRENT ISSUE:** Large memory footprint from dataclass objects

**UPGRADE:**
```python
@dataclass
class VariantDifference:
    """Memory-efficient variant difference with __slots__."""
    
    # NEW: Add __slots__ to reduce memory by 40-50%
    __slots__ = [
        'difference_type', 'chromosome', 'position',
        'exp_ref', 'exp_alt', 'exp_genotype', 'exp_quality',
        'ref_ref', 'ref_alt', 'ref_genotype', 'ref_quality',
        'functional_impact', 'metadata'
    ]
    
    difference_type: DifferenceType
    chromosome: str
    position: int
    
    # Experimental variant data
    exp_ref: Optional[str] = None
    exp_alt: Optional[str] = None
    exp_genotype: Optional[str] = None
    exp_quality: float = 1.0
    
    # Reference variant data
    ref_ref: Optional[str] = None
    ref_alt: Optional[str] = None
    ref_genotype: Optional[str] = None
    ref_quality: float = 1.0
    
    # Annotation
    functional_impact: FunctionalImpact = FunctionalImpact.UNKNOWN
    metadata: Dict[str, any] = field(default_factory=dict)
```

**ALSO ADD TO:**
- `Variant` class in `reference_management.py`
- `GenomeSection` class in `reference_management.py`
- `AlignmentScore` class in `sequence_alignment.py`

**IMPLEMENTATION NOTES:**
- Add `__slots__` to all dataclasses
- **Expected benefit: 40-50% memory reduction, 10-20% speed increase**

---

## Phase 2: Reference Genome Caching (Days 4-5)

### 2.1 In-Memory Reference Cache (`reference_management.py`)

**CREATE NEW FILE:** `genomevault/differential_encoding/reference_cache.py`

```python
"""
Reference Genome Caching System

Implements in-memory caching with FASTA indexing for fast reference access.
Inspired by BWA's FM-index and HTSlib's ref-cache.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from functools import lru_cache

# Try to import pysam for FASTA indexing, fall back to basic implementation
try:
    import pysam
    PYSAM_AVAILABLE = True
except ImportError:
    PYSAM_AVAILABLE = False
    logging.warning("pysam not available, using basic FASTA access")

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class ReferenceCache:
    """
    In-memory reference genome cache with FASTA indexing.
    
    Strategies:
    1. Whole-genome in RAM (best for systems with >8GB RAM)
    2. Per-chromosome caching (good compromise)
    3. LRU segment cache (for memory-constrained systems)
    """
    
    def __init__(
        self,
        reference_path: Path,
        cache_strategy: str = "per_chromosome",
        max_cache_size_mb: int = 4096  # 4GB default
    ):
        """
        Initialize reference cache.
        
        Args:
            reference_path: Path to reference FASTA file
            cache_strategy: One of 'whole_genome', 'per_chromosome', 'lru_segments'
            max_cache_size_mb: Maximum cache size in MB
        """
        self.reference_path = reference_path
        self.cache_strategy = cache_strategy
        self.max_cache_size_mb = max_cache_size_mb
        
        # Cache storage
        self._chromosome_cache: Dict[str, str] = {}
        self._current_chromosome: Optional[str] = None
        
        # FASTA access
        if PYSAM_AVAILABLE:
            self._fasta = pysam.FastaFile(str(reference_path))
            self._use_pysam = True
            logger.info(f"Using pysam for indexed FASTA access: {reference_path}")
        else:
            self._fasta = None
            self._use_pysam = False
            logger.warning("pysam not available, using slower FASTA access")
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(
            f"Initialized ReferenceCache: strategy={cache_strategy}, "
            f"max_size={max_cache_size_mb}MB"
        )
    
    def get_sequence(
        self,
        chromosome: str,
        start: int,
        end: int
    ) -> str:
        """
        Get reference sequence for a genomic region.
        
        Uses caching strategy to minimize disk I/O.
        
        Args:
            chromosome: Chromosome name
            start: Start position (0-based)
            end: End position (exclusive)
            
        Returns:
            Reference sequence string
        """
        # Strategy 1: Whole chromosome in cache
        if self.cache_strategy == "per_chromosome":
            return self._get_with_chromosome_cache(chromosome, start, end)
        
        # Strategy 2: Whole genome in cache
        elif self.cache_strategy == "whole_genome":
            return self._get_with_whole_genome_cache(chromosome, start, end)
        
        # Strategy 3: LRU segment cache
        else:  # "lru_segments"
            return self._get_with_lru_cache(chromosome, start, end)
    
    def _get_with_chromosome_cache(
        self,
        chromosome: str,
        start: int,
        end: int
    ) -> str:
        """Get sequence using per-chromosome caching."""
        # Check if chromosome is in cache
        if chromosome not in self._chromosome_cache:
            self.cache_misses += 1
            self._load_chromosome(chromosome)
        else:
            self.cache_hits += 1
        
        # Extract subsequence
        seq = self._chromosome_cache[chromosome]
        return seq[start:end]
    
    def _load_chromosome(self, chromosome: str) -> None:
        """Load entire chromosome into cache."""
        logger.info(f"Loading chromosome {chromosome} into cache...")
        
        if self._use_pysam:
            # Use pysam for fast indexed access
            seq = self._fasta.fetch(chromosome)
        else:
            # Fallback: read from FASTA file manually
            seq = self._read_chromosome_from_fasta(chromosome)
        
        # Manage cache size
        if self.cache_strategy == "per_chromosome":
            # Keep only one chromosome in memory at a time
            if self._current_chromosome and self._current_chromosome != chromosome:
                logger.debug(f"Evicting chromosome {self._current_chromosome} from cache")
                del self._chromosome_cache[self._current_chromosome]
            
            self._current_chromosome = chromosome
        
        self._chromosome_cache[chromosome] = seq
        
        # Log cache size
        cache_size_mb = len(seq) / (1024 * 1024)
        logger.info(f"Loaded {chromosome}: {cache_size_mb:.2f} MB")
    
    def _get_with_whole_genome_cache(
        self,
        chromosome: str,
        start: int,
        end: int
    ) -> str:
        """Get sequence using whole-genome caching."""
        # Load all chromosomes on first access
        if not self._chromosome_cache:
            self._load_whole_genome()
        
        self.cache_hits += 1
        seq = self._chromosome_cache[chromosome]
        return seq[start:end]
    
    def _load_whole_genome(self) -> None:
        """Load entire reference genome into memory."""
        logger.info("Loading entire reference genome into cache...")
        
        if self._use_pysam:
            for chrom in self._fasta.references:
                self._chromosome_cache[chrom] = self._fasta.fetch(chrom)
        else:
            # Load all chromosomes from FASTA
            raise NotImplementedError("Whole genome caching without pysam not implemented")
        
        total_size_mb = sum(len(seq) for seq in self._chromosome_cache.values()) / (1024 * 1024)
        logger.info(f"Loaded entire genome: {total_size_mb:.2f} MB")
    
    @lru_cache(maxsize=1000)
    def _get_with_lru_cache(
        self,
        chromosome: str,
        start: int,
        end: int
    ) -> str:
        """Get sequence using LRU segment cache."""
        if self._use_pysam:
            return self._fasta.fetch(chromosome, start, end)
        else:
            # Fallback to reading from file
            return self._read_region_from_fasta(chromosome, start, end)
    
    def _read_chromosome_from_fasta(self, chromosome: str) -> str:
        """Manually read chromosome from FASTA file (fallback)."""
        # Basic FASTA parser
        # This is slow - use pysam if possible
        with open(self.reference_path) as f:
            in_target = False
            sequence = []
            
            for line in f:
                if line.startswith('>'):
                    # Check if this is our chromosome
                    chrom_name = line[1:].strip().split()[0]
                    if chrom_name == chromosome:
                        in_target = True
                    elif in_target:
                        break  # Done reading this chromosome
                elif in_target:
                    sequence.append(line.strip())
            
            return ''.join(sequence)
    
    def _read_region_from_fasta(
        self,
        chromosome: str,
        start: int,
        end: int
    ) -> str:
        """Read specific region from FASTA (fallback)."""
        # Load whole chromosome and extract region
        if chromosome not in self._chromosome_cache:
            self._load_chromosome(chromosome)
        
        return self._chromosome_cache[chromosome][start:end]
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cached_chromosomes": list(self._chromosome_cache.keys()),
            "cache_size_mb": sum(len(seq) for seq in self._chromosome_cache.values()) / (1024 * 1024)
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._chromosome_cache.clear()
        self._current_chromosome = None
        logger.info("Cache cleared")
```

**INTEGRATION:**

Update `reference_management.py`:
```python
from genomevault.differential_encoding.reference_cache import ReferenceCache

class SecureReferenceGenomeManager:
    def __init__(self, reference_pool_path: Path, ...):
        # ... existing init ...
        
        # NEW: Add reference cache
        self.reference_cache = None
        if self.pool.reference_fasta_path:
            self.reference_cache = ReferenceCache(
                self.pool.reference_fasta_path,
                cache_strategy="per_chromosome"
            )
    
    def get_reference_sequence(
        self,
        chromosome: str,
        start: int,
        end: int
    ) -> str:
        """Get reference sequence using cache."""
        if self.reference_cache:
            return self.reference_cache.get_sequence(chromosome, start, end)
        else:
            # Fallback to existing method
            return self._get_sequence_from_fasta(chromosome, start, end)
```

**DEPENDENCIES:**
```bash
pip install pysam  # For indexed FASTA access
```

**IMPLEMENTATION NOTES:**
- Create new `reference_cache.py` file
- Integrate with `SecureReferenceGenomeManager`
- Add pysam to requirements.txt (optional but recommended)
- **Expected speedup: 10-100×** for reference sequence access

---

## Phase 3: Parallel Processing (Days 6-7)

### 3.1 Parallel Chunk Processing (`pipeline.py`)

**CREATE NEW FILE:** `genomevault/differential_encoding/parallel_processor.py`

```python
"""
Parallel Processing for Differential Encoding

Implements multi-core chunk processing with load balancing.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Callable, Any, Optional
import multiprocessing as mp
from pathlib import Path

from genomevault.utils.logging import get_logger
from genomevault.differential_encoding.reference_management import GenomeSection

logger = get_logger(__name__)


@dataclass
class ChunkTask:
    """Represents a chunk processing task."""
    chunk_id: str
    section: GenomeSection
    reference_id: str


@dataclass
class ChunkResult:
    """Result from processing a chunk."""
    chunk_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    processing_time_ms: float = 0.0


class ParallelChunkProcessor:
    """
    Parallel processor for genome chunks.
    
    Uses ProcessPoolExecutor for CPU-bound work.
    Implements load balancing and error handling.
    """
    
    def __init__(
        self,
        num_workers: Optional[int] = None,
        use_threads: bool = False,
        chunk_batch_size: int = 10
    ):
        """
        Initialize parallel processor.
        
        Args:
            num_workers: Number of worker processes (default: CPU count - 1)
            use_threads: Use threads instead of processes (for I/O-bound work)
            chunk_batch_size: Number of chunks to batch per worker
        """
        if num_workers is None:
            # Use all cores except 1 for main process
            num_workers = max(1, mp.cpu_count() - 1)
        
        self.num_workers = num_workers
        self.use_threads = use_threads
        self.chunk_batch_size = chunk_batch_size
        
        logger.info(
            f"Initialized ParallelChunkProcessor: "
            f"workers={num_workers}, "
            f"mode={'threads' if use_threads else 'processes'}, "
            f"batch_size={chunk_batch_size}"
        )
    
    def process_chunks(
        self,
        chunks: List[ChunkTask],
        process_func: Callable[[ChunkTask], Any]
    ) -> List[ChunkResult]:
        """
        Process chunks in parallel.
        
        Args:
            chunks: List of chunk tasks
            process_func: Function to process each chunk
            
        Returns:
            List of results
        """
        if len(chunks) == 0:
            return []
        
        # For small number of chunks, don't bother with parallelism
        if len(chunks) < self.num_workers:
            logger.info(f"Processing {len(chunks)} chunks sequentially (too few for parallel)")
            return [self._process_single_chunk(chunk, process_func) for chunk in chunks]
        
        logger.info(f"Processing {len(chunks)} chunks in parallel with {self.num_workers} workers")
        
        # Choose executor type
        ExecutorClass = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        
        results = []
        
        with ExecutorClass(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(self._process_single_chunk, chunk, process_func): chunk
                for chunk in chunks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Log progress
                    if len(results) % 10 == 0:
                        logger.info(f"Processed {len(results)}/{len(chunks)} chunks")
                
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")
                    results.append(ChunkResult(
                        chunk_id=chunk.chunk_id,
                        success=False,
                        result=None,
                        error=str(e)
                    ))
        
        logger.info(f"Parallel processing complete: {len(results)} chunks processed")
        
        # Report any errors
        errors = [r for r in results if not r.success]
        if errors:
            logger.warning(f"Encountered {len(errors)} errors during processing")
        
        return results
    
    def _process_single_chunk(
        self,
        chunk: ChunkTask,
        process_func: Callable[[ChunkTask], Any]
    ) -> ChunkResult:
        """Process a single chunk with error handling and timing."""
        import time
        
        start_time = time.time()
        
        try:
            result = process_func(chunk)
            elapsed_ms = (time.time() - start_time) * 1000
            
            return ChunkResult(
                chunk_id=chunk.chunk_id,
                success=True,
                result=result,
                processing_time_ms=elapsed_ms
            )
        
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Error in chunk {chunk.chunk_id}: {e}")
            
            return ChunkResult(
                chunk_id=chunk.chunk_id,
                success=False,
                result=None,
                error=str(e),
                processing_time_ms=elapsed_ms
            )


def process_chunk_wrapper(chunk_task: ChunkTask) -> Any:
    """
    Wrapper function for processing chunks in parallel.
    
    This function needs to be at module level for pickle serialization.
    """
    from genomevault.differential_encoding.differences import compute_variant_differences
    from genomevault.differential_encoding.reference_management import SecureReferenceGenomeManager
    
    # Get reference section
    # NOTE: This will need access to reference manager
    # Consider passing reference data with the task or using shared memory
    
    # Placeholder for actual implementation
    # You'll need to adapt this based on your pipeline structure
    pass
```

**INTEGRATION:**

Update `pipeline.py` or `enhanced_pipeline.py`:
```python
from genomevault.differential_encoding.parallel_processor import (
    ParallelChunkProcessor,
    ChunkTask,
    ChunkResult
)

class EnhancedDifferentialEncodingPipeline:
    def __init__(self, ..., enable_parallel: bool = True, num_workers: Optional[int] = None):
        # ... existing init ...
        
        # NEW: Parallel processor
        self.enable_parallel = enable_parallel
        if enable_parallel:
            self.parallel_processor = ParallelChunkProcessor(num_workers=num_workers)
        else:
            self.parallel_processor = None
    
    def encode_file(self, input_file: Path, ...) -> Dict:
        """Encode file with optional parallel processing."""
        # ... existing code to get chunks ...
        
        # NEW: Process chunks in parallel
        if self.enable_parallel and len(chunks) > 1:
            # Create tasks
            tasks = [
                ChunkTask(
                    chunk_id=f"chunk_{i}",
                    section=chunk,
                    reference_id=reference_id
                )
                for i, chunk in enumerate(chunks)
            ]
            
            # Process in parallel
            results = self.parallel_processor.process_chunks(
                tasks,
                self._process_chunk_func
            )
            
            # Aggregate results
            # ...
        else:
            # Sequential processing
            # ... existing code ...
```

**IMPLEMENTATION NOTES:**
- Create new `parallel_processor.py` file
- Integrate with existing pipeline
- Handle serialization of reference data for multiprocessing
- **Expected speedup: 4-16×** on multi-core systems

---

## Phase 4: GPU Acceleration (Optional, Days 8-10)

### 4.1 GPU-Accelerated Feature Computation

**CREATE NEW FILE:** `genomevault/differential_encoding/gpu_encoder.py`

```python
"""
GPU-Accelerated Feature Vector Computation

Uses PyTorch or CuPy for GPU acceleration of hypervector encoding.
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from typing import List, Optional
import numpy as np

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class GPUFeatureEncoder:
    """
    GPU-accelerated feature vector encoder.
    
    Batches multiple chunks and computes their feature vectors
    in parallel on GPU.
    """
    
    def __init__(
        self,
        dimension: int = 384,
        batch_size: int = 100,
        use_torch: bool = True
    ):
        """
        Initialize GPU encoder.
        
        Args:
            dimension: Feature vector dimension
            batch_size: Number of chunks to process per batch
            use_torch: Use PyTorch (True) or CuPy (False)
        """
        self.dimension = dimension
        self.batch_size = batch_size
        self.use_torch = use_torch and TORCH_AVAILABLE
        self.use_cupy = not use_torch and CUPY_AVAILABLE
        
        # Check GPU availability
        if self.use_torch:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using PyTorch on {self.device}")
        elif self.use_cupy:
            logger.info("Using CuPy for GPU acceleration")
        else:
            logger.warning("No GPU library available, falling back to CPU")
        
        # Initialize random projection matrix on GPU
        self._init_projection_matrix()
    
    def _init_projection_matrix(self) -> None:
        """Initialize random projection matrix on GPU."""
        if self.use_torch:
            # Random Gaussian projection
            self.projection_matrix = torch.randn(
                self.dimension,
                self.dimension,
                device=self.device
            )
        elif self.use_cupy:
            self.projection_matrix = cp.random.randn(
                self.dimension,
                self.dimension
            )
        else:
            self.projection_matrix = np.random.randn(
                self.dimension,
                self.dimension
            )
    
    def encode_batch(
        self,
        feature_matrices: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Encode batch of feature matrices on GPU.
        
        Args:
            feature_matrices: List of feature matrices (each shape [n_variants, dim])
            
        Returns:
            List of encoded hypervectors
        """
        if not feature_matrices:
            return []
        
        # Process in batches
        results = []
        for i in range(0, len(feature_matrices), self.batch_size):
            batch = feature_matrices[i:i+self.batch_size]
            batch_results = self._encode_batch_gpu(batch)
            results.extend(batch_results)
        
        return results
    
    def _encode_batch_gpu(
        self,
        batch: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Encode batch on GPU."""
        if self.use_torch:
            return self._encode_batch_torch(batch)
        elif self.use_cupy:
            return self._encode_batch_cupy(batch)
        else:
            return self._encode_batch_cpu(batch)
    
    def _encode_batch_torch(
        self,
        batch: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Encode using PyTorch on GPU."""
        results = []
        
        for matrix in batch:
            # Convert to torch tensor on GPU
            tensor = torch.from_numpy(matrix).float().to(self.device)
            
            # Matrix multiplication: result = matrix @ projection_matrix.T
            encoded = torch.matmul(tensor, self.projection_matrix.T)
            
            # Aggregate (sum over variants)
            aggregated = torch.sum(encoded, dim=0)
            
            # Binarize (sign function)
            binary = torch.sign(aggregated)
            
            # Convert back to CPU and numpy
            result = binary.cpu().numpy()
            results.append(result)
        
        return results
    
    def _encode_batch_cupy(
        self,
        batch: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Encode using CuPy on GPU."""
        results = []
        
        for matrix in batch:
            # Convert to cupy array
            cu_matrix = cp.asarray(matrix)
            
            # Matrix multiplication
            encoded = cp.matmul(cu_matrix, self.projection_matrix.T)
            
            # Aggregate
            aggregated = cp.sum(encoded, axis=0)
            
            # Binarize
            binary = cp.sign(aggregated)
            
            # Convert back to numpy
            result = cp.asnumpy(binary)
            results.append(result)
        
        return results
    
    def _encode_batch_cpu(
        self,
        batch: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Fallback CPU encoding."""
        results = []
        
        for matrix in batch:
            # NumPy operations
            encoded = np.matmul(matrix, self.projection_matrix.T)
            aggregated = np.sum(encoded, axis=0)
            binary = np.sign(aggregated)
            results.append(binary)
        
        return results
```

**INTEGRATION:**

Update `hypervector_encoder.py`:
```python
from genomevault.differential_encoding.gpu_encoder import GPUFeatureEncoder

class HypervectorEncoder:
    def __init__(self, ..., use_gpu: bool = False):
        # ... existing init ...
        
        # NEW: GPU encoder
        if use_gpu:
            self.gpu_encoder = GPUFeatureEncoder(dimension=self.dimension)
        else:
            self.gpu_encoder = None
    
    def encode_batch(self, differences_list: List[List[VariantDifference]]):
        """Encode batch of differences."""
        if self.gpu_encoder:
            # Use GPU encoding
            # ...
        else:
            # Use CPU encoding
            # ...
```

**DEPENDENCIES:**
```bash
# For PyTorch GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# OR for CuPy
pip install cupy-cuda11x  # Replace 11x with your CUDA version
```

**IMPLEMENTATION NOTES:**
- Create new `gpu_encoder.py` file
- Make GPU support optional (detect at runtime)
- **Expected speedup: 10-50×** for large batches

---

## Phase 5: Advanced Optimizations (Days 11-14)

### 5.1 Adaptive Chunking Strategy

**UPDATE:** `chunking.py`

```python
class AdaptiveChunkingStrategy:
    """
    Adaptive chunking based on variant density.
    
    Inspired by shotgun sequencing overlapping fragments.
    """
    
    def __init__(
        self,
        base_chunk_size: int = 100000,  # 100kb base
        min_chunk_size: int = 10000,
        max_chunk_size: int = 1000000,
        target_variants_per_chunk: int = 100,
        overlap_size: int = 10000  # 10kb overlap
    ):
        self.base_chunk_size = base_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.target_variants_per_chunk = target_variants_per_chunk
        self.overlap_size = overlap_size
    
    def create_adaptive_chunks(
        self,
        variants: List[Variant],
        chromosome: str
    ) -> List[GenomeSection]:
        """
        Create overlapping, variable-sized chunks based on variant density.
        
        Algorithm:
        1. Scan variants to compute local density
        2. Adjust chunk size inversely to density
        3. Create overlapping chunks
        """
        if not variants:
            return []
        
        # Sort variants by position
        sorted_variants = sorted(variants, key=lambda v: v.position)
        
        # Compute variant density map
        density_map = self._compute_density_map(sorted_variants)
        
        # Create chunks with adaptive sizing
        chunks = []
        current_pos = sorted_variants[0].position
        end_pos = sorted_variants[-1].position
        
        while current_pos < end_pos:
            # Determine chunk size based on local density
            local_density = density_map.get(current_pos, 0)
            chunk_size = self._adaptive_chunk_size(local_density)
            
            chunk_end = min(current_pos + chunk_size, end_pos)
            
            # Get variants in this chunk
            chunk_variants = [
                v for v in sorted_variants
                if current_pos <= v.position < chunk_end
            ]
            
            if chunk_variants:
                chunks.append(GenomeSection(
                    chromosome=chromosome,
                    start_position=current_pos,
                    end_position=chunk_end,
                    variants=chunk_variants
                ))
            
            # Move to next chunk (with overlap)
            current_pos = chunk_end - self.overlap_size
        
        return chunks
    
    def _compute_density_map(
        self,
        variants: List[Variant],
        window_size: int = 50000
    ) -> Dict[int, float]:
        """Compute variant density (variants per kb) in sliding windows."""
        density_map = {}
        
        for i, var in enumerate(variants):
            # Count variants in window
            window_start = var.position - window_size // 2
            window_end = var.position + window_size // 2
            
            count = sum(
                1 for v in variants
                if window_start <= v.position < window_end
            )
            
            density = count / (window_size / 1000)  # Variants per kb
            density_map[var.position] = density
        
        return density_map
    
    def _adaptive_chunk_size(self, density: float) -> int:
        """Compute chunk size based on variant density."""
        if density == 0:
            return self.max_chunk_size
        
        # Inverse relationship: higher density -> smaller chunks
        # Target: ~target_variants_per_chunk variants per chunk
        size = int(self.target_variants_per_chunk / density * 1000)
        
        # Clamp to min/max
        return max(self.min_chunk_size, min(self.max_chunk_size, size))
```

---

## Summary & Implementation Checklist

### Quick Wins (Week 1)
- [ ] Replace SHA-256 with fast hash (50-100× speedup)
- [ ] Add `__slots__` to dataclasses (40% memory reduction)
- [ ] Implement k-mer caching with LRU (2-3× speedup)
- [ ] Add interval tree for position matching (10-50× speedup)

### Core Improvements (Week 2)
- [ ] Implement reference genome caching with pysam
- [ ] Vectorize operations with NumPy (2-5× speedup)
- [ ] Add parallel chunk processing (4-16× speedup)
- [ ] Profile and validate improvements

### Advanced Features (Week 3-4)
- [ ] GPU-accelerated feature computation (10-50× speedup)
- [ ] Adaptive chunking strategy
- [ ] Comprehensive benchmarking
- [ ] Documentation and testing

### Expected Total Improvement
- **Encoding time**: 21.67ms → **<5ms** (4-5× faster)
- **Throughput**: 230K → **1M+ variants/sec** (4-5× increase)
- **Memory**: **40-50% reduction**
- **Scalability**: Linear with CPU cores

### Dependencies to Add
```bash
pip install intervaltree
pip install pysam
pip install xxhash  # Optional, for even faster hashing

# For GPU support (optional)
pip install torch  # OR
pip install cupy-cuda11x
```

---

## Testing Strategy

### 1. Unit Tests
Create tests for each optimization:
```python
# tests/test_optimized_differences.py
def test_fast_hash_correctness():
    """Verify fast hash produces valid results."""
    # Compare against SHA-256 for collision rate
    pass

def test_vectorized_differences():
    """Verify NumPy version matches original."""
    # Compare outputs on same data
    pass

def test_interval_tree_matching():
    """Verify interval tree matching is correct."""
    # Compare against linear search
    pass
```

### 2. Performance Benchmarks
```python
# benchmarks/test_optimizations.py
def benchmark_hash_functions():
    """Compare SHA-256 vs fast hash."""
    pass

def benchmark_parallel_vs_sequential():
    """Measure parallel speedup."""
    pass

def benchmark_with_gpu():
    """Measure GPU acceleration."""
    pass
```

### 3. Integration Tests
```bash
# Run full pipeline on chr22
python benchmarks/differential_encoding/benchmark_end_to_end.py

# Compare before/after
diff benchmark_results/before.json benchmark_results/after.json
```

---

## Rollout Plan

### Phase 1 (Days 1-3): Quick Wins
1. Branch: `feature/fast-hashing`
2. Implement fast k-mer hashing
3. Add `__slots__` to dataclasses
4. Test and benchmark
5. PR and merge

### Phase 2 (Days 4-7): Core Algorithms
1. Branch: `feature/vectorized-operations`
2. Implement interval tree indexing
3. Add NumPy vectorization
4. Implement reference caching
5. Test and benchmark
6. PR and merge

### Phase 3 (Days 8-10): Parallelization
1. Branch: `feature/parallel-processing`
2. Implement parallel chunk processor
3. Integrate with pipeline
4. Test and benchmark
5. PR and merge

### Phase 4 (Days 11-14): GPU & Polish
1. Branch: `feature/gpu-acceleration`
2. Implement GPU encoder
3. Add adaptive chunking
4. Comprehensive testing
5. Documentation
6. Final benchmarks
7. PR and merge

---

This plan provides a complete roadmap for implementing all optimizations. Each section includes:
- Specific code changes
- Implementation notes
- Expected performance improvements
- Testing strategies
- Dependencies

Ready to hand off to Claude Code for implementation!
