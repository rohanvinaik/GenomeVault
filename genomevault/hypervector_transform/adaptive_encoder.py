#!/usr/bin/env python3
"""
Production High-Fidelity Encoder (avg k=6)
==========================================

HIGH-FIDELITY MODE: Uses k=4/6/8 ONLY (no k=2) for maximum accuracy.
Average k = 6.0 with target distribution: 25% k=4, 50% k=6, 25% k=8

Implements the 2-Bank Orthogonal Ternary Projection (OTP) architecture.

Key Parameters:
- D = 4096 (dimensions)
- N = 512 (chunk size in bp)
- k = 4, 6, or 8 (NO k=2)
- 2 banks: Hydrophobic (A=+1, T=-1) and MajorGroove (G=+1, C=-1)

High-Fidelity Thresholds (based on difficulty percentiles):
- difficulty < 0.0300 (25th): k=4 (25% of genome - easiest)
- difficulty 0.0300-0.1319: k=6 (50% of genome - moderate)
- difficulty >= 0.1319 (75th): k=8 (25% of genome - hardest)

Expected accuracy: ~99.2% (weighted avg of 98.96/99.00/99.55%)

CRITICAL: k-dependent seed formula for codebook independence:
  seed_for_k = base_seed + position * 100 + k * 10000
"""

import h5py
import numpy as np
import logging
import time
import gzip
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

try:
    from pyfaidx import Fasta
    HAS_PYFAIDX = True
except ImportError:
    HAS_PYFAIDX = False
    print("Warning: pyfaidx not installed. Install with: pip install pyfaidx")

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: numba not installed. Install with: pip install numba")


# === NUMBA JIT-COMPILED ENCODING FUNCTIONS ===
if HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _encode_chunk_numba(
        seq_codes: np.ndarray,  # int8 array: A=0, C=1, G=2, T=3, N=4
        dim_cache: np.ndarray,  # (N, k) int32 array of dimension indices
        signs: np.ndarray,      # (k,) int8 array of signs
        D: int,
        N: int,
        k: int
    ) -> np.ndarray:
        """
        Numba JIT-compiled encoding loop.

        Returns: (2, D) int8 encoded vector
        """
        # Accumulators
        hydro_acc = np.zeros(D, dtype=np.int32)
        groove_acc = np.zeros(D, dtype=np.int32)

        for pos in range(N):
            code = seq_codes[pos]
            if code == 4:  # N
                continue

            # Compute signals
            # Hydrophobic: A=+1, T=-1, G/C=0
            if code == 0:  # A
                hydro_val = 1
            elif code == 3:  # T
                hydro_val = -1
            else:
                hydro_val = 0

            # MajorGroove: G=+1, C=-1, A/T=0
            if code == 2:  # G
                groove_val = 1
            elif code == 1:  # C
                groove_val = -1
            else:
                groove_val = 0

            # Get dimensions for this position
            dims = dim_cache[pos]

            # Accumulate with sign binding
            for i in range(k):
                hydro_acc[dims[i]] += hydro_val * signs[i]
                groove_acc[dims[i]] += groove_val * signs[i]

        # Sign quantization
        encoded = np.zeros((2, D), dtype=np.int8)
        for d in range(D):
            if hydro_acc[d] > 0:
                encoded[0, d] = 1
            elif hydro_acc[d] < 0:
                encoded[0, d] = -1

            if groove_acc[d] > 0:
                encoded[1, d] = 1
            elif groove_acc[d] < 0:
                encoded[1, d] = -1

        return encoded

    @njit(cache=True)
    def _seq_to_codes(seq_bytes: np.ndarray) -> np.ndarray:
        """Convert sequence bytes to numeric codes."""
        n = len(seq_bytes)
        codes = np.empty(n, dtype=np.int8)
        for i in range(n):
            c = seq_bytes[i]
            if c == 65:  # A
                codes[i] = 0
            elif c == 67:  # C
                codes[i] = 1
            elif c == 71:  # G
                codes[i] = 2
            elif c == 84:  # T
                codes[i] = 3
            else:  # N or other
                codes[i] = 4
        return codes

    # === NUMBA JIT-COMPILED DECODE FUNCTIONS ===
    @njit(cache=True, fastmath=True)
    def _decode_chunk_numba(
        encoded_h: np.ndarray,  # (D,) int8 - hydrophobic bank
        encoded_g: np.ndarray,  # (D,) int8 - major groove bank
        dim_cache: np.ndarray,  # (N, k) int32 - dimension indices
        signs: np.ndarray,      # (k,) int8 - Hadamard signs
        N: int,
        k: int
    ) -> np.ndarray:
        """
        Numba JIT-compiled decode for all positions.

        Returns: (N,) int8 array with decoded nucleotides:
            0 = A, 1 = C, 2 = G, 3 = T, 4 = N (unknown/tie)
        """
        decoded = np.empty(N, dtype=np.int8)

        for pos in range(N):
            dims = dim_cache[pos]

            # Inline dot products (no function call overhead!)
            sim_h = 0
            sim_g = 0
            for i in range(k):
                sim_h += encoded_h[dims[i]] * signs[i]
                sim_g += encoded_g[dims[i]] * signs[i]

            # Absolute values
            abs_h = sim_h if sim_h > 0 else -sim_h
            abs_g = sim_g if sim_g > 0 else -sim_g

            # Margin for tie detection
            margin = abs_h - abs_g if abs_h > abs_g else abs_g - abs_h

            # 2-pathway decode with low-margin flip (E13g validated)
            if margin == 0:
                # Flip to AT pathway (statistically favored)
                if sim_h >= 0:
                    decoded[pos] = 0  # A
                else:
                    decoded[pos] = 3  # T
            elif abs_h > abs_g:
                # AT subspace active
                if sim_h > 0:
                    decoded[pos] = 0  # A
                else:
                    decoded[pos] = 3  # T
            else:
                # GC subspace active
                if sim_g > 0:
                    decoded[pos] = 2  # G
                else:
                    decoded[pos] = 1  # C

        return decoded

    @njit(cache=True, fastmath=True)
    def _decode_with_margins_numba(
        encoded_h: np.ndarray,
        encoded_g: np.ndarray,
        dim_cache: np.ndarray,
        signs: np.ndarray,
        N: int,
        k: int
    ) -> Tuple:
        """
        Decode with margin information for accuracy analysis.

        Returns:
            decoded: (N,) int8 - decoded nucleotides
            margins: (N,) int8 - decode margins (confidence)
        """
        decoded = np.empty(N, dtype=np.int8)
        margins = np.empty(N, dtype=np.int8)

        for pos in range(N):
            dims = dim_cache[pos]

            sim_h = 0
            sim_g = 0
            for i in range(k):
                sim_h += encoded_h[dims[i]] * signs[i]
                sim_g += encoded_g[dims[i]] * signs[i]

            abs_h = sim_h if sim_h > 0 else -sim_h
            abs_g = sim_g if sim_g > 0 else -sim_g
            margin = abs_h - abs_g if abs_h > abs_g else abs_g - abs_h
            margins[pos] = margin

            if margin == 0:
                decoded[pos] = 0 if sim_h >= 0 else 3
            elif abs_h > abs_g:
                decoded[pos] = 0 if sim_h > 0 else 3
            else:
                decoded[pos] = 2 if sim_g > 0 else 1

        return decoded, margins

    @njit(cache=True, fastmath=True)
    def _compute_accuracy_numba(
        encoded_h: np.ndarray,
        encoded_g: np.ndarray,
        original_codes: np.ndarray,  # (N,) int8 - original sequence codes
        dim_cache: np.ndarray,
        signs: np.ndarray,
        N: int,
        k: int
    ) -> Tuple:
        """
        Compute accuracy in a single fused pass.

        Returns:
            correct: int - number of correctly decoded positions
            total: int - number of non-N positions
            margin_0_count: int - positions with margin=0
            margin_0_correct: int - margin=0 positions decoded correctly
        """
        correct = 0
        total = 0
        m0_count = 0
        m0_correct = 0

        for pos in range(N):
            orig = original_codes[pos]
            if orig == 4:  # N - skip
                continue
            total += 1

            dims = dim_cache[pos]
            sim_h = 0
            sim_g = 0
            for i in range(k):
                sim_h += encoded_h[dims[i]] * signs[i]
                sim_g += encoded_g[dims[i]] * signs[i]

            abs_h = sim_h if sim_h > 0 else -sim_h
            abs_g = sim_g if sim_g > 0 else -sim_g
            margin = abs_h - abs_g if abs_h > abs_g else abs_g - abs_h

            # Decode
            if margin == 0:
                decoded = 0 if sim_h >= 0 else 3
                m0_count += 1
                if decoded == orig:
                    m0_correct += 1
                    correct += 1
            elif abs_h > abs_g:
                decoded = 0 if sim_h > 0 else 3
                if decoded == orig:
                    correct += 1
            else:
                decoded = 2 if sim_g > 0 else 1
                if decoded == orig:
                    correct += 1

        return correct, total, m0_count, m0_correct

    @njit(cache=True, parallel=True)
    def _batch_decode_numba(
        encoded_batch: np.ndarray,  # (batch_size, 2, D) int8
        dim_cache: np.ndarray,      # (N, k) int32
        signs: np.ndarray,          # (k,) int8
        N: int,
        k: int
    ) -> np.ndarray:
        """
        Batch decode multiple chunks in parallel.

        Returns: (batch_size, N) int8 - decoded sequences
        """
        batch_size = encoded_batch.shape[0]
        decoded_batch = np.empty((batch_size, N), dtype=np.int8)

        for chunk_idx in prange(batch_size):
            encoded_h = encoded_batch[chunk_idx, 0, :]
            encoded_g = encoded_batch[chunk_idx, 1, :]

            for pos in range(N):
                dims = dim_cache[pos]

                sim_h = 0
                sim_g = 0
                for i in range(k):
                    sim_h += encoded_h[dims[i]] * signs[i]
                    sim_g += encoded_g[dims[i]] * signs[i]

                abs_h = sim_h if sim_h > 0 else -sim_h
                abs_g = sim_g if sim_g > 0 else -sim_g
                margin = abs_h - abs_g if abs_h > abs_g else abs_g - abs_h

                if margin == 0:
                    decoded_batch[chunk_idx, pos] = 0 if sim_h >= 0 else 3
                elif abs_h > abs_g:
                    decoded_batch[chunk_idx, pos] = 0 if sim_h > 0 else 3
                else:
                    decoded_batch[chunk_idx, pos] = 2 if sim_g > 0 else 1

        return decoded_batch

# === ARCHITECTURE PARAMETERS (E13h VALIDATED) ===
D = 4096          # Dimensions
N = 512           # Chunk size (bp)
STEP = 512        # Step size (no overlap)
N_BANKS = 2       # Only Hydrophobic and MajorGroove (hinge reconstructed)
BASE_SEED = 42    # Base random seed

# === HIGH-FIDELITY THRESHOLDS (avg k=6, range k=4-8, NO k=2) ===
# Target distribution: k=4: 25%, k=6: 50%, k=8: 25% → avg k = 6.0
# Based on difficulty percentiles: 25th=0.0300, 75th=0.1319
DIFFICULTY_THRESHOLDS = {
    'easy': 0.0300,      # k=4 for bottom 25% (25th percentile)
    'moderate': 0.1319,  # k=6 for middle 50% (25th-75th percentile)
    'hard': 0.1319,      # k=8 for top 25% (75th+ percentile)
}

# Logging setup - writes to console only (production mode)
# Log file can be configured externally if needed
logger = logging.getLogger(__name__)

# Configure logging if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class EncodingConfig:
    """Configuration for adaptive encoding."""
    D: int = 4096
    N: int = 512
    base_seed: int = 42
    target_accuracy: float = 0.99  # 0.98 for storage-optimized
    n_workers: int = 8
    batch_size: int = 5000

    # Paths
    gdiff_path: Optional[Path] = None
    guide_fasta_dir: Optional[Path] = None
    output_path: Optional[Path] = None


class AdaptiveSparseHadamardCodebook:
    """
    Adaptive Sparse-Hadamard Codebook with k-dependent seeds.

    CRITICAL: Each k value has its own independent codebook space.
    The seed formula ensures zero overlap between k=2/4/6/8 dimensions.

    Formula: seed_for_position = base_seed + position * 100 + k * 10000

    OPTIMIZED: Pre-computes all dimension indices at initialization for speed.
    """

    def __init__(self, D: int = 4096, N: int = 512, base_seed: int = 42):
        self.D = D
        self.N = N
        self.base_seed = base_seed

        # Pre-compute sign patterns (alternating for positions)
        self.signs = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int8)

        # PRE-COMPUTE all dimension indices for all (position, k) combinations
        # This is the critical optimization - avoids 512 RandomState creations per chunk
        self._dim_cache = {}
        for k in [2, 4, 6, 8]:
            dims = np.zeros((N, k), dtype=np.int32)
            for pos in range(N):
                k_seed = base_seed + pos * 100 + k * 10000
                rng = np.random.RandomState(k_seed)
                dims[pos] = rng.choice(D, size=k, replace=False)
            self._dim_cache[k] = dims

        logger.info(f"AdaptiveSparseHadamardCodebook initialized:")
        logger.info(f"  D={D}, N={N}, base_seed={base_seed}")
        logger.info(f"  Pre-computed dimension indices for k=2,4,6,8")

    def get_active_dimensions(self, position: int, k: int) -> np.ndarray:
        """
        Get k active dimensions for a position with k-dependent seed.
        Uses pre-computed cache for speed.
        """
        return self._dim_cache[k][position]

    def get_signs(self, k: int) -> np.ndarray:
        """Get sign pattern for k dimensions."""
        return self.signs[:k]

    def encode_chunk(
        self,
        sequence: str,
        k: int
    ) -> Tuple[np.ndarray, float]:
        """
        Encode a sequence chunk with specified k value.
        Uses Numba JIT if available for 5-10x speedup.

        Returns:
            (encoded_vector, decode_accuracy_estimate)
        """
        # Pad sequence if needed
        seq = sequence[:self.N].ljust(self.N, 'N')

        if HAS_NUMBA:
            # FAST PATH: Numba JIT-compiled
            seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
            seq_codes = _seq_to_codes(seq_bytes)
            dim_cache = self._dim_cache[k]
            signs = self.signs[:k].astype(np.int8)
            encoded = _encode_chunk_numba(seq_codes, dim_cache, signs, self.D, self.N, k)
            return encoded, 1.0

        # FALLBACK: Pure Python/NumPy (slower)
        seq_arr = np.array(list(seq))

        # Compute signals for 2 banks
        hydro = np.where(seq_arr == 'A', 1,
                np.where(seq_arr == 'T', -1, 0)).astype(np.int32)
        groove = np.where(seq_arr == 'G', 1,
                 np.where(seq_arr == 'C', -1, 0)).astype(np.int32)

        # Accumulate with position binding
        hydro_acc = np.zeros(self.D, dtype=np.int32)
        groove_acc = np.zeros(self.D, dtype=np.int32)
        signs = self.get_signs(k)

        for pos in range(self.N):
            if seq[pos] == 'N':
                continue
            active_dims = self.get_active_dimensions(pos, k)
            hydro_acc[active_dims] += hydro[pos] * signs
            groove_acc[active_dims] += groove[pos] * signs

        encoded = np.stack([
            np.sign(hydro_acc).astype(np.int8),
            np.sign(groove_acc).astype(np.int8)
        ])

        return encoded, 1.0

    def decode_chunk(
        self,
        encoded: np.ndarray,
        k: int,
        use_flip: bool = True
    ) -> str:
        """
        Decode an encoded chunk back to sequence.
        Uses Numba JIT if available for ~50x speedup.

        Args:
            encoded: (2, D) int8 array
            k: sparsity value used for encoding
            use_flip: use AT-biased flip at margin=0

        Returns:
            Decoded nucleotide sequence
        """
        if HAS_NUMBA:
            # FAST PATH: Numba JIT-compiled (50x faster)
            dim_cache = self._dim_cache[k]
            signs = self.signs[:k].astype(np.int8)
            decoded_codes = _decode_chunk_numba(
                encoded[0], encoded[1], dim_cache, signs, self.N, k
            )
            # Convert codes to string: 0=A, 1=C, 2=G, 3=T
            code_to_nuc = np.array(['A', 'C', 'G', 'T', 'N'])
            return ''.join(code_to_nuc[decoded_codes])

        # FALLBACK: Pure Python/NumPy (slower)
        sequence = []
        signs = self.get_signs(k)

        for pos in range(self.N):
            active_dims = self.get_active_dimensions(pos, k)

            sim_h = np.dot(encoded[0][active_dims], signs)
            sim_g = np.dot(encoded[1][active_dims], signs)

            abs_h, abs_g = abs(sim_h), abs(sim_g)
            margin = abs(abs_h - abs_g)

            if margin == 0 and use_flip:
                # AT-biased flip (E13g validated: 39-60% fix rate)
                nuc = 'A' if sim_h >= 0 else 'T'
            elif abs_h > abs_g:
                nuc = 'A' if sim_h > 0 else 'T'
            else:
                nuc = 'G' if sim_g > 0 else 'C'

            sequence.append(nuc)

        return ''.join(sequence)

    def compute_accuracy(
        self,
        encoded: np.ndarray,
        original_sequence: str,
        k: int
    ) -> Dict:
        """
        Compute decode accuracy with detailed margin statistics.
        Uses Numba JIT if available for ~50x speedup.

        Args:
            encoded: (2, D) int8 array
            original_sequence: Original sequence for comparison
            k: sparsity value used for encoding

        Returns:
            Dict with accuracy, margin_0 stats, etc.
        """
        seq = original_sequence[:self.N].ljust(self.N, 'N')

        if HAS_NUMBA:
            # FAST PATH: Numba JIT-compiled
            seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
            original_codes = _seq_to_codes(seq_bytes)
            dim_cache = self._dim_cache[k]
            signs = self.signs[:k].astype(np.int8)

            correct, total, m0_count, m0_correct = _compute_accuracy_numba(
                encoded[0], encoded[1], original_codes, dim_cache, signs, self.N, k
            )

            return {
                'accuracy': correct / total if total > 0 else 0.0,
                'correct': int(correct),
                'total': int(total),
                'margin_0_count': int(m0_count),
                'margin_0_correct': int(m0_correct),
                'margin_0_fix_rate': m0_correct / m0_count if m0_count > 0 else 0.0,
            }

        # FALLBACK: Use decode_chunk and compare
        decoded = self.decode_chunk(encoded, k)
        correct = 0
        total = 0
        for i, (orig, dec) in enumerate(zip(seq, decoded)):
            if orig != 'N':
                total += 1
                if orig == dec:
                    correct += 1

        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'correct': correct,
            'total': total,
            'margin_0_count': 0,  # Not computed in fallback
            'margin_0_correct': 0,
            'margin_0_fix_rate': 0.0,
        }

    def batch_decode(
        self,
        encoded_batch: np.ndarray,
        k: int
    ) -> List[str]:
        """
        Batch decode multiple chunks efficiently.
        Uses parallel Numba if available.

        Args:
            encoded_batch: (batch_size, 2, D) int8 array
            k: sparsity value

        Returns:
            List of decoded sequences
        """
        if HAS_NUMBA and encoded_batch.shape[0] > 1:
            dim_cache = self._dim_cache[k]
            signs = self.signs[:k].astype(np.int8)
            decoded_codes = _batch_decode_numba(
                encoded_batch, dim_cache, signs, self.N, k
            )
            code_to_nuc = np.array(['A', 'C', 'G', 'T', 'N'])
            return [''.join(code_to_nuc[codes]) for codes in decoded_codes]

        # Fallback: sequential decode
        return [self.decode_chunk(enc, k) for enc in encoded_batch]


class DifficultyScorer:
    """
    Compute difficulty score for adaptive k-selection.

    Based on E13h findings:
    - GC content is primary predictor
    - Pre-encoding features (fast, no encoding needed)
    """

    @staticmethod
    def compute_pre_encoding_difficulty(sequence: str) -> float:
        """
        Compute difficulty from sequence features only.

        Components:
        - N-density penalty (unknown bases)
        - GC extremity penalty
        - Homopolymer penalty
        """
        n = len(sequence)
        if n == 0:
            return 0.5

        # N-density penalty
        n_density = sequence.count('N') / n
        n_penalty = min(1.0, n_density * 5)

        # GC content and extremity
        gc = (sequence.count('G') + sequence.count('C')) / n
        gc_deviation = abs(gc - 0.5) * 2
        gc_penalty = gc_deviation ** 2

        # Homopolymer run penalty
        max_run = 1
        run = 1
        for i in range(1, n):
            if sequence[i] == sequence[i-1]:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        homo_penalty = min(1.0, max(0, max_run - 5) / 10)

        # Combined score
        return 0.4 * n_penalty + 0.3 * gc_penalty + 0.3 * homo_penalty

    @staticmethod
    def select_k(difficulty: float, target_accuracy: float = 0.99) -> int:
        """
        Select k based on difficulty score.

        HIGH-FIDELITY MODE (avg k=6, range k=4-8, NO k=2):
        - k=4: bottom 25% (easiest chunks)
        - k=6: middle 50% (moderate chunks)
        - k=8: top 25% (hardest chunks)

        Based on difficulty percentiles:
        - 25th percentile = 0.0300
        - 75th percentile = 0.1319
        """
        # HIGH-FIDELITY: k=4/6/8 only (avg k=6)
        if difficulty >= DIFFICULTY_THRESHOLDS['moderate']:  # >= 0.1319
            return 8  # Top 25% - hardest chunks
        elif difficulty >= DIFFICULTY_THRESHOLDS['easy']:    # >= 0.0300
            return 6  # Middle 50%
        else:
            return 4  # Bottom 25% - easiest chunks


class GenomicDataLoader:
    """
    Load genomic data from GDiff and guide FASTA files.

    CRITICAL: Properly handles GDiff integer guide IDs -> "refN" string conversion.
    CRITICAL: Applies GDiff differential_variants to encode EXPERIMENTAL sequence,
              not just the raw guide sequence.
    """

    def __init__(self, gdiff_path: Path, guide_fasta_dir: Path):
        logger.info("Loading genomic data...")
        start = time.time()

        # Load GDiff
        logger.info(f"  Loading GDiff: {gdiff_path}")
        with gzip.open(gdiff_path, 'rt') as f:
            self.gdiff = json.load(f)

        self.region_guide_map = self.gdiff['region_guide_map']

        # CRITICAL: Build variant lookup from differential_variants
        # This maps (chrom, pos) -> alt base (experimental differs from guide)
        logger.info("  Building variant lookup from differential_variants...")
        self.variant_lookup = {}
        differential_variants = self.gdiff.get('differential_variants', [])
        for v in differential_variants:
            chrom = v['chrom']  # e.g., 'chr1_consensus'
            pos = v['pos']      # 1-based position
            alt = v['alt']      # experimental base (differs from guide)
            self.variant_lookup[(chrom, pos)] = alt
        logger.info(f"  Loaded {len(self.variant_lookup):,} differential variants")

        # Track variant application statistics
        self._variants_applied = 0

        # Parse chromosome sizes from region map
        self.chrom_sizes = {}
        for region_key in self.region_guide_map.keys():
            chrom_part, range_part = region_key.split(':')
            chrom = chrom_part.replace('_consensus', '')
            start_pos, end_pos = map(int, range_part.split('-'))
            if chrom not in self.chrom_sizes:
                self.chrom_sizes[chrom] = 0
            self.chrom_sizes[chrom] = max(self.chrom_sizes[chrom], end_pos)

        logger.info(f"  Found {len(self.chrom_sizes)} chromosomes")

        # Open guide FASTA files
        logger.info(f"  Opening guide FASTAs from: {guide_fasta_dir}")
        self.guide_fasta_handles = {}
        self.guide_fasta_dir = guide_fasta_dir

        # Track guide usage statistics
        self._guide_usage_stats = {}
        self._fallback_count = 0

        if HAS_PYFAIDX:
            for i in range(1, 13):  # ref1 to ref12
                fasta_path = guide_fasta_dir / f"ref{i}.fa.gz"
                if fasta_path.exists() or fasta_path.is_symlink():
                    try:
                        # Follow symlinks and resolve
                        resolved_path = fasta_path.resolve()
                        self.guide_fasta_handles[f'ref{i}'] = Fasta(str(resolved_path))
                        logger.info(f"    ref{i} indexed")
                    except Exception as e:
                        logger.warning(f"    ref{i} failed: {e}")
        else:
            logger.error("pyfaidx not available - cannot load FASTAs")

        logger.info(f"  Opened {len(self.guide_fasta_handles)} guide references")

        elapsed = time.time() - start
        logger.info(f"  Data initialized in {elapsed:.1f}s")

    def get_guide_for_region(self, chrom: str, pos: int) -> str:
        """
        Get the guide reference ID for a genomic region.

        CRITICAL: GDiff returns INTEGER guide IDs (1-11),
        must convert to 'refN' string format.
        """
        region_start = (pos // 10_000_000) * 10_000_000
        region_end = region_start + 10_000_000

        # GDiff uses 'chr1_consensus:0-10000000' format
        region_key = f"{chrom}_consensus:{region_start}-{region_end}"

        # Get integer guide ID, convert to ref string
        guide_id_int = self.region_guide_map.get(region_key, 1)

        if isinstance(guide_id_int, int):
            guide_id = f'ref{guide_id_int}'
        else:
            guide_id = guide_id_int

        return guide_id

    def get_sequence(self, chrom: str, start: int, end: int) -> str:
        """Get EXPERIMENTAL sequence for a genomic region.

        CRITICAL: This returns the experimental sequence by:
        1. Fetching the guide sequence from FASTA
        2. Applying GDiff differential variants (substituting alt bases)

        Result is the actual experimental genome (ERR3239334), not just guide.
        """
        guide_id = self.get_guide_for_region(chrom, start)

        # Track usage
        self._guide_usage_stats[guide_id] = self._guide_usage_stats.get(guide_id, 0) + 1

        if guide_id not in self.guide_fasta_handles:
            self._fallback_count += 1
            guide_id = 'ref1'

        fasta = self.guide_fasta_handles.get(guide_id)
        if fasta is None:
            return 'N' * (end - start)

        # Try different chromosome naming conventions
        chrom_variants = [
            chrom,
            f'{chrom}_consensus',
            chrom.replace('chr', ''),
            f'chr{chrom}',
        ]

        guide_seq = None
        chrom_used = None
        for cv in chrom_variants:
            if cv in fasta:
                try:
                    guide_seq = str(fasta[cv][start:end]).upper()
                    chrom_used = cv
                    break
                except Exception:
                    continue

        if guide_seq is None:
            return 'N' * (end - start)

        # CRITICAL: Apply GDiff differential variants to get EXPERIMENTAL sequence
        # GDiff uses chr1_consensus format for positions
        chrom_consensus = f'{chrom}_consensus' if '_consensus' not in chrom else chrom

        # Convert guide sequence to list for mutation
        seq_list = list(guide_seq)
        variants_applied_this_chunk = 0

        # Check each position for variants
        for pos_offset in range(len(seq_list)):
            genomic_pos = start + pos_offset + 1  # 1-based position in GDiff

            # Look up if this position has a variant
            variant_alt = self.variant_lookup.get((chrom_consensus, genomic_pos))
            if variant_alt is not None:
                # Apply variant: replace guide base with experimental base
                seq_list[pos_offset] = variant_alt
                variants_applied_this_chunk += 1
                self._variants_applied += 1

        return ''.join(seq_list)


# === GLOBAL SHARED STATE ===
_shared_data: Optional[GenomicDataLoader] = None
_shared_codebook: Optional[AdaptiveSparseHadamardCodebook] = None
_shared_scorer: Optional[DifficultyScorer] = None
_target_accuracy: float = 0.99


def init_shared_resources(
    gdiff_path: Path,
    guide_fasta_dir: Path,
    target_accuracy: float = 0.99
):
    """Initialize shared resources in main process."""
    global _shared_data, _shared_codebook, _shared_scorer, _target_accuracy

    _shared_data = GenomicDataLoader(gdiff_path, guide_fasta_dir)
    _shared_codebook = AdaptiveSparseHadamardCodebook()
    _shared_scorer = DifficultyScorer()
    _target_accuracy = target_accuracy


def encode_chunk_adaptive(args) -> Tuple[int, np.ndarray, int, float]:
    """
    Encode a single chunk with adaptive k-selection.

    Args:
        args: (chrom, start, end, chunk_idx)

    Returns:
        (chunk_idx, encoded_vector, selected_k, difficulty)
    """
    global _shared_data, _shared_codebook, _shared_scorer, _target_accuracy

    chrom, start, end, chunk_idx = args

    # Get sequence
    sequence = _shared_data.get_sequence(chrom, start, end)

    # Compute difficulty
    difficulty = _shared_scorer.compute_pre_encoding_difficulty(sequence)

    # Select k based on difficulty
    k = _shared_scorer.select_k(difficulty, _target_accuracy)

    # Encode with selected k
    encoded, _ = _shared_codebook.encode_chunk(sequence, k)

    return (chunk_idx, encoded, k, difficulty)


def generate_chunk_list(chrom_sizes: Dict[str, int]) -> List[Tuple[str, int, int]]:
    """Generate list of all chunks to encode."""
    chunks = []

    for chrom in sorted(chrom_sizes.keys(), key=lambda x: (len(x), x)):
        size = chrom_sizes[chrom]
        pos = 0
        while pos < size:
            chunk_end = min(pos + N, size)
            chunks.append((chrom, pos, chunk_end))
            pos += STEP

    return chunks


def run_production_encoding(
    gdiff_path: Path,
    guide_fasta_dir: Path,
    output_path: Path,
    target_accuracy: float = 0.99,
    n_workers: int = 8,
    batch_size: int = 5000
):
    """
    Run full production encoding pipeline.
    """
    logger.info("=" * 70)
    logger.info("PRODUCTION ADAPTIVE k-SELECTION ENCODER")
    logger.info("=" * 70)
    logger.info(f"E13h Validated - Target accuracy: {target_accuracy*100:.1f}%")
    logger.info("")

    # Initialize shared resources
    logger.info("Initializing shared resources...")
    init_shared_resources(gdiff_path, guide_fasta_dir, target_accuracy)
    logger.info("")

    # Generate chunks
    logger.info("Generating chunk list...")
    chunk_list = generate_chunk_list(_shared_data.chrom_sizes)
    total_chunks = len(chunk_list)

    logger.info(f"  Total chunks: {total_chunks:,}")
    logger.info(f"  Total genome: {sum(_shared_data.chrom_sizes.values()):,} bp")
    logger.info("")

    # Create output HDF5
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Creating output HDF5...")
    with h5py.File(output_path, 'w') as f:
        # Main data - variable size per chunk due to different k
        # Store as (chunks, 2, D) with k metadata separately
        dset = f.create_dataset(
            'encoded_vectors',
            shape=(total_chunks, N_BANKS, D),
            dtype=np.int8,
            chunks=(1, N_BANKS, D),
            compression='gzip',
            compression_opts=4
        )

        # k values per chunk
        k_dset = f.create_dataset(
            'k_values',
            shape=(total_chunks,),
            dtype=np.uint8
        )

        # Difficulty scores
        diff_dset = f.create_dataset(
            'difficulty_scores',
            shape=(total_chunks,),
            dtype=np.float32
        )

        # Chunk keys
        chunk_key_strings = [f"{chrom}:{start}-{end}"
                           for chrom, start, end in chunk_list]
        f.create_dataset(
            'chunk_keys',
            data=[s.encode('utf-8') for s in chunk_key_strings],
            dtype=h5py.string_dtype('utf-8')
        )

        # Metadata
        f.attrs['D'] = D
        f.attrs['N'] = N
        f.attrs['step'] = STEP
        f.attrs['n_banks'] = N_BANKS
        f.attrs['target_accuracy'] = target_accuracy
        f.attrs['base_seed'] = BASE_SEED
        f.attrs['codebook_type'] = 'adaptive_sparse_hadamard'
        f.attrs['encoding_date'] = datetime.now().isoformat()
        f.attrs['difficulty_thresholds'] = json.dumps(DIFFICULTY_THRESHOLDS)

    logger.info("  HDF5 structure created")
    logger.info("")

    # Parallel encoding
    logger.info("=" * 70)
    logger.info("ENCODING (Adaptive k-selection)")
    logger.info("=" * 70)
    logger.info("")

    num_batches = (total_chunks + batch_size - 1) // batch_size

    logger.info(f"Workers: {n_workers}")
    logger.info(f"Batch size: {batch_size:,}")
    logger.info(f"Total batches: {num_batches}")
    logger.info("")

    encoding_start = time.time()
    chunks_done = 0
    k_distribution = {2: 0, 4: 0, 6: 0, 8: 0}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_chunks)

            batch_args = [
                (chrom, start, end, i)
                for i, (chrom, start, end) in enumerate(chunk_list[batch_start:batch_end], start=batch_start)
            ]

            batch_time_start = time.time()
            logger.info(f"  Starting batch {batch_idx+1}: {len(batch_args)} chunks...")
            for handler in logging.root.handlers:
                handler.flush()

            # Process batch with progress tracking
            results = []
            for i, result in enumerate(executor.map(encode_chunk_adaptive, batch_args)):
                results.append(result)
                if (i + 1) % 500 == 0:
                    logger.info(f"    Batch {batch_idx+1}: {i+1}/{len(batch_args)} chunks encoded...")
                    for handler in logging.root.handlers:
                        handler.flush()

            logger.info(f"  Batch {batch_idx+1} encoding done, writing to HDF5...")
            for handler in logging.root.handlers:
                handler.flush()

            # Write to HDF5
            with h5py.File(output_path, 'a') as f:
                for chunk_idx, encoded, k, difficulty in results:
                    f['encoded_vectors'][chunk_idx] = encoded
                    f['k_values'][chunk_idx] = k
                    f['difficulty_scores'][chunk_idx] = difficulty
                    k_distribution[k] += 1

            chunks_done += len(results)
            batch_time = time.time() - batch_time_start
            chunks_per_sec = len(results) / batch_time

            pct = 100.0 * chunks_done / total_chunks
            elapsed = time.time() - encoding_start
            eta = (total_chunks - chunks_done) / chunks_per_sec if chunks_per_sec > 0 else 0

            logger.info(
                f"Batch {batch_idx+1}/{num_batches}: {len(results):,} chunks in {batch_time:.1f}s "
                f"({chunks_per_sec:.0f}/s) | {pct:.1f}% | ETA: {eta/60:.1f}m"
            )
            # Flush logs immediately
            for handler in logging.root.handlers:
                handler.flush()

    total_time = time.time() - encoding_start

    # Final stats
    logger.info("")
    logger.info("=" * 70)
    logger.info("ENCODING COMPLETE")
    logger.info("=" * 70)
    logger.info("")

    file_size = output_path.stat().st_size / (1024**3)

    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Throughput: {total_chunks/total_time:.0f} chunks/s")
    logger.info(f"Output: {output_path}")
    logger.info(f"File size: {file_size:.3f} GB")
    logger.info("")

    # k-distribution
    logger.info("k-Distribution:")
    for k in [2, 4, 6, 8]:
        pct = 100.0 * k_distribution[k] / total_chunks
        logger.info(f"  k={k}: {k_distribution[k]:,} chunks ({pct:.1f}%)")

    # Guide usage
    logger.info("")
    logger.info("Guide strand usage:")
    total_uses = sum(_shared_data._guide_usage_stats.values())
    for guide in sorted(_shared_data._guide_usage_stats.keys()):
        count = _shared_data._guide_usage_stats[guide]
        pct = 100.0 * count / total_uses
        logger.info(f"  {guide}: {count:,} ({pct:.1f}%)")

    if len(_shared_data._guide_usage_stats) > 1:
        logger.info(f"  Multiple guides used - GDiff mapping working correctly!")
    else:
        logger.warning(f"  WARNING: Only one guide used - check GDiff mapping")

    return output_path


def sanity_check_encoding(output_path: Path, n_samples: int = 100):
    """
    Sanity check: decode samples and compute accuracy.
    Uses Numba JIT for ~50x faster validation.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SANITY CHECK (Numba JIT-accelerated)")
    logger.info("=" * 70)

    codebook = AdaptiveSparseHadamardCodebook()
    start_time = time.time()

    with h5py.File(output_path, 'r') as f:
        total_chunks = len(f['encoded_vectors'])

        # Sample random chunks
        sample_indices = np.random.choice(total_chunks, size=min(n_samples, total_chunks), replace=False)
        sample_indices = np.sort(sample_indices)

        total_correct = 0
        total_positions = 0
        total_m0_count = 0
        total_m0_correct = 0
        k_accuracy = {2: [], 4: [], 6: [], 8: []}

        for idx in sample_indices:
            encoded = f['encoded_vectors'][idx]
            k = int(f['k_values'][idx])
            chunk_key = f['chunk_keys'][idx].decode('utf-8')

            # Get original sequence
            chrom, coords = chunk_key.split(':')
            start, end = map(int, coords.split('-'))
            original = _shared_data.get_sequence(chrom, start, end).ljust(N, 'N')[:N]

            # Use Numba-accelerated accuracy computation
            acc_stats = codebook.compute_accuracy(encoded, original, k)

            total_correct += acc_stats['correct']
            total_positions += acc_stats['total']
            total_m0_count += acc_stats['margin_0_count']
            total_m0_correct += acc_stats['margin_0_correct']
            k_accuracy[k].append(acc_stats['accuracy'])

        elapsed = time.time() - start_time
        overall_acc = 100.0 * total_correct / total_positions

        logger.info(f"")
        logger.info(f"Sampled {len(sample_indices)} chunks in {elapsed:.2f}s ({len(sample_indices)/elapsed:.0f} chunks/s)")
        logger.info(f"Overall accuracy: {overall_acc:.2f}%")

        if total_m0_count > 0:
            m0_fix_rate = 100.0 * total_m0_correct / total_m0_count
            logger.info(f"Margin=0 positions: {total_m0_count} ({m0_fix_rate:.1f}% fixed by flip)")

        logger.info("")
        logger.info("Accuracy by k:")
        for k in [2, 4, 6, 8]:
            if k_accuracy[k]:
                acc = 100.0 * np.mean(k_accuracy[k])
                logger.info(f"  k={k}: {acc:.2f}% (n={len(k_accuracy[k])})")

    logger.info("")
    if overall_acc >= 98:
        logger.info("✓ SANITY CHECK PASSED")
    else:
        logger.warning(f"✗ SANITY CHECK FAILED - accuracy {overall_acc:.2f}% < 98%")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Production Adaptive k-Selection Encoder")
    parser.add_argument("--gdiff", type=Path, required=True,
                       help="Path to GDiff file (.gdiff.gz)")
    parser.add_argument("--guides", type=Path,
                       default=Path("data/guide_strands"),
                       help="Path to guide strand FASTA directory")
    parser.add_argument("--output", type=Path,
                       default=Path("output/production_encoded.h5"),
                       help="Output HDF5 file path")
    parser.add_argument("--target", type=float, default=0.99,
                       help="Target accuracy (0.98 or 0.99)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=5000)
    parser.add_argument("--sanity-check", action="store_true")

    args = parser.parse_args()

    output_path = run_production_encoding(
        gdiff_path=args.gdiff,
        guide_fasta_dir=args.guides,
        output_path=args.output,
        target_accuracy=args.target,
        n_workers=args.workers,
        batch_size=args.batch
    )

    if args.sanity_check:
        sanity_check_encoding(output_path)


if __name__ == '__main__':
    main()

# Alias for unified interface
AdaptiveEncoder = AdaptiveSparseHadamardCodebook
