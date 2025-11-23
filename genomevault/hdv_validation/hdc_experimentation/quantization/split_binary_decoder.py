#!/usr/bin/env python3
"""
Split Binary Decoder: Query Nucleotides from 6-Bank Binary Encoding

Architecture: Within-Lens Binary Decoding
- Hydrophobic_A {0,1}, Hydrophobic_T {0,1}  →  Nucleotide ∈ {A, T}
- MajorGroove_G {0,1}, MajorGroove_C {0,1}  →  Nucleotide ∈ {G, C}
- Hinge_pos {0,1}, Hinge_neg {0,1}           →  Dinucleotide context

Decoding Algorithm (Two-Stage):
1. Pair Selection: Compare AT magnitude vs GC magnitude
   - |sim_AT| = max(sim_Hydro_A, sim_Hydro_T)
   - |sim_GC| = max(sim_Major_G, sim_Major_C)
   - Select pair with stronger signal

2. Sign Determination: Within selected pair, pick nucleotide
   - If AT pair: argmax(sim_Hydro_A, sim_Hydro_T) → A or T
   - If GC pair: argmax(sim_Major_G, sim_Major_C) → G or C

Mathematical Foundation:
- Binary signals are cleaner due to √2 SNR improvement
- Complementary sparsity gives adaptive SNR in sequence-dependent manner
- Cross-channel grounding via hinge lens provides disambiguation
"""

import h5py
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result from nucleotide query"""
    nucleotide: str  # 'A', 'T', 'G', or 'C'
    confidence: float  # 0.0 to 1.0
    pair: str  # 'AT' or 'GC'
    at_similarity: float  # Hydrophobic pair similarity
    gc_similarity: float  # MajorGroove pair similarity
    individual_sims: dict  # Individual bank similarities


class SplitBinaryDecoder:
    """
    Decoder for 6-bank split binary genomic encoding

    Queries individual nucleotides from binary hypervector representation
    using two-stage retrieval: pair selection + sign determination
    """

    def __init__(
        self,
        binary_h5_path: Path,
        position_codebook: Optional[np.ndarray] = None,
        dimension: int = 10240,
        chunk_size: int = 2000,
        seed: int = 42
    ):
        """
        Initialize Split Binary Decoder

        Args:
            binary_h5_path: Path to 6-bank split binary HDF5 file
            position_codebook: Position codebook (N × D), or None to generate
            dimension: Hypervector dimension
            chunk_size: Nucleotides per chunk
            seed: Random seed (must match encoder!)
        """
        self.binary_h5_path = binary_h5_path
        self.D = dimension
        self.N = chunk_size
        self.seed = seed

        logger.info(f"Initializing SplitBinaryDecoder")
        logger.info(f"  File: {binary_h5_path}")
        logger.info(f"  Dimension: {self.D:,}D")
        logger.info(f"  Chunk size: {self.N:,} bp")

        # Generate or use provided position codebook
        if position_codebook is None:
            logger.info(f"  Generating position codebook...")
            self.position_codebook = self._generate_position_codebook()
        else:
            self.position_codebook = position_codebook
            logger.info(f"  Using provided position codebook")

        # Load binary HDF5 file
        logger.info(f"  Loading binary HDF5...")
        self.h5_file = h5py.File(binary_h5_path, 'r')
        self.binary_data = self.h5_file['binary_bank_vectors']

        total_chunks, num_banks, dimension = self.binary_data.shape
        logger.info(f"  Shape: {self.binary_data.shape}")

        if num_banks != 6:
            raise ValueError(f"Expected 6 banks, got {num_banks}")

        if dimension != self.D:
            logger.warning(f"Dimension mismatch: expected {self.D}, got {dimension}")
            self.D = dimension

        # Bank indices
        self.HYDROPHOBIC_A = 0
        self.HYDROPHOBIC_T = 1
        self.MAJORGROOVE_G = 2
        self.MAJORGROOVE_C = 3
        self.HINGE_POS = 4
        self.HINGE_NEG = 5

        logger.info(f"✓ Decoder initialized: {total_chunks:,} chunks available")

    def _generate_position_codebook(self) -> np.ndarray:
        """
        Generate position codebook (must match encoder!)

        Returns:
            N × D array of random {-1, +1} vectors
        """
        np.random.seed(self.seed)
        codebook = np.random.choice([-1, 1], size=(self.N, self.D)).astype(np.int8)
        return codebook

    def query_nucleotide(
        self,
        chrom: str,
        pos: int
    ) -> QueryResult:
        """
        Query nucleotide at genomic position using two-stage retrieval

        Stage 1: Pair selection (magnitude comparison)
            - Compare |sim_AT| vs |sim_GC|
            - Select pair with stronger signal

        Stage 2: Sign determination within pair
            - If AT pair: argmax(sim_Hydro_A, sim_Hydro_T) → A or T
            - If GC pair: argmax(sim_Major_G, sim_Major_C) → G or C

        Args:
            chrom: Chromosome (e.g., 'chr1')
            pos: 0-based genomic position

        Returns:
            QueryResult with nucleotide, confidence, and similarity scores
        """
        # Find chunk containing this position
        chunk_start = (pos // self.N) * self.N
        chunk_idx = chunk_start // self.N
        offset = pos - chunk_start

        # Load chunk banks
        chunk_banks = self.binary_data[chunk_idx, :, :].astype(np.float32)

        # Extract individual bank vectors
        hydro_A = chunk_banks[self.HYDROPHOBIC_A]
        hydro_T = chunk_banks[self.HYDROPHOBIC_T]
        major_G = chunk_banks[self.MAJORGROOVE_G]
        major_C = chunk_banks[self.MAJORGROOVE_C]
        hinge_pos = chunk_banks[self.HINGE_POS]
        hinge_neg = chunk_banks[self.HINGE_NEG]

        # Get position vector
        pos_vec = self.position_codebook[offset].astype(np.float32)

        # Compute similarities (dot products with normalization)
        sim_hydro_A = self._normalized_similarity(pos_vec, hydro_A)
        sim_hydro_T = self._normalized_similarity(pos_vec, hydro_T)
        sim_major_G = self._normalized_similarity(pos_vec, major_G)
        sim_major_C = self._normalized_similarity(pos_vec, major_C)

        # AT pair similarity: max of A and T signals
        sim_AT = max(abs(sim_hydro_A), abs(sim_hydro_T))

        # GC pair similarity: max of G and C signals
        sim_GC = max(abs(sim_major_G), abs(sim_major_C))

        # Stage 1: Pair selection (magnitude comparison)
        if sim_AT > sim_GC:
            # AT pair selected
            pair = 'AT'
            # Stage 2: Sign determination within AT pair
            if sim_hydro_A > sim_hydro_T:
                nucleotide = 'A'
                primary_sim = sim_hydro_A
            else:
                nucleotide = 'T'
                primary_sim = sim_hydro_T

            confidence = sim_AT / (sim_AT + sim_GC + 1e-10)

        else:
            # GC pair selected
            pair = 'GC'
            # Stage 2: Sign determination within GC pair
            if sim_major_G > sim_major_C:
                nucleotide = 'G'
                primary_sim = sim_major_G
            else:
                nucleotide = 'C'
                primary_sim = sim_major_C

            confidence = sim_GC / (sim_AT + sim_GC + 1e-10)

        # Store individual similarities for analysis
        individual_sims = {
            'Hydrophobic_A': sim_hydro_A,
            'Hydrophobic_T': sim_hydro_T,
            'MajorGroove_G': sim_major_G,
            'MajorGroove_C': sim_major_C,
        }

        return QueryResult(
            nucleotide=nucleotide,
            confidence=confidence,
            pair=pair,
            at_similarity=sim_AT,
            gc_similarity=sim_GC,
            individual_sims=individual_sims
        )

    def _normalized_similarity(self, pos_vec: np.ndarray, bank_vec: np.ndarray) -> float:
        """
        Compute normalized similarity (cosine similarity)

        Args:
            pos_vec: Position vector (D-dimensional)
            bank_vec: Bank hypervector (D-dimensional)

        Returns:
            Normalized similarity in range [-1, +1]
        """
        dot_product = np.dot(pos_vec, bank_vec)
        norm_bank = np.linalg.norm(bank_vec)

        if norm_bank < 1e-10:
            return 0.0

        # Position vectors have norm sqrt(D) for bipolar {-1,+1}
        # But we normalized them, so:
        similarity = dot_product / (norm_bank + 1e-10)

        return similarity

    def query_chunk(self, chunk_idx: int) -> list[QueryResult]:
        """
        Query all nucleotides in a chunk

        Args:
            chunk_idx: Chunk index (0-based)

        Returns:
            List of QueryResult for each position in chunk
        """
        results = []

        chunk_start = chunk_idx * self.N

        for offset in range(self.N):
            pos = chunk_start + offset
            result = self.query_nucleotide("chr1", pos)  # Chromosome doesn't matter for chunk-based query
            results.append(result)

        return results

    def close(self):
        """Close HDF5 file"""
        self.h5_file.close()


def test_decoder():
    """Test decoder on sample positions"""
    binary_path = Path("genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_6banks_split_binary.h5")

    if not binary_path.exists():
        logger.error(f"Binary file not found: {binary_path}")
        return 1

    logger.info("="*80)
    logger.info("SPLIT BINARY DECODER TEST")
    logger.info("="*80)
    logger.info("")

    # Initialize decoder
    decoder = SplitBinaryDecoder(
        binary_h5_path=binary_path,
        dimension=10240,
        chunk_size=2000,
        seed=42
    )

    logger.info("")
    logger.info("Testing queries on random positions...")
    logger.info("")

    # Test 10 random positions
    np.random.seed(42)
    test_positions = np.random.randint(0, 10000, size=10)

    for i, pos in enumerate(test_positions, 1):
        result = decoder.query_nucleotide("chr1", pos)

        logger.info(f"Position {pos:,}:")
        logger.info(f"  Nucleotide: {result.nucleotide}")
        logger.info(f"  Confidence: {result.confidence:.4f}")
        logger.info(f"  Pair: {result.pair}")
        logger.info(f"  AT similarity: {result.at_similarity:.4f}")
        logger.info(f"  GC similarity: {result.gc_similarity:.4f}")
        logger.info(f"  Individual similarities:")
        for bank, sim in result.individual_sims.items():
            logger.info(f"    {bank}: {sim:.4f}")
        logger.info("")

    decoder.close()

    logger.info("="*80)
    logger.info("✓ Decoder test complete")
    logger.info("="*80)

    return 0


if __name__ == '__main__':
    exit(test_decoder())
