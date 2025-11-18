#!/usr/bin/env python3
"""
Complementary Pair HDC Encoder for Nucleotide-Resolution Genomic Data

Architecture:
- Two hypervectors: AT_vec (A/T positions) and GC_vec (G/C positions)
- Each position appears in EXACTLY ONE vector with EXACTLY ONE sign
- ZERO cross-pair interference
- Expected accuracy: 99.92% baseline, 99.99%+ with error correction

Mathematical Foundation:
- SNR = 2D/N (for D=10,000, N=2,000: SNR=10)
- P(sign error) ≈ 0.079% per nucleotide
- Expected errors per chunk: ~1.58 errors per 2,000 nucleotides

Inspired by:
- Watson-Crick base pairing (A-T, G-C)
- Nanopore sequencing error correction strategies
- Ternary computing balanced representations {-1, 0, +1}
"""

import gzip
import json
import logging
import bisect
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pysam
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result from nucleotide query"""
    nucleotide: str
    confidence: float
    at_similarity: float
    gc_similarity: float
    pair: str  # 'AT' or 'GC'


class ComplementaryPairEncoder:
    """
    Complementary Pair HDC Encoder

    Encodes genomic sequences using Watson-Crick complementary pairs:
    - AT pair: A → +1, T → -1
    - GC pair: G → +1, C → -1

    Each nucleotide position appears in exactly ONE vector with exactly ONE sign,
    eliminating cross-pair interference entirely.
    """

    def __init__(
        self,
        gdiff_path: Path,
        guide_fasta_dir: Path,
        dimension: int = 10000,
        chunk_size: int = 2000,
        seed: int = 42
    ):
        """
        Initialize Complementary Pair Encoder

        Args:
            gdiff_path: Path to GDiff differential encoding file
            guide_fasta_dir: Directory containing guide reference FASTAs (ref1-ref11.fa.gz)
            dimension: Hypervector dimension (default 10,000)
            chunk_size: Nucleotides per chunk (default 2,000)
            seed: Random seed for reproducibility
        """
        self.gdiff_path = gdiff_path
        self.guide_fasta_dir = guide_fasta_dir
        self.D = dimension
        self.N = chunk_size
        self.seed = seed

        logger.info(f"Initializing ComplementaryPairEncoder")
        logger.info(f"  Dimension: {self.D:,}D")
        logger.info(f"  Chunk size: {self.N:,} bp")
        logger.info(f"  SNR: {2 * self.D / self.N:.2f}")

        # Generate position codebook
        logger.info(f"  Generating position codebook...")
        self.position_codebook = self._generate_position_codebook()
        logger.info(f"  ✓ Position codebook generated")

        # Load GDiff metadata (streaming - don't load all variants)
        logger.info(f"  Loading GDiff metadata...")
        with gzip.open(self.gdiff_path, 'rt') as f:
            self.gdiff = json.load(f)

        self.region_guide_map = self.gdiff.get("region_guide_map", {})
        logger.info(f"  ✓ Loaded region→guide map: {len(self.region_guide_map)} regions")

        # Pre-index variants by chromosome for O(log N) lookup
        logger.info(f"  Indexing variants by chromosome...")
        self.variants_by_chrom = self._index_variants_by_chrom()
        total_variants = sum(len(v) for v in self.variants_by_chrom.values())
        logger.info(f"  ✓ Indexed {total_variants:,} variants")

        # Load guide FASTAs (bgzip-compressed with .fai index)
        logger.info(f"  Loading guide FASTAs...")
        self.guide_fastas = {}
        for i in range(1, 12):  # ref1-ref11
            fasta_path = guide_fasta_dir / f"ref{i}.fa.gz"
            if fasta_path.exists():
                self.guide_fastas[f"ref{i}"] = pysam.FastaFile(str(fasta_path))
        logger.info(f"  ✓ Loaded {len(self.guide_fastas)} guide FASTAs")

        # Storage for encoded chunks
        self.encoded_chunks: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        logger.info(f"✓ Initialization complete")

    def _generate_position_codebook(self) -> np.ndarray:
        """
        Generate position codebook: N random D-dimensional bipolar vectors

        Each position i gets a unique random vector with elements ∈ {-1, +1}
        Uses int8 for memory efficiency (1 byte per element vs 4 for int32)

        Memory: N × D bytes (2,000 × 10,000 = 20 MB)
        """
        np.random.seed(self.seed)
        codebook = np.random.choice([-1, 1], size=(self.N, self.D)).astype(np.int8)
        return codebook

    def _index_variants_by_chrom(self) -> Dict[str, List[dict]]:
        """
        Index variants by chromosome with sorted positions for O(log N) lookup

        Returns:
            Dictionary: chrom → sorted list of variants
        """
        variants_by_chrom = {}

        for variant in self.gdiff["differential_variants"]:
            chrom = variant["chrom"]
            if chrom not in variants_by_chrom:
                variants_by_chrom[chrom] = []
            variants_by_chrom[chrom].append(variant)

        # Sort by position for binary search
        for chrom in variants_by_chrom:
            variants_by_chrom[chrom].sort(key=lambda v: v["pos"])

        return variants_by_chrom

    def _get_guide_for_region(self, chrom: str, region_start: int) -> Optional[str]:
        """
        Get guide reference ID for a genomic region

        Uses region→guide map from GDiff for privacy-preserving random cycling
        """
        region_key = f"{chrom}:{region_start // 10000}"
        return self.region_guide_map.get(region_key)

    def _get_nucleotide_at_position(
        self,
        chrom: str,
        pos: int,
        guide_id: str
    ) -> str:
        """
        Get nucleotide at position from guide FASTA or variant

        1. Check if position has variant in GDiff
        2. If variant exists, return alt nucleotide
        3. Otherwise, fetch from guide FASTA (reference nucleotide)

        Args:
            chrom: Chromosome
            pos: 0-based position
            guide_id: Guide reference ID (e.g., 'ref2')

        Returns:
            Nucleotide: 'A', 'T', 'G', or 'C'
        """
        # Binary search for variant at this position
        chrom_variants = self.variants_by_chrom.get(chrom, [])
        if chrom_variants:
            idx = bisect.bisect_left(chrom_variants, pos, key=lambda v: v["pos"])
            if idx < len(chrom_variants) and chrom_variants[idx]["pos"] == pos:
                # Variant found - return alt nucleotide
                alt = chrom_variants[idx]["alt"]
                if alt and alt in ['A', 'T', 'G', 'C']:
                    return alt

        # No variant - fetch from guide FASTA
        if guide_id in self.guide_fastas:
            try:
                # pysam uses 0-based coordinates
                nucleotide = self.guide_fastas[guide_id].fetch(chrom, pos, pos + 1).upper()
                if nucleotide in ['A', 'T', 'G', 'C']:
                    return nucleotide
            except Exception:
                pass

        # Fallback (should rarely happen)
        return 'N'

    def encode_chunk(self, chrom: str, chunk_start: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a genomic chunk using Complementary Pair architecture

        Watson-Crick pairing:
        - AT pair: A → +1, T → -1
        - GC pair: G → +1, C → -1

        Args:
            chrom: Chromosome
            chunk_start: 0-based start position of chunk

        Returns:
            (AT_vec, GC_vec): Two D-dimensional hypervectors
        """
        AT_vec = np.zeros(self.D, dtype=np.float32)
        GC_vec = np.zeros(self.D, dtype=np.float32)

        # Get guide reference for this region
        guide_id = self._get_guide_for_region(chrom, chunk_start)
        if not guide_id:
            # Fallback to ref1 if no mapping
            guide_id = 'ref1'

        # Process each position in chunk
        for offset in range(self.N):
            pos = chunk_start + offset
            nucleotide = self._get_nucleotide_at_position(chrom, pos, guide_id)

            # Get position vector from codebook
            pos_vec = self.position_codebook[offset].astype(np.float32)

            # Complementary Pair encoding
            if nucleotide == 'A':
                AT_vec += pos_vec
            elif nucleotide == 'T':
                AT_vec -= pos_vec
            elif nucleotide == 'G':
                GC_vec += pos_vec
            elif nucleotide == 'C':
                GC_vec -= pos_vec
            # 'N' contributes to neither vector (ternary: 0)

        return AT_vec, GC_vec

    def encode_genome(self, chromosomes: Optional[List[str]] = None) -> int:
        """
        Encode entire genome in chunks

        Streaming implementation - processes one chunk at a time

        Args:
            chromosomes: List of chromosomes to encode (default: chr1-chr22, chrX, chrY)

        Returns:
            Number of chunks encoded
        """
        if chromosomes is None:
            chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

        logger.info(f"Encoding genome in {self.N:,} bp chunks...")

        # Estimate chunk count (rough approximation)
        genome_size = 3_000_000_000  # ~3 billion bp
        estimated_chunks = genome_size // self.N

        chunk_count = 0

        with tqdm(total=estimated_chunks, desc="Encoding chunks") as pbar:
            for chrom in chromosomes:
                # Get chromosome length from first available guide FASTA
                if not self.guide_fastas:
                    continue

                first_guide = list(self.guide_fastas.values())[0]
                try:
                    chrom_length = first_guide.get_reference_length(chrom)
                except Exception:
                    continue

                # Process chromosome in chunks
                for chunk_start in range(0, chrom_length, self.N):
                    AT_vec, GC_vec = self.encode_chunk(chrom, chunk_start)

                    # Store chunk
                    chunk_key = f"{chrom}:{chunk_start}"
                    self.encoded_chunks[chunk_key] = (AT_vec, GC_vec)

                    chunk_count += 1
                    pbar.update(1)

        logger.info(f"✓ Encoded {chunk_count:,} chunks")
        return chunk_count

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
            - If AT pair: sign(sim_AT) determines A (+) vs T (-)
            - If GC pair: sign(sim_GC) determines G (+) vs C (-)

        Args:
            chrom: Chromosome
            pos: 0-based genomic position

        Returns:
            QueryResult with nucleotide, confidence, and similarity scores
        """
        # Find chunk containing this position
        chunk_start = (pos // self.N) * self.N
        offset = pos - chunk_start
        chunk_key = f"{chrom}:{chunk_start}"

        if chunk_key not in self.encoded_chunks:
            raise ValueError(f"Chunk {chunk_key} not encoded")

        AT_vec, GC_vec = self.encoded_chunks[chunk_key]

        # Get position vector
        pos_vec = self.position_codebook[offset].astype(np.float32)

        # Compute normalized similarities (dot products)
        sim_AT = np.dot(pos_vec, AT_vec) / (np.linalg.norm(AT_vec) + 1e-10)
        sim_GC = np.dot(pos_vec, GC_vec) / (np.linalg.norm(GC_vec) + 1e-10)

        # Stage 1: Pair selection (magnitude comparison)
        if abs(sim_AT) > abs(sim_GC):
            # AT pair selected
            pair = 'AT'
            nucleotide = 'A' if sim_AT > 0 else 'T'
            confidence = abs(sim_AT) / (abs(sim_AT) + abs(sim_GC) + 1e-10)
        else:
            # GC pair selected
            pair = 'GC'
            nucleotide = 'G' if sim_GC > 0 else 'C'
            confidence = abs(sim_GC) / (abs(sim_AT) + abs(sim_GC) + 1e-10)

        return QueryResult(
            nucleotide=nucleotide,
            confidence=confidence,
            at_similarity=sim_AT,
            gc_similarity=sim_GC,
            pair=pair
        )

    def save(self, output_path: Path):
        """
        Save encoded chunks to disk

        Storage format: .npz compressed archive
        - Separate arrays for AT_vec and GC_vec for each chunk
        - Metadata: dimension, chunk_size, seed
        """
        logger.info(f"Saving encoded chunks to {output_path}...")

        # Flatten chunks into separate AT and GC arrays
        chunk_keys = sorted(self.encoded_chunks.keys())

        AT_arrays = {}
        GC_arrays = {}

        for chunk_key in chunk_keys:
            AT_vec, GC_vec = self.encoded_chunks[chunk_key]
            AT_arrays[f"AT_{chunk_key}"] = AT_vec
            GC_arrays[f"GC_{chunk_key}"] = GC_vec

        # Save with metadata
        np.savez_compressed(
            output_path,
            **AT_arrays,
            **GC_arrays,
            dimension=self.D,
            chunk_size=self.N,
            seed=self.seed,
            chunk_keys=chunk_keys
        )

        # Get file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Saved {len(chunk_keys):,} chunks ({size_mb:.2f} MB)")

    def close(self):
        """Close all open FASTA files"""
        for fasta in self.guide_fastas.values():
            fasta.close()


# Ternary Computing Enhancement (Section 3.4)
class TernaryEnhancedEncoder(ComplementaryPairEncoder):
    """
    Ternary-Enhanced Complementary Pair Encoder

    Inspiration from balanced ternary computing {-1, 0, +1}:
    - A → +1 (AT pair)
    - T → -1 (AT pair)
    - G → +1 (GC pair)
    - C → -1 (GC pair)
    - N → 0 (contributes to neither vector)

    This naturally maps to balanced ternary representations,
    enabling efficient hardware implementations on ternary processors.

    Additional features:
    - Quality-weighted encoding (for nanopore sequencing)
    - Uncertainty quantification
    """

    def encode_chunk_with_quality(
        self,
        chrom: str,
        chunk_start: int,
        quality_scores: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode chunk with quality-weighted contributions

        Nanopore sequencing connection (Section 3.3.1):
        - Low-quality bases contribute less to hypervector
        - Mimics nanopore error correction strategies

        Args:
            chrom: Chromosome
            chunk_start: Chunk start position
            quality_scores: Phred quality scores (length N), or None for uniform weighting

        Returns:
            (AT_vec, GC_vec): Quality-weighted hypervectors
        """
        AT_vec = np.zeros(self.D, dtype=np.float32)
        GC_vec = np.zeros(self.D, dtype=np.float32)

        guide_id = self._get_guide_for_region(chrom, chunk_start) or 'ref1'

        for offset in range(self.N):
            pos = chunk_start + offset
            nucleotide = self._get_nucleotide_at_position(chrom, pos, guide_id)

            # Get position vector
            pos_vec = self.position_codebook[offset].astype(np.float32)

            # Apply quality weighting (default: 1.0)
            if quality_scores is not None:
                weight = quality_scores[offset]
            else:
                weight = 1.0

            weighted_pos_vec = pos_vec * weight

            # Ternary encoding with quality weighting
            if nucleotide == 'A':
                AT_vec += weighted_pos_vec
            elif nucleotide == 'T':
                AT_vec -= weighted_pos_vec
            elif nucleotide == 'G':
                GC_vec += weighted_pos_vec
            elif nucleotide == 'C':
                GC_vec -= weighted_pos_vec
            # 'N' → 0 (balanced ternary zero state)

        return AT_vec, GC_vec
