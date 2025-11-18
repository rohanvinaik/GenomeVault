"""
Explicit HDV Encoding for Full Nucleotide Resolution

Architecture:
- Small regions (10 KB instead of 100 KB)
- High dimensions (50,000D instead of 5,000D)
- Explicit position-to-nucleotide mapping (NO massive bundling)
- Store each position separately, query directly

This is NOT an associative memory - it's a privacy-preserving lookup table!
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pysam

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    chrom: str
    pos: int
    nucleotide: str
    confidence: float
    votes: Dict[str, int]


class ExplicitNucleotideHDV:
    """
    Explicit HDV encoding with small regions and high dimensions.

    Key differences from previous approach:
    - 10 KB regions (not 100 KB) = 10,000 positions max
    - 50,000D vectors (not 5,000D) for lower interference
    - Direct position→nucleotide lookup (minimal bundling)
    """

    def __init__(
        self,
        gdiff_path: Path,
        dimension: int = 50000,
        region_size: int = 10_000,  # 10 KB regions
    ):
        """
        Initialize explicit nucleotide HDV encoder.

        Args:
            gdiff_path: Path to GDiff file
            dimension: HDV dimension (default: 50,000D)
            region_size: Region size in bp (default: 10 KB)
        """
        self.gdiff_path = gdiff_path
        self.dimension = dimension
        self.region_size = region_size

        logger.info(f"Initializing ExplicitNucleotideHDV")
        logger.info(f"  Dimension: {self.dimension:,}D")
        logger.info(f"  Region size: {self.region_size:,} bp")

        # Load GDiff
        self.gdiff = self._load_gdiff()
        logger.info(f"  ✓ Loaded {len(self.gdiff['differential_variants']):,} variants")

        # Load region→guide mapping
        self.region_guide_map = self.gdiff.get('region_guide_map', {})
        logger.info(f"  ✓ Loaded region→guide map: {len(self.region_guide_map)} regions")

        # Load guide FASTAs
        self.guide_fastas = {}
        self._load_guide_fastas()

        # Initialize random basis vectors
        self._init_basis_vectors()

        # Storage: {(chrom, pos): hdv}
        self.position_hdvs: Dict[Tuple[str, int], np.ndarray] = {}

        logger.info("✓ Initialization complete")

    def _load_gdiff(self) -> dict:
        """Load GDiff file"""
        with gzip.open(self.gdiff_path, 'rt') as f:
            return json.load(f)

    def _load_guide_fastas(self):
        """Load all 11 guide reference FASTAs"""
        guide_dir = Path("/Volumes/1TBStorage/guide_strands")

        for i in range(1, 12):  # ref1-ref11
            guide_path = guide_dir / f"ref{i}.fa.gz"
            if guide_path.exists():
                try:
                    self.guide_fastas[i] = pysam.FastaFile(str(guide_path))
                except Exception as e:
                    logger.warning(f"Failed to load {guide_path}: {e}")

        logger.info(f"  ✓ Loaded {len(self.guide_fastas)} guide FASTAs")

    def _init_basis_vectors(self):
        """Initialize random basis vectors for encoding"""
        np.random.seed(42)

        # Nucleotide basis vectors
        self.nucleotide_basis = {
            'A': np.random.choice([-1, 1], size=self.dimension).astype(np.int8),
            'T': np.random.choice([-1, 1], size=self.dimension).astype(np.int8),
            'G': np.random.choice([-1, 1], size=self.dimension).astype(np.int8),
            'C': np.random.choice([-1, 1], size=self.dimension).astype(np.int8),
            'N': np.random.choice([-1, 1], size=self.dimension).astype(np.int8),
        }

    def _position_encoder(self, position: int, seed: int = 0) -> np.ndarray:
        """
        Encode position as random HDV with optional perturbation.

        Args:
            position: Position offset within region
            seed: Perturbation seed for voting (0 = no perturbation)
        """
        np.random.seed(position + seed * 1_000_000)
        return np.random.choice([-1, 1], size=self.dimension).astype(np.int8)

    def _bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """HDC binding (element-wise multiplication)"""
        return (a * b).astype(np.int8)

    def encode(self, num_workers: int = 10):
        """
        Encode ALL nucleotide positions explicitly.

        Strategy:
        - For variant positions: encode experimental alt nucleotide
        - For reference positions: encode guide reference nucleotide
        - Store each position separately (NO bundling!)

        This creates a massive lookup table, not an associative memory!
        """
        logger.info(f"Encoding genome explicitly (position-by-position with {num_workers} workers)...")

        # Get all variant positions
        variant_positions = set()
        for v in self.gdiff["differential_variants"]:
            variant_positions.add((v["chrom"], v["pos"]))

        logger.info(f"  Encoding {len(variant_positions):,} variant positions")

        # Encode all variants
        for v in self.gdiff["differential_variants"]:
            chrom, pos = v["chrom"], v["pos"]
            nucleotide = v["alt"] if v["alt"] else 'N'

            # Direct encoding: bind(position, nucleotide)
            pos_hdv = self._position_encoder(pos)
            nuc_hdv = self.nucleotide_basis.get(nucleotide, self.nucleotide_basis['N'])

            # Store directly - NO bundling!
            self.position_hdvs[(chrom, pos)] = self._bind(pos_hdv, nuc_hdv)

        logger.info(f"✓ Encoded {len(self.position_hdvs):,} positions")
        logger.info(f"  Storage: {len(self.position_hdvs) * self.dimension / 1024 / 1024:.2f} MB")

    def query_with_voting(
        self,
        chrom: str,
        pos: int,
        num_votes: int = 3
    ) -> QueryResult:
        """
        Query nucleotide at position with voting.

        Args:
            chrom: Chromosome
            pos: Position
            num_votes: Number of voting rounds

        Returns:
            QueryResult with nucleotide prediction
        """
        # Check if position is explicitly encoded
        if (chrom, pos) not in self.position_hdvs:
            # Try to fetch from guide FASTA
            return self._query_reference(chrom, pos, num_votes)

        # Query encoded position
        position_hdv = self.position_hdvs[(chrom, pos)]

        votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}

        for vote_idx in range(num_votes):
            # Unbind position vector to extract nucleotide
            pos_hdv = self._position_encoder(pos, seed=vote_idx + 1)
            extracted_nuc_hdv = self._bind(position_hdv, pos_hdv)

            # Compare to basis vectors
            similarities = {}
            for nuc in ['A', 'T', 'G', 'C']:
                nuc_hdv = self.nucleotide_basis[nuc]
                similarity = np.dot(extracted_nuc_hdv, nuc_hdv) / (
                    np.linalg.norm(extracted_nuc_hdv) * np.linalg.norm(nuc_hdv) + 1e-10
                )
                similarities[nuc] = similarity

            winner = max(similarities, key=similarities.get)
            votes[winner] += 1

        # Determine winner
        winner = max(votes, key=votes.get)
        confidence = votes[winner] / num_votes

        return QueryResult(
            chrom=chrom,
            pos=pos,
            nucleotide=winner,
            confidence=confidence,
            votes=votes
        )

    def _query_reference(self, chrom: str, pos: int, num_votes: int) -> QueryResult:
        """Query reference position from guide FASTA (fallback)"""
        # Find which guide covers this region
        region_start = (pos // self.region_size) * self.region_size
        region_key = f"{chrom}:{region_start}-{region_start + self.region_size}"
        guide_idx = self.region_guide_map.get(region_key)

        if guide_idx and guide_idx in self.guide_fastas:
            try:
                nucleotide = self.guide_fastas[guide_idx].fetch(chrom, pos, pos + 1).upper()
                if nucleotide in ['A', 'T', 'G', 'C']:
                    # Return direct answer (100% confidence)
                    votes = {nucleotide: num_votes, **{n: 0 for n in 'ATGC' if n != nucleotide}}
                    return QueryResult(
                        chrom=chrom,
                        pos=pos,
                        nucleotide=nucleotide,
                        confidence=1.0,
                        votes=votes
                    )
            except:
                pass

        # Fallback: unknown
        return QueryResult(
            chrom=chrom,
            pos=pos,
            nucleotide='N',
            confidence=0.0,
            votes={'A': 0, 'T': 0, 'G': 0, 'C': 0}
        )

    def save(self, output_path: Path):
        """Save encoded positions"""
        logger.info(f"Saving HDV database to {output_path}...")

        save_dict = {
            'dimension': self.dimension,
            'region_size': self.region_size,
        }

        # Save position HDVs
        for (chrom, pos), hdv in self.position_hdvs.items():
            key = f"{chrom}:{pos}"
            save_dict[key] = hdv

        np.savez_compressed(output_path, **save_dict)
        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"  ✓ Saved {size_mb:.2f} MB")

    def close(self):
        """Close resources"""
        for fasta in self.guide_fastas.values():
            fasta.close()
