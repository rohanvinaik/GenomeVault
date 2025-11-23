#!/usr/bin/env python3
"""
Lens-Aware Decoder for 3-Bank Ternary Genomic HDC - CORRECTED

Aligns with encode_3bank_split_architecture.py which stores:
- Bank 1: Hydrophobic (T=+1, A=-1, G/C=0)
- Bank 2: Major Groove (G=+1, C=-1, A/T=0)
- Bank 3: Hinge (YR=+1, RY=-1, neutral=0)

Implements:
1. ZCR-based texture classification (Bank 3 Hinge rhythm)
2. Structural motif lens library (3 ternary lenses)
3. LINEAR magnitude-based compositional weighting
4. Genomic Monty Hall 4-way classification

Version: 2.0 - Architecturally Corrected
Date: November 2025
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextureClassification:
    """Result of texture classification using Hinge bank."""
    texture_type: str
    magnitude: float
    variance: float
    zcr: float
    confidence: float


class TextureClassifier:
    """
    Classifies genomic texture using Bank 3 (Hinge) ZCR.

    Texture Types:
    - HOMOPOLYMER: ZCR < 0.05 (RR or YY runs)
    - ALTERNATING: ZCR > 0.8 (TATA-like YRYRY...)
    - CPG_LIKE: High magnitude + variance (frequent dinucleotide steps)
    - ALU_LIKE: Moderate (GC-rich body + A-tail)
    - COMPLEX_CODING: High variance, moderate ZCR
    """

    def __init__(self):
        # Adaptive thresholds
        self.magnitude_high = None
        self.magnitude_moderate = None
        self.variance_high = None
        self.variance_moderate = None
        self.calibrated = False

    def calibrate(self, hinge_vectors: np.ndarray):
        """Calibrate thresholds from sample of Hinge vectors."""
        magnitudes = np.linalg.norm(hinge_vectors, axis=1)
        variances = np.var(hinge_vectors, axis=1)

        self.magnitude_high = np.percentile(magnitudes, 75)
        self.magnitude_moderate = np.percentile(magnitudes, 50)
        self.variance_high = np.percentile(variances, 75)
        self.variance_moderate = np.percentile(variances, 50)
        self.calibrated = True

        logger.info(f"Texture classifier calibrated: mag_high={self.magnitude_high:.2f}, var_high={self.variance_high:.2f}")

    def classify(self, hinge_vector: np.ndarray) -> TextureClassification:
        """
        Classify texture using Hinge bank ZCR + statistics.

        Args:
            hinge_vector: Bank 3 (Hinge) ternary vector {-1, 0, +1}

        Returns:
            TextureClassification
        """
        if not self.calibrated:
            # Default thresholds
            self.magnitude_high = 0.75 * len(hinge_vector)
            self.magnitude_moderate = 0.5 * len(hinge_vector)
            self.variance_high = 0.3
            self.variance_moderate = 0.2

        magnitude = np.linalg.norm(hinge_vector)
        variance = np.var(hinge_vector)

        # Zero-Crossing Rate (ZCR) - O(N) rhythm detector
        sign_changes = np.diff(np.sign(hinge_vector)) != 0
        zcr = np.sum(sign_changes) / len(hinge_vector)

        # Classification
        if magnitude > self.magnitude_high and zcr < 0.05:
            texture_type = 'HOMOPOLYMER'
            confidence = min(1.0, (magnitude / self.magnitude_high) * (0.05 / (zcr + 1e-6)))

        elif zcr > 0.8:
            texture_type = 'ALTERNATING'
            confidence = min(1.0, zcr * (variance / (self.variance_high + 1e-6)))

        elif magnitude > self.magnitude_high and variance > self.variance_moderate:
            texture_type = 'CPG_LIKE'
            confidence = min(1.0, (magnitude / self.magnitude_high) * (variance / (self.variance_high + 1e-6)))

        elif variance > self.variance_high and magnitude < self.magnitude_moderate:
            texture_type = 'COMPLEX_CODING'
            confidence = min(1.0, variance / self.variance_high)

        else:
            texture_type = 'ALU_LIKE'
            confidence = 0.5

        return TextureClassification(
            texture_type=texture_type,
            magnitude=magnitude,
            variance=variance,
            zcr=zcr,
            confidence=confidence
        )


@dataclass
class MotifLens:
    """Precomputed consensus for a structural motif (3 ternary banks)."""
    name: str
    texture_type: str
    bank1: np.ndarray  # Hydrophobic {-1, 0, +1}
    bank2: np.ndarray  # Major Groove {-1, 0, +1}
    bank3: np.ndarray  # Hinge {-1, 0, +1}
    prevalence: float
    typical_size: int


class LensLibrary:
    """Manages structural motif consensus lenses (3 ternary banks)."""

    def __init__(self, D: int = 5120):
        self.D = D
        self.lenses: Dict[str, MotifLens] = {}
        logger.info(f"Initializing LensLibrary with D={D}")

    def build_from_reference(self, reference_fasta: str, position_codebook: np.ndarray):
        """Build lens library from reference genome."""
        logger.info(f"Building lens library from {reference_fasta}")

        # Generate consensus sequences
        alu_seq = self._generate_alu_consensus()
        self.lenses['ALU_YI'] = self._encode_motif_to_lens(
            name='ALU_YI',
            sequence=alu_seq,
            texture_type='ALU_LIKE',
            prevalence=0.11,
            typical_size=300,
            position_codebook=position_codebook
        )

        cpg_seq = self._generate_cpg_consensus()
        self.lenses['CPG_ISLAND'] = self._encode_motif_to_lens(
            name='CPG_ISLAND',
            sequence=cpg_seq,
            texture_type='CPG_LIKE',
            prevalence=0.01,
            typical_size=1000,
            position_codebook=position_codebook
        )

        tata_seq = "TATAWAAW" * 4
        self.lenses['TATA_BOX'] = self._encode_motif_to_lens(
            name='TATA_BOX',
            sequence=tata_seq.replace('W', 'A'),
            texture_type='ALTERNATING',
            prevalence=0.001,
            typical_size=30,
            position_codebook=position_codebook
        )

        polya_seq = "A" * 50
        self.lenses['POLY_A'] = self._encode_motif_to_lens(
            name='POLY_A',
            sequence=polya_seq,
            texture_type='HOMOPOLYMER',
            prevalence=0.02,
            typical_size=50,
            position_codebook=position_codebook
        )

        logger.info(f"Built {len(self.lenses)} lenses: {list(self.lenses.keys())}")

    def _generate_alu_consensus(self) -> str:
        """Alu: GC-rich body + poly-A tail."""
        gc_body = "GCGCGCTAGCTAGCGCGCTAGCTAGCGCGC" * 8
        polya_tail = "A" * 20
        return gc_body + polya_tail

    def _generate_cpg_consensus(self) -> str:
        """CpG island: High CG content."""
        return "CGCGCGCGCGCGCGCGCG" * 20

    def _encode_motif_to_lens(
        self,
        name: str,
        sequence: str,
        texture_type: str,
        prevalence: float,
        typical_size: int,
        position_codebook: np.ndarray
    ) -> MotifLens:
        """
        Encode motif to 3-bank ternary lens (matching encoder logic).
        """
        N = len(position_codebook)
        sequence = sequence.upper()

        PURINES = {'A', 'G'}
        PYRIMIDINES = {'C', 'T'}

        # Ternary accumulators
        acc_hydro = np.zeros(self.D, dtype=np.int16)
        acc_groove = np.zeros(self.D, dtype=np.int16)
        acc_hinge = np.zeros(self.D, dtype=np.int16)

        for i, nuc in enumerate(sequence[:N]):
            pos_vec = position_codebook[i]

            # Bank 1: Hydrophobic
            if nuc == 'T':
                acc_hydro += pos_vec
            elif nuc == 'A':
                acc_hydro -= pos_vec

            # Bank 2: Major Groove
            if nuc == 'G':
                acc_groove += pos_vec
            elif nuc == 'C':
                acc_groove -= pos_vec

            # Bank 3: Hinge (contextual)
            if i < len(sequence) - 1:
                next_nuc = sequence[i + 1]
                is_YR = (nuc in PYRIMIDINES) and (next_nuc in PURINES)
                is_RY = (nuc in PURINES) and (next_nuc in PYRIMIDINES)

                if is_YR:
                    acc_hinge += pos_vec
                elif is_RY:
                    acc_hinge -= pos_vec

        # Sparsify to ternary
        def sparsify_ternary(acc, percentile=92):
            result = np.zeros_like(acc, dtype=np.int8)
            pos_vals = acc[acc > 0]
            if len(pos_vals) > 0:
                pos_thresh = np.percentile(pos_vals, percentile)
                result[acc > pos_thresh] = 1
            neg_vals = acc[acc < 0]
            if len(neg_vals) > 0:
                neg_thresh = np.percentile(neg_vals, 100 - percentile)
                result[acc < neg_thresh] = -1
            return result

        return MotifLens(
            name=name,
            texture_type=texture_type,
            bank1=sparsify_ternary(acc_hydro),
            bank2=sparsify_ternary(acc_groove),
            bank3=sparsify_ternary(acc_hinge),
            prevalence=prevalence,
            typical_size=typical_size
        )

    def get_lenses_for_texture(self, texture: str) -> List[MotifLens]:
        """Return lenses matching texture type."""
        return [lens for lens in self.lenses.values() if lens.texture_type == texture]

    def save(self, output_path: Path):
        """Save to HDF5."""
        with h5py.File(output_path, 'w') as f:
            f.attrs['D'] = self.D
            f.attrs['num_lenses'] = len(self.lenses)

            for lens_name, lens in self.lenses.items():
                grp = f.create_group(lens_name)
                grp.create_dataset('bank1', data=lens.bank1, compression='gzip')
                grp.create_dataset('bank2', data=lens.bank2, compression='gzip')
                grp.create_dataset('bank3', data=lens.bank3, compression='gzip')
                grp.attrs['texture_type'] = lens.texture_type
                grp.attrs['prevalence'] = lens.prevalence
                grp.attrs['typical_size'] = lens.typical_size

        logger.info(f"Saved lens library to {output_path}")

    @classmethod
    def load(cls, lens_library_path: Path) -> 'LensLibrary':
        """Load from HDF5."""
        with h5py.File(lens_library_path, 'r') as f:
            library = cls(D=f.attrs['D'])

            for lens_name in f.keys():
                grp = f[lens_name]
                library.lenses[lens_name] = MotifLens(
                    name=lens_name,
                    texture_type=grp.attrs['texture_type'],
                    bank1=grp['bank1'][:],
                    bank2=grp['bank2'][:],
                    bank3=grp['bank3'][:],
                    prevalence=grp.attrs['prevalence'],
                    typical_size=grp.attrs['typical_size']
                )

        logger.info(f"Loaded lens library from {lens_library_path} ({len(library.lenses)} lenses)")
        return library


class ChunkCache:
    """Cache magnitude computations."""

    def __init__(self, max_size: int = 10000):
        self.cache: Dict[int, Tuple[float, float]] = {}
        self.max_size = max_size

    def get_magnitudes(self, chunk_idx: int, bank1: np.ndarray, bank2: np.ndarray) -> Tuple[float, float]:
        """Get or compute AT vs GC magnitudes."""
        if chunk_idx not in self.cache:
            # AT magnitude = |bank1| (hydrophobic signal strength)
            # GC magnitude = |bank2| (major groove signal strength)
            mag_AT = np.linalg.norm(bank1)
            mag_GC = np.linalg.norm(bank2)

            if len(self.cache) >= self.max_size:
                oldest = next(iter(self.cache))
                del self.cache[oldest]

            self.cache[chunk_idx] = (mag_AT, mag_GC)

        return self.cache[chunk_idx]


class LensAwareDecoder:
    """
    Lens-aware decoder using 3 ternary banks + Genomic Monty Hall logic.
    """

    def __init__(
        self,
        encoded_h5_path: str,
        lens_library: Optional[LensLibrary] = None,
        use_magnitude_weighting: bool = True,
        lens_alpha: float = 0.3
    ):
        self.encoded_h5_path = Path(encoded_h5_path)
        self.lens_library = lens_library
        self.use_magnitude_weighting = use_magnitude_weighting
        self.lens_alpha = lens_alpha

        self.texture_classifier = TextureClassifier()
        self.magnitude_cache = ChunkCache()

        # Load genome
        self.h5_file = h5py.File(self.encoded_h5_path, 'r')
        self.D = self.h5_file.attrs.get('D', 5120)
        self.N = self.h5_file.attrs.get('N', 1024)

        # Build chunk index
        self.chunk_index = self._build_chunk_index()

        logger.info(f"Initialized LensAwareDecoder: D={self.D}, N={self.N}")
        logger.info(f"  Lenses: {len(self.lens_library.lenses) if lens_library else 0}")
        logger.info(f"  Magnitude weighting: LINEAR (not squared)")

    def _build_chunk_index(self) -> Dict[str, List[Tuple[int, int, int]]]:
        """Build index: chrom -> [(start, end, chunk_idx), ...]."""
        index = {}

        if 'chunk_keys' not in self.h5_file:
            logger.warning("No chunk_keys in H5, using linear indexing")
            return index

        for chunk_idx, key_bytes in enumerate(self.h5_file['chunk_keys'][:]):
            key = key_bytes.decode('utf-8')
            chrom, range_str = key.split(':')
            start, end = map(int, range_str.split('-'))

            if chrom not in index:
                index[chrom] = []
            index[chrom].append((start, end, chunk_idx))

        logger.info(f"Built chunk index for {len(index)} chromosomes")
        return index

    def decode_position(
        self,
        chrom: str,
        pos: int,
        position_codebook: np.ndarray
    ) -> Tuple[str, float, Optional[str], Optional[str]]:
        """
        Decode nucleotide using Genomic Monty Hall + lens overlay.

        Returns:
            (nucleotide, confidence, texture_type, lens_name)
        """
        # Step 1: Find chunk
        chunk_idx, offset = self._find_chunk_and_offset(chrom, pos)

        # Step 2: Load 3 ternary banks
        bank1 = self.h5_file['bank1'][chunk_idx, :]  # Hydrophobic
        bank2 = self.h5_file['bank2'][chunk_idx, :]  # Major Groove
        bank3 = self.h5_file['bank3'][chunk_idx, :]  # Hinge

        # Step 3: Classify texture (using Bank 3 Hinge)
        texture_type = None
        lens_name = None
        if self.lens_library:
            texture_class = self.texture_classifier.classify(bank3)
            texture_type = texture_class.texture_type

            # Step 4: Select and apply lens
            best_lens = self._select_best_lens(texture_type, bank1, bank2, bank3)
            if best_lens is not None:
                lens_name = best_lens.name
                bank1 = bank1 + self.lens_alpha * best_lens.bank1
                bank2 = bank2 + self.lens_alpha * best_lens.bank2
                bank3 = bank3 + self.lens_alpha * best_lens.bank3

        # Step 5: Compute similarities
        query_vec = position_codebook[offset]

        sim_hydro = np.dot(bank1, query_vec) / self.D
        sim_groove = np.dot(bank2, query_vec) / self.D
        sim_hinge = np.dot(bank3, query_vec) / self.D

        # Step 6: Genomic Monty Hall 4-way classification
        nucleotide, confidence = self._monty_hall_decode(
            sim_hydro, sim_groove, sim_hinge,
            chunk_idx, bank1, bank2
        )

        return nucleotide, confidence, texture_type, lens_name

    def _find_chunk_and_offset(self, chrom: str, pos: int) -> Tuple[int, int]:
        """Find chunk containing position."""
        if chrom not in self.chunk_index:
            raise ValueError(f"Chromosome {chrom} not found")

        for start, end, chunk_idx in self.chunk_index[chrom]:
            if start <= pos < end:
                return chunk_idx, pos - start

        raise ValueError(f"Position {chrom}:{pos} not in any chunk")

    def _select_best_lens(
        self,
        texture: str,
        bank1: np.ndarray,
        bank2: np.ndarray,
        bank3: np.ndarray
    ) -> Optional[MotifLens]:
        """Select best matching lens using batch similarity."""
        candidates = self.lens_library.get_lenses_for_texture(texture)
        if not candidates:
            return None

        # Batch similarity across all 3 banks
        best_lens = None
        best_score = -1.0

        for lens in candidates:
            sim1 = np.dot(bank1, lens.bank1) / self.D
            sim2 = np.dot(bank2, lens.bank2) / self.D
            sim3 = np.dot(bank3, lens.bank3) / self.D

            combined = (sim1 + sim2 + sim3) / 3.0

            if combined > best_score:
                best_score = combined
                best_lens = lens

        return best_lens

    def _monty_hall_decode(
        self,
        sim_hydro: float,
        sim_groove: float,
        sim_hinge: float,
        chunk_idx: int,
        bank1: np.ndarray,
        bank2: np.ndarray
    ) -> Tuple[str, float]:
        """
        Genomic Monty Hall: Use 3 orthogonal lenses to resolve 4-way.

        Hydrophobic: A(-1) vs T(+1) vs GC(0)
        Major Groove: G(+1) vs C(-1) vs AT(0)
        Hinge: Purine(YR steps) vs Pyrimidine(RY steps)
        """
        # Magnitude weighting (LINEAR)
        if self.use_magnitude_weighting:
            mag_AT, mag_GC = self.magnitude_cache.get_magnitudes(chunk_idx, bank1, bank2)
            total_mag = mag_AT + mag_GC

            if total_mag > 0:
                AT_weight = mag_AT / total_mag  # LINEAR
                GC_weight = mag_GC / total_mag
            else:
                AT_weight = GC_weight = 0.5
        else:
            AT_weight = GC_weight = 0.5

        # Monty Hall logic: Cross-validate 3 signals
        scores = {
            'A': AT_weight * (-sim_hydro) + (sim_hinge if sim_hinge > 0 else 0),  # A: hydro negative, purine
            'T': AT_weight * (sim_hydro) + (sim_hinge if sim_hinge < 0 else 0),   # T: hydro positive, pyrimidine
            'G': GC_weight * (sim_groove) + (sim_hinge if sim_hinge > 0 else 0),  # G: groove positive, purine
            'C': GC_weight * (-sim_groove) + (sim_hinge if sim_hinge < 0 else 0), # C: groove negative, pyrimidine
        }

        best_nuc = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_nuc] / total_score if total_score > 0 else 0.25

        return best_nuc, confidence

    def close(self):
        """Close H5 file."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================================
# CLI
# ============================================================================

def build_lens_library_cli():
    """CLI to build lens library."""
    import argparse

    parser = argparse.ArgumentParser(description="Build structural motif lens library")
    parser.add_argument('--reference', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--D', type=int, default=5120)
    parser.add_argument('--N', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Generate position codebook
    np.random.seed(args.seed)
    position_codebook = np.random.choice([-1, 1], size=(args.N, args.D)).astype(np.int8)

    # Build library
    library = LensLibrary(D=args.D)
    library.build_from_reference(args.reference, position_codebook)

    # Save
    library.save(Path(args.output))
    print(f"✓ Saved lens library to {args.output}")


if __name__ == '__main__':
    build_lens_library_cli()
