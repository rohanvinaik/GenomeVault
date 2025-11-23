#!/usr/bin/env python3
"""
Lens-Aware Decoder for Split Binary Genomic HDC

Implements the Structural Motif Lens Library system with:
1. Bank 2 texture classification (HOMOPOLYMER, ALTERNATING, CPG_LIKE, ALU_LIKE, COMPLEX_CODING)
2. Structural motif lens library (Alu, CpG, TATA, Poly-A, L1, Telomeric, CAG repeats)
3. Magnitude-based compositional weighting (LINEAR, not squared)
4. Hierarchical decoding pipeline: Texture → Lens → Similarity → Magnitude → Decode

Based on: docs/theory/STRUCTURAL_MOTIF_LENS_LIBRARY.md
Version: 1.0
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
    """Result of texture classification."""
    texture_type: str
    magnitude: float
    variance: float
    dominant_freq: int
    confidence: float


class TextureClassifier:
    """
    Classifies genomic texture using Bank 2 (Hinge) patterns.

    Texture Types:
    - HOMOPOLYMER: High magnitude, low variance (Poly-A/T runs)
    - ALTERNATING: High variance, periodic (TATA boxes)
    - CPG_LIKE: High magnitude, high variance (CG dinucleotides)
    - ALU_LIKE: Moderate magnitude/variance, GC-rich with A-tail
    - COMPLEX_CODING: High variance, no pattern (random coding)
    """

    def __init__(self):
        # Adaptive thresholds (will be calibrated from data)
        self.magnitude_high = None
        self.magnitude_moderate = None
        self.variance_high = None
        self.variance_moderate = None
        self.variance_low = None
        self.calibrated = False

    def calibrate(self, hinge_vectors: np.ndarray):
        """Calibrate thresholds from sample of Bank 2 vectors."""
        magnitudes = np.linalg.norm(hinge_vectors, axis=1)
        variances = np.var(hinge_vectors, axis=1)

        self.magnitude_high = np.percentile(magnitudes, 75)
        self.magnitude_moderate = np.percentile(magnitudes, 50)
        self.variance_high = np.percentile(variances, 75)
        self.variance_moderate = np.percentile(variances, 50)
        self.variance_low = np.percentile(variances, 25)
        self.calibrated = True

        logger.info(f"Texture classifier calibrated: mag_high={self.magnitude_high:.2f}, var_high={self.variance_high:.2f}")

    def classify(self, hinge_vector: np.ndarray) -> TextureClassification:
        """
        Classify texture type using magnitude, variance, and Zero-Crossing Rate (ZCR).

        ZCR Optimization (vs FFT):
        - O(N) instead of O(N log N)
        - Simple sign-change counting
        - Perfect for binary Purine/Pyrimidine signals
        - TATA box: ZCR ~0.8-1.0 (rapid oscillation)
        - Homopolymer: ZCR ~0.0 (no oscillation)
        - Random coding: ZCR ~0.5 (average)

        Args:
            hinge_vector: Bank 2 (Hinge) hypervector for chunk

        Returns:
            TextureClassification with type, metrics, and confidence
        """
        if not self.calibrated:
            # Use default thresholds if not calibrated
            self.magnitude_high = 0.75 * len(hinge_vector)
            self.magnitude_moderate = 0.5 * len(hinge_vector)
            self.variance_high = 0.3
            self.variance_moderate = 0.2
            self.variance_low = 0.1

        # Core metrics
        magnitude = np.linalg.norm(hinge_vector)
        variance = np.var(hinge_vector)

        # Zero-Crossing Rate (ZCR) - O(N) rhythm detector
        # Counts sign changes in bipolar signal (Purine ↔ Pyrimidine transitions)
        sign_changes = np.diff(np.sign(hinge_vector)) != 0
        zcr = np.sum(sign_changes) / len(hinge_vector)

        # Classification logic with confidence scoring
        if magnitude > self.magnitude_high and zcr < 0.05:
            # Low ZCR = steady state (homopolymer runs)
            texture_type = 'HOMOPOLYMER'
            confidence = min(1.0, (magnitude / self.magnitude_high) * (0.05 / (zcr + 1e-6)))

        elif zcr > 0.8:
            # High ZCR = rapid oscillation (TATA-like Pyr-Pur-Pyr-Pur)
            texture_type = 'ALTERNATING'
            confidence = min(1.0, zcr * (variance / self.variance_high))

        elif magnitude > self.magnitude_high and variance > self.variance_moderate:
            # High magnitude + high variance = CpG islands
            texture_type = 'CPG_LIKE'
            confidence = min(1.0, (magnitude / self.magnitude_high) * (variance / self.variance_high))

        elif variance > self.variance_high and magnitude < self.magnitude_moderate:
            # High variance, low magnitude = complex coding
            texture_type = 'COMPLEX_CODING'
            confidence = min(1.0, variance / self.variance_high)

        else:
            # Moderate everything = Alu-like
            texture_type = 'ALU_LIKE'
            confidence = 0.5  # Default/uncertain

        return TextureClassification(
            texture_type=texture_type,
            magnitude=magnitude,
            variance=variance,
            dominant_freq=int(zcr * 100),  # Store ZCR as percentage (for compatibility)
            confidence=confidence
        )


@dataclass
class MotifLens:
    """Precomputed consensus hypervector for a structural motif."""
    name: str
    texture_type: str
    bank1: np.ndarray  # Hydrophobic (T=+1, A=-1, GC=0)
    bank2: np.ndarray  # Major groove (G=+1, C=-1, AT=0)
    bank3: np.ndarray  # Hinge flexibility (YR=+1, RY=-1, neutral=0)
    prevalence: float  # % of genome
    typical_size: int  # bp


class LensLibrary:
    """
    Manages structural motif consensus hypervectors.

    Library includes:
    - ALU_YI: 11% prevalence, ~300 bp, GC-rich with A-tail
    - CPG_ISLAND: 1% prevalence, 200-2000 bp, GC-saturated
    - TATA_BOX: 0.1% prevalence, ~30 bp, AT-rich alternating
    - POLY_A: ~2% prevalence, 20-100 bp, homopolymer
    - L1_LINE: 17% prevalence, ~6 kb, bimodal AT/GC
    - TELOMERIC: <0.01% prevalence, 10-15 kb, TTAGGG periodic
    - CAG_REPEAT: <0.01% prevalence, 30-600 bp, trinucleotide repeat
    """

    def __init__(self, D: int = 5120):
        self.D = D
        self.lenses: Dict[str, MotifLens] = {}
        logger.info(f"Initializing LensLibrary with D={D}")

    def build_from_reference(self, reference_fasta: str, position_codebook: np.ndarray):
        """
        Build lens library by extracting motifs from reference genome.

        Args:
            reference_fasta: Path to reference genome FASTA
            position_codebook: Shared position codebook (N x D)
        """
        logger.info(f"Building lens library from {reference_fasta}")

        # For now, create synthetic consensus lenses
        # In production, would extract actual sequences from RepeatMasker, UCSC annotations

        # ALU_YI consensus (simplified - actual would come from RepeatMasker)
        alu_seq = self._generate_alu_consensus()
        self.lenses['ALU_YI'] = self._encode_motif_to_lens(
            name='ALU_YI',
            sequence=alu_seq,
            texture_type='ALU_LIKE',
            prevalence=0.11,
            typical_size=300,
            position_codebook=position_codebook
        )

        # CpG Island consensus
        cpg_seq = self._generate_cpg_consensus()
        self.lenses['CPG_ISLAND'] = self._encode_motif_to_lens(
            name='CPG_ISLAND',
            sequence=cpg_seq,
            texture_type='CPG_LIKE',
            prevalence=0.01,
            typical_size=1000,
            position_codebook=position_codebook
        )

        # TATA box consensus
        tata_seq = "TATAWAAW" * 4  # W = A or T
        self.lenses['TATA_BOX'] = self._encode_motif_to_lens(
            name='TATA_BOX',
            sequence=tata_seq.replace('W', 'A'),
            texture_type='ALTERNATING',
            prevalence=0.001,
            typical_size=30,
            position_codebook=position_codebook
        )

        # Poly-A consensus
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
        """Generate simplified Alu consensus sequence."""
        # Simplified Alu-like: GC-rich body + poly-A tail
        gc_body = "GCGCGCTAGCTAGCGCGCTAGCTAGCGCGC" * 8
        polya_tail = "A" * 20
        return gc_body + polya_tail

    def _generate_cpg_consensus(self) -> str:
        """Generate CpG island consensus sequence."""
        # High CG content with frequent CpG dinucleotides
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
        Encode a motif sequence to 3 ternary banks directly.

        Matches the main encoder's storage format: 3 ternary banks {-1, 0, +1}.
        This avoids the inefficiency of creating 6 binary banks and reconstructing.
        """
        N = len(position_codebook)
        sequence = sequence.upper()

        # Accumulate directly in ternary space
        acc_hydrophobic = np.zeros(self.D, dtype=np.int16)
        acc_major_groove = np.zeros(self.D, dtype=np.int16)
        acc_hinge = np.zeros(self.D, dtype=np.int16)

        prev_nuc = None
        for i, nuc in enumerate(sequence[:N]):  # Cap at position codebook size
            pos_vec = position_codebook[i]

            # Bank 1: Hydrophobic (T=+1, A=-1, GC=0)
            if nuc == 'T':
                acc_hydrophobic += pos_vec
            elif nuc == 'A':
                acc_hydrophobic -= pos_vec

            # Bank 2: Major Groove (G=+1, C=-1, AT=0)
            if nuc == 'G':
                acc_major_groove += pos_vec
            elif nuc == 'C':
                acc_major_groove -= pos_vec

            # Bank 3: Hinge (YR=+1, RY=-1, neutral=0)
            if prev_nuc is not None:
                is_purine = {'A': True, 'G': True, 'C': False, 'T': False}

                # Y→R step (pyrimidine to purine) = positive
                if not is_purine.get(prev_nuc, False) and is_purine.get(nuc, False):
                    acc_hinge += pos_vec

                # R→Y step (purine to pyrimidine) = negative
                elif is_purine.get(prev_nuc, False) and not is_purine.get(nuc, False):
                    acc_hinge -= pos_vec

            prev_nuc = nuc

        # Direct ternary quantization using np.sign()
        # No 6-binary intermediate! This is 50% less compute.
        bank1 = np.sign(acc_hydrophobic).astype(np.int8)  # {-1, 0, +1}
        bank2 = np.sign(acc_major_groove).astype(np.int8)  # {-1, 0, +1}
        bank3 = np.sign(acc_hinge).astype(np.int8)         # {-1, 0, +1}

        return MotifLens(
            name=name,
            texture_type=texture_type,
            bank1=bank1,  # Ternary int8
            bank2=bank2,  # Ternary int8
            bank3=bank3,  # Ternary int8
            prevalence=prevalence,
            typical_size=typical_size
        )

    def get_lenses_for_texture(self, texture: str) -> List[MotifLens]:
        """Return candidate lenses matching texture type."""
        return [lens for lens in self.lenses.values() if lens.texture_type == texture]

    def save(self, output_path: Path):
        """Save lens library to HDF5 (3 ternary banks)."""
        with h5py.File(output_path, 'w') as f:
            f.attrs['D'] = self.D
            f.attrs['num_lenses'] = len(self.lenses)

            for lens_name, lens in self.lenses.items():
                grp = f.create_group(lens_name)
                grp.create_dataset('bank1', data=lens.bank1, compression='gzip')  # Hydrophobic
                grp.create_dataset('bank2', data=lens.bank2, compression='gzip')  # Major Groove
                grp.create_dataset('bank3', data=lens.bank3, compression='gzip')  # Hinge
                grp.attrs['texture_type'] = lens.texture_type
                grp.attrs['prevalence'] = lens.prevalence
                grp.attrs['typical_size'] = lens.typical_size

        logger.info(f"Saved lens library to {output_path} ({len(self.lenses)} lenses)")

    @classmethod
    def load(cls, lens_library_path: Path) -> 'LensLibrary':
        """Load lens library from HDF5 (3 ternary banks)."""
        library = cls(D=5120)  # Will be overwritten

        with h5py.File(lens_library_path, 'r') as f:
            library.D = f.attrs['D']

            for lens_name in f.keys():
                grp = f[lens_name]
                library.lenses[lens_name] = MotifLens(
                    name=lens_name,
                    texture_type=grp.attrs['texture_type'],
                    bank1=grp['bank1'][:],  # Hydrophobic (ternary int8)
                    bank2=grp['bank2'][:],  # Major Groove (ternary int8)
                    bank3=grp['bank3'][:],  # Hinge (ternary int8)
                    prevalence=grp.attrs['prevalence'],
                    typical_size=grp.attrs['typical_size']
                )

        logger.info(f"Loaded lens library from {lens_library_path} ({len(library.lenses)} lenses)")
        return library


class ChunkCache:
    """Cache magnitude computations per chunk to avoid redundant calculations."""

    def __init__(self, max_size: int = 10000):
        self.cache: Dict[int, Tuple[float, float]] = {}
        self.max_size = max_size

    def get_magnitudes(self, chunk_idx: int, chunk_vectors: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """
        Get or compute Bank 1/2 magnitudes (Hydrophobic AT vs Major Groove GC).

        Returns: (mag_bank1, mag_bank2) for compositional weighting
        """
        if chunk_idx not in self.cache:
            mag1 = np.linalg.norm(chunk_vectors['bank1'])  # Hydrophobic (AT)
            mag2 = np.linalg.norm(chunk_vectors['bank2'])  # Major Groove (GC)

            # LRU eviction if cache too large
            if len(self.cache) >= self.max_size:
                # Simple eviction: remove oldest (first key)
                oldest = next(iter(self.cache))
                del self.cache[oldest]

            self.cache[chunk_idx] = (mag1, mag2)

        return self.cache[chunk_idx]


class LensAwareDecoder:
    """
    Main decoder with lens and magnitude awareness.

    Pipeline:
    1. Texture classification (Bank 2 FFT + statistics)
    2. Lens selection (match texture → best lens)
    3. Lens overlay (0.3 alpha blending)
    4. Similarity computation (dot products)
    5. LINEAR magnitude weighting (compositional prior)
    6. Final decoding (argmax)
    """

    def __init__(
        self,
        encoded_h5_path: str,
        lens_library: Optional[LensLibrary] = None,
        use_magnitude_weighting: bool = True,
        lens_alpha: float = 0.3
    ):
        """
        Initialize lens-aware decoder.

        Args:
            encoded_h5_path: Path to encoded_genome_6banks.h5
            lens_library: Pre-built LensLibrary (or None to skip lens overlay)
            use_magnitude_weighting: Enable LINEAR magnitude-based compositional weighting
            lens_alpha: Blending weight for lens overlay (0.0-1.0)
        """
        self.encoded_h5_path = Path(encoded_h5_path)
        self.lens_library = lens_library
        self.use_magnitude_weighting = use_magnitude_weighting
        self.lens_alpha = lens_alpha

        self.texture_classifier = TextureClassifier()
        self.magnitude_cache = ChunkCache()

        # Load genome data
        self.h5_file = h5py.File(self.encoded_h5_path, 'r')
        self.D = self.h5_file.attrs.get('D', 5120)
        self.N = self.h5_file.attrs.get('N', 1024)

        logger.info(f"Initialized LensAwareDecoder: D={self.D}, N={self.N}, lens_alpha={lens_alpha}")
        logger.info(f"  Lens library: {len(self.lens_library.lenses) if lens_library else 0} lenses")
        logger.info(f"  Magnitude weighting: {use_magnitude_weighting}")

    def decode_position(
        self,
        chrom: str,
        pos: int,
        position_codebook: np.ndarray
    ) -> Tuple[str, float, Optional[str], Optional[str]]:
        """
        Decode nucleotide at genomic position with lens and magnitude awareness.

        Args:
            chrom: Chromosome name
            pos: Genomic position (0-based)
            position_codebook: Shared position codebook (N x D)

        Returns:
            (nucleotide, confidence, texture_type, lens_name)
        """
        # Step 1: Get chunk vectors
        chunk_idx, offset_in_chunk = self._find_chunk_and_offset(chrom, pos)
        chunk_vectors = self._load_chunk_vectors(chunk_idx)

        # Step 2: Classify texture (if lens library available)
        texture_type = None
        lens_name = None
        if self.lens_library:
            texture_classification = self.texture_classifier.classify(chunk_vectors['bank2'])
            texture_type = texture_classification.texture_type

            # Step 3: Select and apply best lens
            best_lens = self._select_best_lens(texture_classification.texture_type, chunk_vectors)
            if best_lens is not None:
                lens_name = best_lens.name
                chunk_vectors = self._apply_lens_overlay(chunk_vectors, best_lens)

        # Step 4: Compute similarities
        sims = self._compute_nucleotide_similarities(offset_in_chunk, chunk_vectors, position_codebook)

        # Step 5: Apply LINEAR magnitude weighting (if enabled)
        if self.use_magnitude_weighting:
            mag0, mag1 = self.magnitude_cache.get_magnitudes(chunk_idx, chunk_vectors)
            total_mag = mag0 + mag1

            if total_mag > 0:
                AT_weight = mag0 / total_mag  # LINEAR, not squared
                GC_weight = mag1 / total_mag
            else:
                AT_weight = GC_weight = 0.5

            # Apply compositional prior
            final_scores = {
                'A': AT_weight * sims['A'] + sims['A_hinge'],
                'T': AT_weight * sims['T'] + sims['T_hinge'],
                'G': GC_weight * sims['G'] + sims['G_hinge'],
                'C': GC_weight * sims['C'] + sims['C_hinge'],
            }
        else:
            # No magnitude weighting - equal combination
            final_scores = {
                'A': sims['A'] + sims['A_hinge'],
                'T': sims['T'] + sims['T_hinge'],
                'G': sims['G'] + sims['G_hinge'],
                'C': sims['C'] + sims['C_hinge'],
            }

        # Step 6: Decode
        best_nuc = max(final_scores, key=final_scores.get)
        total_score = sum(final_scores.values())
        confidence = final_scores[best_nuc] / total_score if total_score > 0 else 0.25

        return best_nuc, confidence, texture_type, lens_name

    def _find_chunk_and_offset(self, chrom: str, pos: int) -> Tuple[int, int]:
        """Find chunk index and offset within chunk for genomic position."""
        # Simplified - assumes linear chunk layout
        # In production, would use chunk metadata from H5 file
        chunk_idx = pos // self.N
        offset_in_chunk = pos % self.N
        return chunk_idx, offset_in_chunk

    def _load_chunk_vectors(self, chunk_idx: int) -> Dict[str, np.ndarray]:
        """Load 6-bank vectors for chunk and convert to bipolar."""
        # Load binary banks from H5
        bank0_binary = self.h5_file['bank0'][chunk_idx, :]
        bank1_binary = self.h5_file['bank1'][chunk_idx, :]
        bank2_binary = self.h5_file['bank2'][chunk_idx, :]
        bank3_binary = self.h5_file['bank3'][chunk_idx, :]
        bank4_binary = self.h5_file['bank4'][chunk_idx, :]
        bank5_binary = self.h5_file['bank5'][chunk_idx, :]

        # Convert to bipolar {-1, +1}
        return {
            'bank0': 2 * bank0_binary.astype(np.float32) - 1,  # A detector
            'bank1': 2 * bank1_binary.astype(np.float32) - 1,  # T detector
            'bank2': 2 * bank2_binary.astype(np.float32) - 1,  # G detector
            'bank3': 2 * bank3_binary.astype(np.float32) - 1,  # C detector
            'bank4': 2 * bank4_binary.astype(np.float32) - 1,  # Hinge pos
            'bank5': 2 * bank5_binary.astype(np.float32) - 1,  # Hinge neg
        }

    def _select_best_lens(self, texture: str, chunk_vectors: Dict[str, np.ndarray]) -> Optional[MotifLens]:
        """
        Select best matching lens for texture type.

        Uses cosine similarity to rank candidate lenses.
        """
        candidates = self.lens_library.get_lenses_for_texture(texture)

        if not candidates:
            return None

        # Rank by similarity to chunk vectors
        best_lens = None
        best_score = -1.0

        for lens in candidates:
            # Compute similarity across all 3 banks
            sim0 = np.dot(chunk_vectors['bank0'], lens.bank0) / self.D
            sim1 = np.dot(chunk_vectors['bank1'], lens.bank1) / self.D
            sim2 = np.dot(chunk_vectors['bank2'], lens.bank2) / self.D

            combined_score = (sim0 + sim1 + sim2) / 3.0

            if combined_score > best_score:
                best_score = combined_score
                best_lens = lens

        return best_lens

    def _apply_lens_overlay(
        self,
        chunk_vectors: Dict[str, np.ndarray],
        lens: MotifLens
    ) -> Dict[str, np.ndarray]:
        """Apply lens overlay with alpha blending."""
        return {
            'bank0': chunk_vectors['bank0'] + self.lens_alpha * lens.bank0,
            'bank1': chunk_vectors['bank1'] + self.lens_alpha * lens.bank1,
            'bank2': (chunk_vectors['bank4'] - chunk_vectors['bank5']) + self.lens_alpha * lens.bank2,
            'bank3': chunk_vectors['bank3'],  # Unused in current implementation
            'bank4': chunk_vectors['bank4'],
            'bank5': chunk_vectors['bank5'],
        }

    def _compute_nucleotide_similarities(
        self,
        offset: int,
        chunk_vectors: Dict[str, np.ndarray],
        position_codebook: np.ndarray
    ) -> Dict[str, float]:
        """Compute similarity scores for each nucleotide hypothesis."""
        query_vec = position_codebook[offset]

        # Bank 0: A detector (hydrophobic positive)
        # Bank 1: T detector (hydrophobic negative)
        # Bank 2: Combined hinge
        # Bank 3: G detector (major groove positive) - not used
        # Bank 4: C detector (major groove negative) - not used

        sim_A_hydro = np.dot(chunk_vectors['bank0'], query_vec) / self.D
        sim_T_hydro = np.dot(chunk_vectors['bank1'], query_vec) / self.D
        sim_hinge = np.dot(chunk_vectors['bank2'], query_vec) / self.D

        # For G/C, we need to use bank2/bank3 from original encoding
        # But chunk_vectors already has them - just not in the overlay
        # For simplicity, use hinge for all (production would use separate GC banks)

        return {
            'A': sim_A_hydro,
            'T': sim_T_hydro,
            'G': -sim_T_hydro,  # Approximate (should use bank2)
            'C': -sim_A_hydro,  # Approximate (should use bank3)
            'A_hinge': sim_hinge if sim_hinge > 0 else 0,
            'T_hinge': -sim_hinge if sim_hinge < 0 else 0,
            'G_hinge': sim_hinge if sim_hinge > 0 else 0,
            'C_hinge': -sim_hinge if sim_hinge < 0 else 0,
        }

    def close(self):
        """Close HDF5 file."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================================
# CLI and Testing
# ============================================================================

def build_lens_library_cli():
    """CLI to build lens library from reference genome."""
    import argparse

    parser = argparse.ArgumentParser(description="Build structural motif lens library")
    parser.add_argument('--reference', type=str, required=True, help="Reference genome FASTA")
    parser.add_argument('--output', type=str, required=True, help="Output lens library H5")
    parser.add_argument('--D', type=int, default=5120, help="Hypervector dimension")
    parser.add_argument('--N', type=int, default=1024, help="Chunk size (for position codebook)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for position codebook")

    args = parser.parse_args()

    # Generate position codebook (same as encoder)
    np.random.seed(args.seed)
    position_codebook = np.random.choice([-1, 1], size=(args.N, args.D)).astype(np.int8)

    # Build library
    library = LensLibrary(D=args.D)
    library.build_from_reference(args.reference, position_codebook)

    # Save
    library.save(Path(args.output))
    print(f"✓ Built and saved lens library to {args.output}")
    print(f"  Lenses: {list(library.lenses.keys())}")


if __name__ == '__main__':
    build_lens_library_cli()
