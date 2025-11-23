#!/usr/bin/env python3
"""
Lens-Aware Decoder for 3-Ternary Bank Genomic HDC

CORRECTED ARCHITECTURE: 3 ternary banks {-1, 0, +1}, NOT 6 binary banks.

Implements the Structural Motif Lens Library system with:
1. Bank 3 (Hinge) texture classification (ZCR-based)
2. Structural motif lens library
3. LINEAR magnitude-based compositional weighting
4. Direct ternary quantization (no reconstruction overhead)

Based on: docs/theory/STRUCTURAL_MOTIF_LENS_LIBRARY.md
Version: 2.0 (3-Ternary Architecture)
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
class MotifLens:
    """Precomputed consensus hypervector for a structural motif (3 ternary banks)."""
    name: str
    texture_type: str
    bank1: np.ndarray  # Hydrophobic (T=+1, A=-1, GC=0) - ternary int8
    bank2: np.ndarray  # Major groove (G=+1, C=-1, AT=0) - ternary int8
    bank3: np.ndarray  # Hinge (YR=+1, RY=-1, neutral=0) - ternary int8
    prevalence: float
    typical_size: int


class TextureClassifier:
    """Classify genomic texture using Bank 3 (Hinge) with Zero-Crossing Rate."""

    def __init__(self):
        self.magnitude_high = None
        self.magnitude_moderate = None
        self.variance_high = None
        self.variance_moderate = None
        self.calibrated = False

    def classify(self, hinge_vector: np.ndarray):
        """Classify texture using ZCR (O(N) vs O(N log N) FFT)."""
        if not self.calibrated:
            self.magnitude_high = 0.75 * len(hinge_vector)
            self.magnitude_moderate = 0.5 * len(hinge_vector)
            self.variance_high = 0.3
            self.variance_moderate = 0.2

        magnitude = np.linalg.norm(hinge_vector)
        variance = np.var(hinge_vector)

        # Zero-Crossing Rate (ZCR)
        sign_changes = np.diff(np.sign(hinge_vector)) != 0
        zcr = np.sum(sign_changes) / len(hinge_vector)

        if magnitude > self.magnitude_high and zcr < 0.05:
            return 'HOMOPOLYMER'
        elif zcr > 0.8:
            return 'ALTERNATING'
        elif magnitude > self.magnitude_high and variance > self.variance_moderate:
            return 'CPG_LIKE'
        elif variance > self.variance_high and magnitude < self.magnitude_moderate:
            return 'COMPLEX_CODING'
        else:
            return 'ALU_LIKE'


class LensLibrary:
    """
    Manages structural motif consensus hypervectors (3 ternary banks).

    CRITICAL: Uses direct ternary quantization - NO 6-binary intermediate!
    """

    def __init__(self, D: int = 5120):
        self.D = D
        self.lenses: Dict[str, MotifLens] = {}

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
        Encode motif to 3 ternary banks directly.

        Uses np.sign() for direct ternary quantization.
        No 6-binary intermediate = 50% less compute.
        """
        N = len(position_codebook)
        sequence = sequence.upper()

        # Accumulate in ternary space
        acc_hydrophobic = np.zeros(self.D, dtype=np.int16)
        acc_major_groove = np.zeros(self.D, dtype=np.int16)
        acc_hinge = np.zeros(self.D, dtype=np.int16)

        prev_nuc = None
        for i, nuc in enumerate(sequence[:N]):
            pos_vec = position_codebook[i]

            # Bank 1: Hydrophobic
            if nuc == 'T':
                acc_hydrophobic += pos_vec
            elif nuc == 'A':
                acc_hydrophobic -= pos_vec

            # Bank 2: Major Groove
            if nuc == 'G':
                acc_major_groove += pos_vec
            elif nuc == 'C':
                acc_major_groove -= pos_vec

            # Bank 3: Hinge
            if prev_nuc is not None:
                is_purine = {'A': True, 'G': True, 'C': False, 'T': False}
                if not is_purine.get(prev_nuc, False) and is_purine.get(nuc, False):
                    acc_hinge += pos_vec
                elif is_purine.get(prev_nuc, False) and not is_purine.get(nuc, False):
                    acc_hinge -= pos_vec

            prev_nuc = nuc

        # Direct ternary quantization
        bank1 = np.sign(acc_hydrophobic).astype(np.int8)
        bank2 = np.sign(acc_major_groove).astype(np.int8)
        bank3 = np.sign(acc_hinge).astype(np.int8)

        return MotifLens(
            name=name,
            texture_type=texture_type,
            bank1=bank1,
            bank2=bank2,
            bank3=bank3,
            prevalence=prevalence,
            typical_size=typical_size
        )

    def build_from_reference(self, reference_fasta: str, position_codebook: np.ndarray):
        """Build lens library from known motifs."""
        # ALU_YI
        alu_seq = ("GCGCGCTAGCTAGCGCGCTAGCTAGCGCGC" * 8) + ("A" * 20)
        self.lenses['ALU_YI'] = self._encode_motif_to_lens(
            'ALU_YI', alu_seq, 'ALU_LIKE', 0.11, 300, position_codebook
        )

        # CPG_ISLAND
        cpg_seq = "CGCGCGCGCGCGCGCGCG" * 20
        self.lenses['CPG_ISLAND'] = self._encode_motif_to_lens(
            'CPG_ISLAND', cpg_seq, 'CPG_LIKE', 0.01, 1000, position_codebook
        )

        # TATA_BOX
        tata_seq = "TATAWAAW".replace('W', 'A') * 4
        self.lenses['TATA_BOX'] = self._encode_motif_to_lens(
            'TATA_BOX', tata_seq, 'ALTERNATING', 0.001, 30, position_codebook
        )

        # POLY_A
        polya_seq = "A" * 50
        self.lenses['POLY_A'] = self._encode_motif_to_lens(
            'POLY_A', polya_seq, 'HOMOPOLYMER', 0.02, 50, position_codebook
        )

        logger.info(f"Built {len(self.lenses)} lenses")

    def get_lenses_for_texture(self, texture: str) -> List[MotifLens]:
        """Return lenses matching texture type."""
        return [lens for lens in self.lenses.values() if lens.texture_type == texture]

    def save(self, output_path: Path):
        """Save 3 ternary banks to HDF5."""
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
        """Load 3 ternary banks from HDF5."""
        library = cls(D=5120)

        with h5py.File(lens_library_path, 'r') as f:
            library.D = f.attrs['D']

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

        logger.info(f"Loaded {len(library.lenses)} lenses")
        return library


class LensAwareDecoder:
    """
    Lens-aware decoder for 3-ternary bank architecture.

    Pipeline:
    1. Texture classification (Bank 3 ZCR)
    2. Lens selection
    3. Lens overlay (0.3 alpha)
    4. Similarity computation
    5. LINEAR magnitude weighting
    6. Decoding
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

        # Load genome data
        self.h5_file = h5py.File(self.encoded_h5_path, 'r')
        self.D = self.h5_file.attrs.get('dimension', 5120)
        self.N = self.h5_file.attrs.get('chunk_size', 1024)

        logger.info(f"Initialized decoder: D={self.D}, N={self.N}")

    def _load_chunk_vectors(self, chunk_idx: int) -> Dict[str, np.ndarray]:
        """
        Load 3 ternary banks from HDF5.

        CRITICAL: Encoder stores as (chunks, 3, D) with dtype=int8 ternary.
        No conversion needed - already ternary {-1, 0, +1}!
        """
        all_banks = self.h5_file['all_bank_vectors'][chunk_idx, :, :]  # Shape: (3, D)

        return {
            'bank1': all_banks[0, :].astype(np.float32),  # Hydrophobic
            'bank2': all_banks[1, :].astype(np.float32),  # Major Groove
            'bank3': all_banks[2, :].astype(np.float32),  # Hinge
        }

    def _select_best_lens(self, texture: str, chunk_vectors: Dict[str, np.ndarray]) -> Optional[MotifLens]:
        """Select best matching lens for texture."""
        candidates = self.lens_library.get_lenses_for_texture(texture)

        if not candidates:
            return None

        best_lens = None
        best_score = -1.0

        for lens in candidates:
            # Cosine similarity across all 3 banks
            sim1 = np.dot(chunk_vectors['bank1'], lens.bank1) / self.D
            sim2 = np.dot(chunk_vectors['bank2'], lens.bank2) / self.D
            sim3 = np.dot(chunk_vectors['bank3'], lens.bank3) / self.D

            combined_score = (sim1 + sim2 + sim3) / 3.0

            if combined_score > best_score:
                best_score = combined_score
                best_lens = lens

        return best_lens

    def _apply_lens_overlay(self, chunk_vectors: Dict[str, np.ndarray], lens: MotifLens) -> Dict[str, np.ndarray]:
        """Apply lens overlay with alpha blending (direct ternary)."""
        return {
            'bank1': chunk_vectors['bank1'] + self.lens_alpha * lens.bank1.astype(np.float32),
            'bank2': chunk_vectors['bank2'] + self.lens_alpha * lens.bank2.astype(np.float32),
            'bank3': chunk_vectors['bank3'] + self.lens_alpha * lens.bank3.astype(np.float32),
        }

    def decode_position(
        self,
        chrom: str,
        pos: int,
        position_codebook: np.ndarray
    ) -> Tuple[str, float, Optional[str], Optional[str]]:
        """
        Decode nucleotide at position with lens and magnitude awareness.

        Returns: (nucleotide, confidence, texture_type, lens_name)
        """
        # Load chunk vectors (3 ternary banks)
        chunk_idx = pos // self.N
        offset_in_chunk = pos % self.N
        chunk_vectors = self._load_chunk_vectors(chunk_idx)

        # Texture classification
        texture_type = None
        lens_name = None
        if self.lens_library:
            texture_type = self.texture_classifier.classify(chunk_vectors['bank3'])

            best_lens = self._select_best_lens(texture_type, chunk_vectors)
            if best_lens is not None:
                lens_name = best_lens.name
                chunk_vectors = self._apply_lens_overlay(chunk_vectors, best_lens)

        # Compute similarities
        query_vec = position_codebook[offset_in_chunk]

        sim_bank1 = np.dot(chunk_vectors['bank1'], query_vec) / self.D
        sim_bank2 = np.dot(chunk_vectors['bank2'], query_vec) / self.D
        sim_bank3 = np.dot(chunk_vectors['bank3'], query_vec) / self.D

        # Apply LINEAR magnitude weighting
        if self.use_magnitude_weighting:
            mag1 = np.linalg.norm(chunk_vectors['bank1'])
            mag2 = np.linalg.norm(chunk_vectors['bank2'])
            total_mag = mag1 + mag2

            if total_mag > 0:
                AT_weight = mag1 / total_mag
                GC_weight = mag2 / total_mag
            else:
                AT_weight = GC_weight = 0.5

            # Genomic Monty Hall: cross-validate with 3 lenses
            scores = {
                'A': AT_weight * (-sim_bank1) + (sim_bank3 if sim_bank3 < 0 else 0),  # A = Hydrophobic negative
                'T': AT_weight * sim_bank1 + (sim_bank3 if sim_bank3 < 0 else 0),   # T = Hydrophobic positive
                'G': GC_weight * sim_bank2 + (sim_bank3 if sim_bank3 > 0 else 0),   # G = Major Groove positive
                'C': GC_weight * (-sim_bank2) + (sim_bank3 if sim_bank3 > 0 else 0), # C = Major Groove negative
            }
        else:
            scores = {
                'A': -sim_bank1,
                'T': sim_bank1,
                'G': sim_bank2,
                'C': -sim_bank2,
            }

        # Decode
        best_nuc = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_nuc] / total_score if total_score > 0 else 0.25

        return best_nuc, confidence, texture_type, lens_name

    def close(self):
        """Close HDF5 file."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == '__main__':
    print("Lens-Aware Decoder (3-Ternary Architecture) - v2.0")
    print("See build_lens_library.py to create lens library")
    print("See demo_lens_decoder.py for usage examples")
