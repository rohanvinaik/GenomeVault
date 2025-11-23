"""
Build Offline Motif Index for Structural Elements

Uses zero-count density + magnitude ratios (NOT template matching) to identify
structural motifs in encoded genome. Position-dependent encoding makes template
matching impossible, but structural fingerprints work.

Author: Phase 1 Week 3 - Lens System Debugging
Date: November 21, 2025
"""

import h5py
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# GROUND TRUTH THRESHOLDS from barbie method - n=50 real genome samples
# Architecture: D=5120, N=1024, sparse position codebook (SNR=5.0)
# Encoded density: ~9% (median), range 7-10%
#
# REAL MAGNITUDE PROFILES (from actual encoded genome):
# GC-RICH:  bank1=18.67±1.30, bank2=24.07±1.19, bank3=20.10±0.82
# AT-RICH:  bank1=23.74±0.67, bank2=19.32±0.72, bank3=20.50±0.52
# BALANCED: bank1=21.77±0.33, bank2=21.78±0.35, bank3=20.28±0.45
# BASELINE: bank1=23.32±1.25, bank2=19.97±1.50, bank3=20.38±0.73
#
# Key insight: Magnitude profiles show biophysical signal even in sparse encoding

# GC-RICH thresholds (conservative: 2σ from baseline)
GC_BANK1_MAX = 20.0      # bank1 suppressed (real: 18.67)
GC_BANK2_MIN = 23.5      # bank2 elevated (real: 24.07)
GC_BANK3_MIN = 21.0      # bank3 elevated (real: 20.10)

# AT-RICH thresholds (conservative: 2σ from baseline)
AT_BANK1_MIN = 26.0      # bank1 elevated (real: 23.74)
AT_BANK2_MAX = 17.0      # bank2 suppressed (real: 19.32)
AT_BANK3_MAX = 19.5      # bank3 suppressed (real: 20.50)

TOLERANCE = 1.0          # ±1.0 tolerance around thresholds

# Confidence threshold for indexing
CONFIDENCE_THRESHOLD = 0.7  # 70% profile match required


def count_zeros(bank: np.ndarray) -> int:
    """Count number of zero elements in a bank."""
    return np.sum(bank == 0)


def compute_structural_fingerprint(banks: dict) -> Dict[str, float]:
    """
    Compute structural fingerprint from 3 banks.

    Returns RAW magnitudes (NOT normalized) - normalization destroys the biophysical signal!
    Our HDC implementation is unusual: we WANT the noise to be loud because it's actually signal.

    Returns:
        dict with keys: 'density', 'bank1_mag', 'bank2_mag', 'bank3_mag'
    """
    # Count total zeros across all banks
    total_zeros = sum(count_zeros(bank) for bank in banks.values())
    total_elements = sum(bank.size for bank in banks.values())
    density = 1 - (total_zeros / total_elements)

    # Compute RAW magnitudes (DO NOT NORMALIZE)
    bank_mags = {name: np.linalg.norm(bank) for name, bank in banks.items()}

    return {
        'density': density,
        'bank1_mag': bank_mags['bank1'],
        'bank2_mag': bank_mags['bank2'],
        'bank3_mag': bank_mags['bank3'],
    }


def classify_motif(fingerprint: Dict[str, float]) -> Tuple[str, float]:
    """
    Classify structural motif using ABSOLUTE magnitude profile matching.

    Based on barbie method ground truth (extract_motif_profiles.py):
    - GC-rich (ALU): bank2=24.84, bank3=21.19, bank1=19.82
    - AT-rich (poly-A): bank1=32.34, bank2=0.00, bank3=0.00

    The key insight: Match the INTERSECTION of all 3 banks to reflect sequence motifs.
    ALL 3 banks matter, not just "which one wins".

    Returns:
        (motif_type, confidence) or (None, 0.0) if no match
    """
    bank1_mag = fingerprint['bank1_mag']
    bank2_mag = fingerprint['bank2_mag']
    bank3_mag = fingerprint['bank3_mag']

    # GC-RICH motifs (ALU / CpG islands)
    # Profile: bank2 AND bank3 elevated, bank1 suppressed
    if (bank2_mag >= GC_BANK2_MIN - TOLERANCE and
        bank3_mag >= GC_BANK3_MIN - TOLERANCE and
        bank1_mag <= GC_BANK1_MAX + TOLERANCE):

        # Calculate confidence based on how well it matches the profile
        bank2_match = 1.0 - abs(bank2_mag - GC_BANK2_MIN) / TOLERANCE
        bank3_match = 1.0 - abs(bank3_mag - GC_BANK3_MIN) / TOLERANCE
        bank1_suppress = 1.0 - max(0, bank1_mag - GC_BANK1_MAX) / TOLERANCE
        confidence = (bank2_match + bank3_match + bank1_suppress) / 3.0

        # Distinguish ALU (bank2 slightly higher) vs CpG (bank3 slightly higher)
        if bank2_mag > bank3_mag:
            return ('ALU_MAJOR_GROOVE', confidence)
        else:
            return ('CPG_HINGE', confidence)

    # AT-RICH motifs (poly-A tails / AT repeats)
    # Profile: bank1 maxed out, bank2 AND bank3 suppressed (near zero)
    elif (bank1_mag >= AT_BANK1_MIN - TOLERANCE and
          bank2_mag <= AT_BANK2_MAX + TOLERANCE and
          bank3_mag <= AT_BANK3_MAX + TOLERANCE):

        # Calculate confidence
        bank1_match = 1.0 - abs(bank1_mag - AT_BANK1_MIN) / TOLERANCE
        bank2_suppress = 1.0 - max(0, bank2_mag - AT_BANK2_MAX) / TOLERANCE
        bank3_suppress = 1.0 - max(0, bank3_mag - AT_BANK3_MAX) / TOLERANCE
        confidence = (bank1_match + bank2_suppress + bank3_suppress) / 3.0

        return ('POLYA_HYDROPHOBIC', confidence)

    # No structural motif match
    else:
        return (None, 0.0)


def build_motif_index(h5_path: str, output_path: str = None):
    """
    Build offline motif index by scanning all chunks.

    Args:
        h5_path: Path to encoded genome H5 file
        output_path: Path to save motif index JSON (default: output/motif_index.json)
    """
    if output_path is None:
        output_path = "genomevault/hdv_validation/hdc_experimentation/output/motif_index.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building motif index from {h5_path}")

    motif_index = {}
    stats = {
        'total_chunks': 0,
        'indexed_chunks': 0,
        # GC-rich motifs (98-100% dense)
        'ALU_MAJOR_GROOVE': 0,
        'CPG_HINGE': 0,
        'DENSE_GC_RICH': 0,
        # Medium density motifs (90-98% dense)
        'LINE_PURINE': 0,
        'DENSE_REPEAT': 0,
        # AT-rich motifs (0-20% dense)
        'POLYA_HYDROPHOBIC': 0,
        'SPARSE_AT_RICH': 0,
    }

    with h5py.File(h5_path, 'r') as f:
        all_banks = f['all_bank_vectors']
        total_chunks = all_banks.shape[0]
        stats['total_chunks'] = total_chunks

        logger.info(f"Scanning {total_chunks:,} chunks...")

        for chunk_idx in range(total_chunks):
            if chunk_idx % 10000 == 0:
                logger.info(f"Progress: {chunk_idx:,}/{total_chunks:,} ({100*chunk_idx/total_chunks:.1f}%)")

            # Load chunk banks
            all_banks_data = all_banks[chunk_idx, :, :]
            banks = {
                'bank1': all_banks_data[0, :],
                'bank2': all_banks_data[1, :],
                'bank3': all_banks_data[2, :],
            }

            # Compute structural fingerprint
            fingerprint = compute_structural_fingerprint(banks)

            # Classify motif
            motif_type, confidence = classify_motif(fingerprint)

            # Only index if high confidence
            if motif_type and confidence > CONFIDENCE_THRESHOLD:
                motif_index[chunk_idx] = {
                    'motif_type': motif_type,
                    'confidence': float(confidence),
                    'fingerprint': {k: float(v) for k, v in fingerprint.items()},
                }
                stats['indexed_chunks'] += 1
                stats[motif_type] += 1

    # Save index to JSON
    index_data = {
        'motif_index': motif_index,
        'stats': stats,
        'thresholds': {
            'GC_BANK1_MAX': GC_BANK1_MAX,
            'GC_BANK2_MIN': GC_BANK2_MIN,
            'GC_BANK3_MIN': GC_BANK3_MIN,
            'AT_BANK1_MIN': AT_BANK1_MIN,
            'AT_BANK2_MAX': AT_BANK2_MAX,
            'AT_BANK3_MAX': AT_BANK3_MAX,
            'TOLERANCE': TOLERANCE,
            'CONFIDENCE_THRESHOLD': CONFIDENCE_THRESHOLD,
        },
    }

    with open(output_path, 'w') as f:
        json.dump(index_data, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info(f"MOTIF INDEX BUILD COMPLETE (RAW MAGNITUDES)")
    logger.info(f"{'='*80}")
    logger.info(f"Total chunks:       {stats['total_chunks']:,}")
    logger.info(f"Indexed chunks:     {stats['indexed_chunks']:,} ({100*stats['indexed_chunks']/stats['total_chunks']:.2f}%)")
    logger.info(f"")
    logger.info(f"Motif breakdown (density + raw magnitude patterns):")
    logger.info(f"")
    logger.info(f"GC-rich (98-100% dense):")
    logger.info(f"  ALU_MAJOR_GROOVE:        {stats['ALU_MAJOR_GROOVE']:,} (bank2 dominant)")
    logger.info(f"  CPG_HINGE:               {stats['CPG_HINGE']:,} (bank3 dominant)")
    logger.info(f"  DENSE_GC_RICH:           {stats['DENSE_GC_RICH']:,} (balanced)")
    logger.info(f"")
    logger.info(f"Medium density (90-98% dense):")
    logger.info(f"  LINE_PURINE:             {stats['LINE_PURINE']:,} (bank3 dominant)")
    logger.info(f"  DENSE_REPEAT:            {stats['DENSE_REPEAT']:,} (balanced)")
    logger.info(f"")
    logger.info(f"AT-rich (0-20% dense):")
    logger.info(f"  POLYA_HYDROPHOBIC:       {stats['POLYA_HYDROPHOBIC']:,} (bank1 dominant)")
    logger.info(f"  SPARSE_AT_RICH:          {stats['SPARSE_AT_RICH']:,} (balanced)")
    logger.info(f"")
    logger.info(f"Index size: {len(json.dumps(index_data)):,} bytes")
    logger.info(f"Saved to: {output_path}")
    logger.info(f"{'='*80}")

    return motif_index, stats


if __name__ == '__main__':
    # Build index from encoded genome
    h5_path = "genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_3banks.h5"

    if not Path(h5_path).exists():
        logger.error(f"ERROR: Encoded genome not found at {h5_path}")
        logger.error("Run encode_3bank_split_architecture.py first!")
        exit(1)

    motif_index, stats = build_motif_index(h5_path)
