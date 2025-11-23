#!/usr/bin/env python3
"""
Extract Biological Motifs from Real Genomic Data

This script scans chr22 for REAL biological motif consensus sequences
(TATA_BOX, CAAT_BOX, GC_BOX, ALU_CONSENSUS, etc.) and extracts their
HDC bank signatures for "corrective lenses" calibration.

Purpose: Unlike synthetic motif categories (GC_SUPPRESS, AT_SUPPRESS),
this finds ACTUAL biological sequences and their interactions with the
HDC encoding system.

Author: Phase 1 Week 3-4
Date: November 22, 2025
"""

import h5py
import json
import gzip
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging
import re
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === BIOLOGICAL MOTIF CONSENSUS SEQUENCES ===
# These are REAL regulatory motifs found in human genomic DNA
BIOLOGICAL_MOTIFS = {
    'TATA_BOX': {
        'consensus': 'TATAAA',
        'description': 'TATA box promoter element (-25 to -30)',
        'allow_mismatch': 1,  # Allow 1bp mismatch for biological variation
    },
    'CAAT_BOX': {
        'consensus': 'CCAAT',
        'description': 'CAAT box promoter element',
        'allow_mismatch': 1,
    },
    'GC_BOX': {
        'consensus': 'GGGCGG',
        'description': 'GC box (Sp1 binding site)',
        'allow_mismatch': 1,
    },
    'POLY_A_SIGNAL': {
        'consensus': 'AATAAA',
        'description': 'Polyadenylation signal',
        'allow_mismatch': 1,
    },
    'ALU_CONSENSUS_5': {
        'consensus': 'GGCCGGGCGCGGTGGCTCACGCC',
        'description': 'Alu repeat consensus (5\' end, 23bp)',
        'allow_mismatch': 2,
    },
    'LINE1_5': {
        'consensus': 'GGAGCCAAGATGGCCGAATAGGG',
        'description': 'LINE-1 consensus (5\' end, 23bp)',
        'allow_mismatch': 2,
    },
    'CPG_ISLAND': {
        'consensus': 'CGCGCGCGCG',
        'description': 'CpG island motif (10bp CG-rich)',
        'allow_mismatch': 1,
    },
}

# Encoding parameters (from encode_3bank_split_architecture.py)
STRIDE = 896  # bp between chunk starts
CHUNK_SIZE = 1024  # bp per chunk
OVERLAP = 128  # bp overlap between chunks


def load_reference_genome(fasta_path: str) -> Tuple[str, str]:
    """
    Load reference genome FASTA file.

    Returns:
        (chromosome_name, sequence_string)
    """
    logger.info(f"Loading reference genome: {fasta_path}")

    if fasta_path.endswith('.gz'):
        opener = gzip.open
    else:
        opener = open

    with opener(fasta_path, 'rt') as f:
        lines = f.readlines()

    # First line is header (>chr22)
    chr_name = lines[0].strip().lstrip('>')

    # Remaining lines are sequence
    sequence = ''.join(line.strip().upper() for line in lines[1:])

    logger.info(f"Loaded {chr_name}: {len(sequence):,} bp")

    return chr_name, sequence


def find_motif_matches(sequence: str, motif_name: str, motif_def: dict) -> List[int]:
    """
    Find all occurrences of a motif consensus sequence in the genome.

    Args:
        sequence: Full chromosome sequence
        motif_name: Name of the motif
        motif_def: Motif definition dict with 'consensus' and 'allow_mismatch'

    Returns:
        List of genomic positions (0-based) where motif matches
    """
    consensus = motif_def['consensus']
    max_mismatch = motif_def['allow_mismatch']

    logger.info(f"Scanning for {motif_name}: {consensus} (max {max_mismatch} mismatches)")

    matches = []
    cons_len = len(consensus)

    # Scan entire sequence
    for pos in range(len(sequence) - cons_len + 1):
        window = sequence[pos:pos + cons_len]

        # Count mismatches
        mismatches = sum(1 for a, b in zip(window, consensus) if a != b and a != 'N')

        if mismatches <= max_mismatch:
            matches.append(pos)

    logger.info(f"  Found {len(matches):,} matches for {motif_name}")

    return matches


def position_to_chunk_idx(genomic_pos: int) -> int:
    """
    Convert genomic position to chunk index.

    Encoding uses:
    - Chunk 0: position 0-1023
    - Chunk 1: position 896-1919 (STRIDE=896bp overlap)
    - Chunk i: position (i * STRIDE) to (i * STRIDE + 1023)

    For a given position, return the chunk that contains it
    (use the chunk where position is closest to center).
    """
    # Which chunk has this position closest to its center?
    # Chunk center is at: chunk_idx * STRIDE + CHUNK_SIZE/2

    chunk_idx = int(genomic_pos / STRIDE)

    return chunk_idx


def load_split_binary_h5(h5_path: str) -> dict:
    """
    Load the split binary H5 file with 6 banks.

    Returns dict with bank arrays and metadata.
    """
    logger.info(f"Loading encoded genome: {h5_path}")

    f = h5py.File(h5_path, 'r')

    # Load packed format: (n_chunks, 6, D)
    packed_data = f['binary_bank_vectors'][:]
    n_chunks = packed_data.shape[0]

    logger.info(f"Loaded {n_chunks:,} chunks with {packed_data.shape[2]} byte-packed dimensions")

    # Extract individual banks (axis 1 indexing)
    return {
        'bank1_pos': packed_data[:, 0, :],  # Hydrophobic T
        'bank1_neg': packed_data[:, 1, :],  # Hydrophobic A
        'bank2_pos': packed_data[:, 2, :],  # Major groove G
        'bank2_neg': packed_data[:, 3, :],  # Major groove C
        'bank3_pos': packed_data[:, 4, :],  # Hinge Y→R
        'bank3_neg': packed_data[:, 5, :],  # Hinge R→Y
        'file': f  # Keep handle for cleanup
    }


def compute_bank_magnitudes(chunk_idx: int, data: dict) -> dict:
    """
    Compute RAW ACTIVATION COUNTS for each of the 6 banks.

    Returns dict with bank*_*_mag keys.
    """
    mags = {}

    # Extract each bank's vector (D dimensions, binary {0,1})
    bank1_pos_vec = data['bank1_pos'][chunk_idx]
    bank1_neg_vec = data['bank1_neg'][chunk_idx]
    bank2_pos_vec = data['bank2_pos'][chunk_idx]
    bank2_neg_vec = data['bank2_neg'][chunk_idx]
    bank3_pos_vec = data['bank3_pos'][chunk_idx]
    bank3_neg_vec = data['bank3_neg'][chunk_idx]

    # Compute RAW ACTIVATION COUNTS (Hamming weight)
    mags['bank1_pos_mag'] = float(np.sum(bank1_pos_vec))
    mags['bank1_neg_mag'] = float(np.sum(bank1_neg_vec))
    mags['bank2_pos_mag'] = float(np.sum(bank2_pos_vec))
    mags['bank2_neg_mag'] = float(np.sum(bank2_neg_vec))
    mags['bank3_pos_mag'] = float(np.sum(bank3_pos_vec))
    mags['bank3_neg_mag'] = float(np.sum(bank3_neg_vec))

    return mags


def extract_composition_from_sequence(sequence: str, genomic_pos: int) -> dict:
    """
    Extract nucleotide composition from the genomic sequence.

    Uses the chunk window (1024bp) centered around the position.
    """
    # Get chunk boundaries
    chunk_idx = position_to_chunk_idx(genomic_pos)
    chunk_start = chunk_idx * STRIDE
    chunk_end = chunk_start + CHUNK_SIZE

    # Extract chunk sequence
    chunk_seq = sequence[chunk_start:chunk_end]

    # Count nucleotides
    total = len(chunk_seq)
    a_count = chunk_seq.count('A')
    t_count = chunk_seq.count('T')
    g_count = chunk_seq.count('G')
    c_count = chunk_seq.count('C')

    # Convert to percentages
    return {
        'A_pct': (a_count / total) * 100,
        'T_pct': (t_count / total) * 100,
        'G_pct': (g_count / total) * 100,
        'C_pct': (c_count / total) * 100,
    }


def extract_biological_motifs(
    fasta_path: str,
    h5_path: str,
    output_path: str,
    n_samples_per_motif: int = 100,
    sample_spacing: int = 10000,  # Minimum bp between samples
):
    """
    Extract biological motif signatures from real genomic data.

    Args:
        fasta_path: Path to chr22 reference FASTA
        h5_path: Path to encoded genome H5
        output_path: Output JSON path
        n_samples_per_motif: Number of samples to collect per motif type
        sample_spacing: Minimum genomic distance between samples (to ensure independence)
    """
    logger.info("\n" + "="*80)
    logger.info("BIOLOGICAL MOTIF EXTRACTION FROM REAL GENOMIC DATA")
    logger.info("="*80)

    # Load reference genome
    chr_name, sequence = load_reference_genome(fasta_path)

    # Load encoded genome
    encoded_data = load_split_binary_h5(h5_path)

    # Find all motif matches
    motif_matches = {}
    for motif_name, motif_def in BIOLOGICAL_MOTIFS.items():
        matches = find_motif_matches(sequence, motif_name, motif_def)
        motif_matches[motif_name] = matches

    # Sample motif instances (spatially separated)
    logger.info(f"\nSampling {n_samples_per_motif} instances per motif (min {sample_spacing}bp apart)...")

    ground_truth = {}

    for motif_name, matches in motif_matches.items():
        logger.info(f"\n{motif_name}:")

        if len(matches) < n_samples_per_motif:
            logger.warning(f"  Only found {len(matches)} matches, using all")
            selected_positions = matches
        else:
            # Sample spatially-separated instances
            selected_positions = []
            sorted_matches = sorted(matches)

            for pos in sorted_matches:
                # Check if far enough from previous samples
                if not selected_positions or (pos - selected_positions[-1]) >= sample_spacing:
                    selected_positions.append(pos)

                if len(selected_positions) >= n_samples_per_motif:
                    break

            logger.info(f"  Selected {len(selected_positions)} spatially-separated samples")

        # Extract HDC signatures for each sample
        chunks = []
        for genomic_pos in selected_positions:
            chunk_idx = position_to_chunk_idx(genomic_pos)

            # Skip if chunk index out of bounds
            if chunk_idx >= len(encoded_data['bank1_pos']):
                logger.warning(f"  Skipping position {genomic_pos} (chunk {chunk_idx} out of bounds)")
                continue

            # Compute bank magnitudes
            signals = compute_bank_magnitudes(chunk_idx, encoded_data)

            # Extract composition from sequence
            composition = extract_composition_from_sequence(sequence, genomic_pos)

            chunks.append({
                'chunk_idx': int(chunk_idx),
                'genomic_position': int(genomic_pos),
                'position': f'chunk_{chunk_idx:07d}',
                'signals': signals,
                'composition': composition,
            })

        ground_truth[motif_name] = {
            'consensus': BIOLOGICAL_MOTIFS[motif_name]['consensus'],
            'description': BIOLOGICAL_MOTIFS[motif_name]['description'],
            'total_matches': len(matches),
            'samples_collected': len(chunks),
            'chunks': chunks,
        }

        logger.info(f"  Collected {len(chunks)} samples with HDC signatures")

    # Save ground truth
    logger.info(f"\nSaving biological motif ground truth to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("BIOLOGICAL MOTIF EXTRACTION COMPLETE")
    logger.info("="*80)

    total_samples = sum(len(data['chunks']) for data in ground_truth.values())
    logger.info(f"Total samples collected: {total_samples}")

    for motif_name, data in ground_truth.items():
        logger.info(f"  {motif_name}: {data['samples_collected']} samples (from {data['total_matches']:,} total matches)")

    logger.info(f"\nOutput saved to: {output_path}")

    # Cleanup
    encoded_data['file'].close()


if __name__ == '__main__':
    # Paths
    fasta_path = "data/downloaded/reference/hg38_chr22.fa.gz"
    h5_path = "genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_6banks_split_binary.h5"
    output_path = "genomevault/hdv_validation/hdc_experimentation/output/biological_motif_ground_truth.json"

    # Run extraction
    extract_biological_motifs(
        fasta_path=fasta_path,
        h5_path=h5_path,
        output_path=output_path,
        n_samples_per_motif=100,  # Collect 100 samples per motif type
        sample_spacing=10000,  # 10kb minimum spacing for independence
    )
