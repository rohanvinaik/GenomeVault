"""
Sample Chunks Across Full Composition Spectrum (AT and GC pathways)

Goal: Build COMPLETE transfer functions for SPLIT BINARY architecture.

Strategy:
1. AT Pathway: Sample chunks by AT content percentiles (0%, 25%, 50%, 75%, 100%)
   - Analyze Bank1_pos (A contributions) vs A%
   - Analyze Bank1_neg (T contributions) vs T%

2. GC Pathway: Sample chunks by GC content percentiles (0%, 25%, 50%, 75%, 100%)
   - Analyze Bank2_pos (G contributions) vs G%
   - Analyze Bank2_neg (C contributions) vs C%

3. Bank3 (Hinge): Analyze Y-R dinucleotide contributions

The AT and GC pathways are ORTHOGONAL - analyzed separately, combined via dot product at query time.

Author: Phase 1 Week 3 - Complete Degradation Curves
Date: November 22, 2025
"""

import h5py
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import pysam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_compositions_for_sample(
    encoded_genome_path: str,
    reference_fasta: str,
    sample_size: int = 100000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate AT% and GC% for a large sample of chunks to establish percentiles.

    Args:
        encoded_genome_path: Path to encoded genome H5 file
        reference_fasta: Path to reference genome FASTA
        sample_size: Number of chunks to sample

    Returns:
        (chunk_indices, at_percentages, gc_percentages)
    """
    logger.info(f"Calculating compositions for {sample_size} chunks...")

    with h5py.File(encoded_genome_path, 'r') as f:
        total_chunks = f['all_bank_vectors'].shape[0]
        chunk_keys = f['chunk_keys'][:]

        # Sample random chunks
        sample_indices = np.random.choice(total_chunks, size=min(sample_size, total_chunks), replace=False)

        at_percentages = []
        gc_percentages = []

        for idx in sample_indices:
            # Get genomic position
            chunk_key = chunk_keys[idx].decode('utf-8')
            chrom, coords = chunk_key.split(':')
            start, end = map(int, coords.split('-'))

            # Extract sequence
            sequence = extract_sequence(reference_fasta, chrom, start, end)
            if len(sequence) == 0:
                # Skip chunks with no sequence
                continue

            # Calculate composition
            seq_len = len(sequence)
            at_count = sequence.count('A') + sequence.count('T')
            gc_count = sequence.count('G') + sequence.count('C')

            at_pct = (at_count / seq_len) * 100
            gc_pct = (gc_count / seq_len) * 100

            at_percentages.append(at_pct)
            gc_percentages.append(gc_pct)

        logger.info(f"AT% range: {min(at_percentages):.1f}% - {max(at_percentages):.1f}%")
        logger.info(f"GC% range: {min(gc_percentages):.1f}% - {max(gc_percentages):.1f}%")

        return sample_indices, np.array(at_percentages), np.array(gc_percentages)


def sample_chunks_by_composition(
    chunk_indices: np.ndarray,
    compositions: np.ndarray,
    pathway_name: str,
    n_per_bin: int = 100
) -> Dict[str, np.ndarray]:
    """
    Sample chunks from each composition percentile bin.

    Args:
        chunk_indices: Array of chunk indices
        compositions: Array of composition percentages (AT% or GC%)
        pathway_name: "AT" or "GC"
        n_per_bin: Number of chunks to sample per bin

    Returns:
        Dict mapping bin names to chunk indices
    """
    percentiles = [0, 25, 50, 75, 100]
    comp_thresholds = np.percentile(compositions, percentiles)

    sampled_chunks = {}

    for i in range(len(percentiles) - 1):
        bin_name = f"{pathway_name}_p{percentiles[i]}-{percentiles[i+1]}"
        low, high = comp_thresholds[i], comp_thresholds[i+1]

        # Find chunks in this composition range
        mask = (compositions >= low) & (compositions <= high)
        bin_indices = chunk_indices[mask]

        # Sample from this bin
        if len(bin_indices) > n_per_bin:
            sampled = np.random.choice(bin_indices, size=n_per_bin, replace=False)
        else:
            sampled = bin_indices

        sampled_chunks[bin_name] = sampled
        logger.info(f"{bin_name}: sampled {len(sampled)} chunks ({pathway_name} {low:.1f}%-{high:.1f}%)")

    return sampled_chunks


def count_yr_dinucleotides(sequence: str) -> float:
    """
    Count Y-R (pyrimidine-purine) dinucleotide frequency.

    Y (pyrimidine): C, T
    R (purine): A, G
    YR dinucleotides: CA, CG, TA, TG

    Returns:
        YR dinucleotide percentage
    """
    if len(sequence) < 2:
        return 0.0

    yr_count = 0
    for i in range(len(sequence) - 1):
        dinuc = sequence[i:i+2]
        if dinuc in ['CA', 'CG', 'TA', 'TG']:
            yr_count += 1

    return (yr_count / (len(sequence) - 1)) * 100


def extract_sequence(fasta_path: str, chrom: str, start: int, end: int) -> str:
    """Extract DNA sequence from reference FASTA."""
    try:
        fasta = pysam.FastaFile(fasta_path)

        # Handle consensus naming convention
        if chrom in fasta.references:
            fetch_chrom = chrom
        elif f"{chrom}_consensus" in fasta.references:
            fetch_chrom = f"{chrom}_consensus"
        else:
            fasta.close()
            return ""

        chrom_length = fasta.get_reference_length(fetch_chrom)
        if start >= chrom_length or end > chrom_length:
            fasta.close()
            return ""

        sequence = fasta.fetch(fetch_chrom, start, end)
        fasta.close()
        return sequence.upper()
    except:
        return ""


def analyze_full_spectrum(
    encoded_genome_path: str,
    reference_fasta: str,
    output_file: str,
    composition_sample_size: int = 100000,
    chunks_per_bin: int = 100
):
    """
    Analyze chunks across full composition spectrum for AT and GC pathways.

    Args:
        encoded_genome_path: Path to encoded genome H5 file
        reference_fasta: Path to reference genome FASTA
        output_file: Where to save results
        composition_sample_size: Number of chunks to sample for composition calculation
        chunks_per_bin: Number of chunks to sample per composition bin
    """
    # Step 1: Calculate compositions for large sample
    chunk_indices, at_percentages, gc_percentages = calculate_compositions_for_sample(
        encoded_genome_path, reference_fasta, composition_sample_size
    )

    # Step 2: Sample chunks by AT percentiles (for AT pathway)
    logger.info("\n=== Sampling AT Pathway ===")
    at_sampled = sample_chunks_by_composition(chunk_indices, at_percentages, "AT", chunks_per_bin)

    # Step 3: Sample chunks by GC percentiles (for GC pathway)
    logger.info("\n=== Sampling GC Pathway ===")
    gc_sampled = sample_chunks_by_composition(chunk_indices, gc_percentages, "GC", chunks_per_bin)

    # Step 4: Extract sequences and analyze both pathways
    logger.info("\n=== Analyzing AT Pathway ===")
    at_results = analyze_pathway_chunks(at_sampled, encoded_genome_path, reference_fasta, "AT")

    logger.info("\n=== Analyzing GC Pathway ===")
    gc_results = analyze_pathway_chunks(gc_sampled, encoded_genome_path, reference_fasta, "GC")

    # Combine results
    results = {
        'AT_pathway': at_results,
        'GC_pathway': gc_results,
    }

    with h5py.File(encoded_genome_path, 'r') as h5f:
        chunk_keys = h5f['chunk_keys'][:]
        all_banks = h5f['all_bank_vectors']

        for bin_name, bin_chunk_indices in sampled_by_bin.items():
            logger.info(f"\nProcessing {bin_name}...")
            results[bin_name] = []

            for chunk_idx in bin_chunk_indices:
                # Get genomic position
                chunk_key = chunk_keys[chunk_idx].decode('utf-8')
                chrom, coords = chunk_key.split(':')
                start, end = map(int, coords.split('-'))

                # Extract sequence
                sequence = extract_sequence(reference_fasta, chrom, start, end)
                if len(sequence) == 0:
                    continue

                # Get encoded data (ternary: -1, 0, +1)
                chunk_data = all_banks[chunk_idx, :, :]

                # Split into positive and negative banks
                bank1_ternary = chunk_data[0, :]
                bank2_ternary = chunk_data[1, :]
                bank3_ternary = chunk_data[2, :]

                # Extract positive and negative components
                bank1_pos = np.maximum(bank1_ternary, 0)  # A contributions
                bank1_neg = np.abs(np.minimum(bank1_ternary, 0))  # T contributions
                bank2_pos = np.maximum(bank2_ternary, 0)  # G contributions
                bank2_neg = np.abs(np.minimum(bank2_ternary, 0))  # C contributions
                bank3_pos = np.maximum(bank3_ternary, 0)  # Y-R dinuc
                bank3_neg = np.abs(np.minimum(bank3_ternary, 0))  # R-Y dinuc

                # Compute overall density (ternary)
                total_zeros = (np.sum(bank1_ternary == 0) +
                               np.sum(bank2_ternary == 0) +
                               np.sum(bank3_ternary == 0))
                density = 1 - (total_zeros / (3 * 5120))

                # Compute magnitudes for SPLIT binary banks
                bank_mags = {
                    'bank1_pos': float(np.linalg.norm(bank1_pos)),  # A
                    'bank1_neg': float(np.linalg.norm(bank1_neg)),  # T
                    'bank2_pos': float(np.linalg.norm(bank2_pos)),  # G
                    'bank2_neg': float(np.linalg.norm(bank2_neg)),  # C
                    'bank3_pos': float(np.linalg.norm(bank3_pos)),  # Y-R
                    'bank3_neg': float(np.linalg.norm(bank3_neg)),  # R-Y
                }

                # Calculate composition
                seq_len = len(sequence)
                a_count = sequence.count('A')
                t_count = sequence.count('T')
                g_count = sequence.count('G')
                c_count = sequence.count('C')

                composition = {
                    'A_percent': (a_count / seq_len) * 100,
                    'T_percent': (t_count / seq_len) * 100,
                    'G_percent': (g_count / seq_len) * 100,
                    'C_percent': (c_count / seq_len) * 100,
                    'GC_content': ((g_count + c_count) / seq_len) * 100,
                    'AT_content': ((a_count + t_count) / seq_len) * 100,
                    'YR_dinuc_percent': count_yr_dinucleotides(sequence),
                }

                results[bin_name].append({
                    'chunk_idx': int(chunk_idx),
                    'position': f"{chrom}:{start}-{end}",
                    'density': float(density),
                    'composition': composition,
                    'bank_magnitudes': {k: float(v) for k, v in bank_mags.items()},
                })

            logger.info(f"{bin_name}: analyzed {len(results[bin_name])} chunks")

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("FULL DENSITY SPECTRUM SAMPLING SUMMARY")
    print("="*80)

    for bin_name in sorted(results.keys()):
        bin_data = results[bin_name]
        if len(bin_data) == 0:
            continue

        densities = [r['density'] for r in bin_data]
        bank1_signals = [r['bank_magnitudes']['bank1'] for r in bin_data]
        bank2_signals = [r['bank_magnitudes']['bank2'] for r in bin_data]
        bank3_signals = [r['bank_magnitudes']['bank3'] for r in bin_data]
        gc_contents = [r['composition']['GC_content'] for r in bin_data]
        at_contents = [r['composition']['AT_content'] for r in bin_data]
        yr_dinucs = [r['composition']['YR_dinuc_percent'] for r in bin_data]

        print(f"\n{bin_name}:")
        print(f"  Chunks: {len(bin_data)}")
        print(f"  Density range: {min(densities):.4f} - {max(densities):.4f}")
        print(f"  Bank1 range: {min(bank1_signals):.2f} - {max(bank1_signals):.2f}")
        print(f"  Bank2 range: {min(bank2_signals):.2f} - {max(bank2_signals):.2f}")
        print(f"  Bank3 range: {min(bank3_signals):.2f} - {max(bank3_signals):.2f}")
        print(f"  GC% range: {min(gc_contents):.1f}% - {max(gc_contents):.1f}%")
        print(f"  AT% range: {min(at_contents):.1f}% - {max(at_contents):.1f}%")
        print(f"  YR dinuc% range: {min(yr_dinucs):.1f}% - {max(yr_dinucs):.1f}%")

    print("="*80)

    return results


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)

    # Paths
    encoded_genome_path = "genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_3banks.h5"
    reference_fasta = "benchmark_results/enhanced_privacy_k13_phase123_optimized/layer1_consensus/consensus.fa"
    output_file = "genomevault/hdv_validation/hdc_experimentation/output/full_spectrum_analysis.json"

    # Run analysis
    results = analyze_full_spectrum(
        encoded_genome_path,
        reference_fasta,
        output_file,
        density_sample_size=100000,
        chunks_per_bin=100
    )
