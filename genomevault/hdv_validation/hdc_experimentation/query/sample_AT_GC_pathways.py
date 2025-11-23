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


def run_analysis(
    encoded_genome_path: str,
    reference_fasta: str,
    output_file: str,
    sample_size: int = 10000,
    chunks_per_bin: int = 100
):
    """
    Sample chunks by AT% and GC% percentiles, analyze split binary signals.

    Args:
        encoded_genome_path: Path to H5 file
        reference_fasta: Path to reference genome
        output_file: Where to save JSON results
        sample_size: Number of chunks to sample for percentile calculation
        chunks_per_bin: Number of chunks per percentile bin
    """
    np.random.seed(42)

    logger.info(f"Sampling {sample_size} chunks to calculate composition percentiles...")

    # Step 1: Sample chunks and calculate compositions
    with h5py.File(encoded_genome_path, 'r') as f:
        total_chunks = f['all_bank_vectors'].shape[0]
        chunk_keys = f['chunk_keys'][:]

        # Random sample
        sample_indices = np.random.choice(total_chunks, size=min(sample_size, total_chunks), replace=False)

        at_pcts = []
        gc_pcts = []
        valid_indices = []

        for idx in sample_indices:
            chunk_key = chunk_keys[idx].decode('utf-8')
            chrom, coords = chunk_key.split(':')
            start, end = map(int, coords.split('-'))

            sequence = extract_sequence(reference_fasta, chrom, start, end)
            if len(sequence) == 0:
                continue

            seq_len = len(sequence)
            at_count = sequence.count('A') + sequence.count('T')
            gc_count = sequence.count('G') + sequence.count('C')

            at_pct = (at_count / seq_len) * 100
            gc_pct = (gc_count / seq_len) * 100

            at_pcts.append(at_pct)
            gc_pcts.append(gc_pct)
            valid_indices.append(idx)

        at_pcts = np.array(at_pcts)
        gc_pcts = np.array(gc_pcts)
        valid_indices = np.array(valid_indices)

        logger.info(f"Valid chunks: {len(valid_indices)}")
        logger.info(f"AT% range: {at_pcts.min():.1f}% - {at_pcts.max():.1f}%")
        logger.info(f"GC% range: {gc_pcts.min():.1f}% - {gc_pcts.max():.1f}%")

    # Step 2: Sample by AT percentiles
    at_percentiles = [0, 25, 50, 75, 100]
    at_thresholds = np.percentile(at_pcts, at_percentiles)

    logger.info("\n=== AT Pathway Sampling ===")
    at_samples = {}
    for i in range(len(at_percentiles) - 1):
        bin_name = f"AT_p{at_percentiles[i]}-{at_percentiles[i+1]}"
        low, high = at_thresholds[i], at_thresholds[i+1]

        mask = (at_pcts >= low) & (at_pcts <= high)
        bin_indices = valid_indices[mask]

        if len(bin_indices) > chunks_per_bin:
            sampled = np.random.choice(bin_indices, size=chunks_per_bin, replace=False)
        else:
            sampled = bin_indices

        at_samples[bin_name] = sampled
        logger.info(f"{bin_name}: {len(sampled)} chunks (AT {low:.1f}%-{high:.1f}%)")

    # Step 3: Sample by GC percentiles
    gc_percentiles = [0, 25, 50, 75, 100]
    gc_thresholds = np.percentile(gc_pcts, gc_percentiles)

    logger.info("\n=== GC Pathway Sampling ===")
    gc_samples = {}
    for i in range(len(gc_percentiles) - 1):
        bin_name = f"GC_p{gc_percentiles[i]}-{gc_percentiles[i+1]}"
        low, high = gc_thresholds[i], gc_thresholds[i+1]

        mask = (gc_pcts >= low) & (gc_pcts <= high)
        bin_indices = valid_indices[mask]

        if len(bin_indices) > chunks_per_bin:
            sampled = np.random.choice(bin_indices, size=chunks_per_bin, replace=False)
        else:
            sampled = bin_indices

        gc_samples[bin_name] = sampled
        logger.info(f"{bin_name}: {len(sampled)} chunks (GC {low:.1f}%-{high:.1f}%)")

    # Step 4: Extract split binary signals
    logger.info("\n=== Extracting Split Binary Signals ===")

    results = {
        'AT_pathway': {},
        'GC_pathway': {},
    }

    with h5py.File(encoded_genome_path, 'r') as f:
        chunk_keys = f['chunk_keys'][:]
        all_banks = f['all_bank_vectors']

        # Analyze AT pathway
        for bin_name, bin_indices in at_samples.items():
            bin_results = []

            for chunk_idx in bin_indices:
                chunk_key = chunk_keys[chunk_idx].decode('utf-8')
                chrom, coords = chunk_key.split(':')
                start, end = map(int, coords.split('-'))

                sequence = extract_sequence(reference_fasta, chrom, start, end)
                if len(sequence) == 0:
                    continue

                # Get ternary banks
                chunk_data = all_banks[chunk_idx, :, :]
                bank1_ternary = chunk_data[0, :]
                bank2_ternary = chunk_data[1, :]
                bank3_ternary = chunk_data[2, :]

                # Split into pos/neg
                bank1_pos = np.maximum(bank1_ternary, 0)  # A
                bank1_neg = np.abs(np.minimum(bank1_ternary, 0))  # T
                bank2_pos = np.maximum(bank2_ternary, 0)  # G
                bank2_neg = np.abs(np.minimum(bank2_ternary, 0))  # C
                bank3_pos = np.maximum(bank3_ternary, 0)
                bank3_neg = np.abs(np.minimum(bank3_ternary, 0))

                # Calculate composition
                seq_len = len(sequence)
                composition = {
                    'A_pct': (sequence.count('A') / seq_len) * 100,
                    'T_pct': (sequence.count('T') / seq_len) * 100,
                    'G_pct': (sequence.count('G') / seq_len) * 100,
                    'C_pct': (sequence.count('C') / seq_len) * 100,
                    'AT_pct': ((sequence.count('A') + sequence.count('T')) / seq_len) * 100,
                    'GC_pct': ((sequence.count('G') + sequence.count('C')) / seq_len) * 100,
                }

                # Calculate magnitudes
                signals = {
                    'bank1_pos_mag': float(np.linalg.norm(bank1_pos)),  # A signal
                    'bank1_neg_mag': float(np.linalg.norm(bank1_neg)),  # T signal
                    'bank2_pos_mag': float(np.linalg.norm(bank2_pos)),  # G signal
                    'bank2_neg_mag': float(np.linalg.norm(bank2_neg)),  # C signal
                    'bank3_pos_mag': float(np.linalg.norm(bank3_pos)),
                    'bank3_neg_mag': float(np.linalg.norm(bank3_neg)),
                }

                bin_results.append({
                    'chunk_idx': int(chunk_idx),
                    'position': f"{chrom}:{start}-{end}",
                    'composition': composition,
                    'signals': signals,
                })

            results['AT_pathway'][bin_name] = bin_results
            logger.info(f"{bin_name}: analyzed {len(bin_results)} chunks")

        # Analyze GC pathway
        for bin_name, bin_indices in gc_samples.items():
            bin_results = []

            for chunk_idx in bin_indices:
                chunk_key = chunk_keys[chunk_idx].decode('utf-8')
                chrom, coords = chunk_key.split(':')
                start, end = map(int, coords.split('-'))

                sequence = extract_sequence(reference_fasta, chrom, start, end)
                if len(sequence) == 0:
                    continue

                # Get ternary banks
                chunk_data = all_banks[chunk_idx, :, :]
                bank1_ternary = chunk_data[0, :]
                bank2_ternary = chunk_data[1, :]
                bank3_ternary = chunk_data[2, :]

                # Split into pos/neg
                bank1_pos = np.maximum(bank1_ternary, 0)  # A
                bank1_neg = np.abs(np.minimum(bank1_ternary, 0))  # T
                bank2_pos = np.maximum(bank2_ternary, 0)  # G
                bank2_neg = np.abs(np.minimum(bank2_ternary, 0))  # C
                bank3_pos = np.maximum(bank3_ternary, 0)
                bank3_neg = np.abs(np.minimum(bank3_ternary, 0))

                # Calculate composition
                seq_len = len(sequence)
                composition = {
                    'A_pct': (sequence.count('A') / seq_len) * 100,
                    'T_pct': (sequence.count('T') / seq_len) * 100,
                    'G_pct': (sequence.count('G') / seq_len) * 100,
                    'C_pct': (sequence.count('C') / seq_len) * 100,
                    'AT_pct': ((sequence.count('A') + sequence.count('T')) / seq_len) * 100,
                    'GC_pct': ((sequence.count('G') + sequence.count('C')) / seq_len) * 100,
                }

                # Calculate magnitudes
                signals = {
                    'bank1_pos_mag': float(np.linalg.norm(bank1_pos)),  # A signal
                    'bank1_neg_mag': float(np.linalg.norm(bank1_neg)),  # T signal
                    'bank2_pos_mag': float(np.linalg.norm(bank2_pos)),  # G signal
                    'bank2_neg_mag': float(np.linalg.norm(bank2_neg)),  # C signal
                    'bank3_pos_mag': float(np.linalg.norm(bank3_pos)),
                    'bank3_neg_mag': float(np.linalg.norm(bank3_neg)),
                }

                bin_results.append({
                    'chunk_idx': int(chunk_idx),
                    'position': f"{chrom}:{start}-{end}",
                    'composition': composition,
                    'signals': signals,
                })

            results['GC_pathway'][bin_name] = bin_results
            logger.info(f"{bin_name}: analyzed {len(bin_results)} chunks")

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as outf:
        json.dump(results, outf, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("AT AND GC PATHWAY SAMPLING COMPLETE")
    print("="*80)

    for pathway_name in ['AT_pathway', 'GC_pathway']:
        print(f"\n{pathway_name.upper()}:")

        for bin_name in sorted(results[pathway_name].keys()):
            bin_data = results[pathway_name][bin_name]
            if len(bin_data) == 0:
                continue

            # Extract signal ranges
            bank1_pos = [d['signals']['bank1_pos_mag'] for d in bin_data]
            bank1_neg = [d['signals']['bank1_neg_mag'] for d in bin_data]
            bank2_pos = [d['signals']['bank2_pos_mag'] for d in bin_data]
            bank2_neg = [d['signals']['bank2_neg_mag'] for d in bin_data]

            # Extract composition ranges
            a_pcts = [d['composition']['A_pct'] for d in bin_data]
            t_pcts = [d['composition']['T_pct'] for d in bin_data]
            g_pcts = [d['composition']['G_pct'] for d in bin_data]
            c_pcts = [d['composition']['C_pct'] for d in bin_data]
            at_pcts = [d['composition']['AT_pct'] for d in bin_data]
            gc_pcts = [d['composition']['GC_pct'] for d in bin_data]

            print(f"\n  {bin_name}:")
            print(f"    Chunks: {len(bin_data)}")
            print(f"    Composition ranges:")
            print(f"      A%:  {min(a_pcts):.1f}% - {max(a_pcts):.1f}%")
            print(f"      T%:  {min(t_pcts):.1f}% - {max(t_pcts):.1f}%")
            print(f"      G%:  {min(g_pcts):.1f}% - {max(g_pcts):.1f}%")
            print(f"      C%:  {min(c_pcts):.1f}% - {max(c_pcts):.1f}%")
            print(f"      AT%: {min(at_pcts):.1f}% - {max(at_pcts):.1f}%")
            print(f"      GC%: {min(gc_pcts):.1f}% - {max(gc_pcts):.1f}%")
            print(f"    Signal ranges:")
            print(f"      Bank1_pos (A): {min(bank1_pos):.2f} - {max(bank1_pos):.2f}")
            print(f"      Bank1_neg (T): {min(bank1_neg):.2f} - {max(bank1_neg):.2f}")
            print(f"      Bank2_pos (G): {min(bank2_pos):.2f} - {max(bank2_pos):.2f}")
            print(f"      Bank2_neg (C): {min(bank2_neg):.2f} - {max(bank2_neg):.2f}")

    print("="*80)

    return results


if __name__ == '__main__':
    # Paths
    encoded_genome_path = "genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_3banks.h5"
    reference_fasta = "benchmark_results/enhanced_privacy_k13_phase123_optimized/layer1_consensus/consensus.fa"
    output_file = "genomevault/hdv_validation/hdc_experimentation/output/AT_GC_pathway_analysis.json"

    # Run analysis
    results = run_analysis(
        encoded_genome_path,
        reference_fasta,
        output_file,
        sample_size=10000,
        chunks_per_bin=100
    )
