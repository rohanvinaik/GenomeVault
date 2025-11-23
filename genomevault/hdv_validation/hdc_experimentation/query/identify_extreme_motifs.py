"""
Identify Structural Motifs from Extreme Chunks

Takes the extreme chunks found by find_extreme_motifs.py and:
1. Maps them back to genome positions
2. Extracts actual DNA sequences
3. Identifies what motifs they contain (ALU, TATA, poly-A, etc.)
4. Measures density vs signal strength correlation
5. Builds regression models for intelligent threshold setting

Author: Phase 1 Week 3 - Barbie Method Ground Truth
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


# Known motif consensus sequences
MOTIF_PATTERNS = {
    'TATA_BOX': 'TATAAA',
    'CAAT_BOX': 'GGCCAATCT',
    'GC_BOX': 'GGGCGG',
    'ALU_CONSENSUS': 'GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGG',
    'POLY_A': 'AAAAAAAAAA',
    'POLY_T': 'TTTTTTTTTT',
    'CG_ISLAND': 'CGCGCGCGCG',
}


def load_chunk_keys(encoded_genome_path: str) -> np.ndarray:
    """Load chunk keys from H5 file."""
    import h5py
    with h5py.File(encoded_genome_path, 'r') as f:
        return f['chunk_keys'][:]


def chunk_idx_to_genomic_position(chunk_idx: int, chunk_keys: np.ndarray) -> Tuple[str, int, int]:
    """
    Convert chunk index to genomic position using chunk_keys from H5 file.

    Args:
        chunk_idx: Index of the chunk
        chunk_keys: Array of chunk keys from H5 file (format: b'chr1:0-1024')

    Returns:
        (chromosome, start, end)
    """
    # Parse chunk key (format: b'chr1:0-1024')
    chunk_key = chunk_keys[chunk_idx].decode('utf-8')
    chrom, coords = chunk_key.split(':')
    start, end = map(int, coords.split('-'))
    return chrom, start, end


def extract_sequence(fasta_path: str, chrom: str, start: int, end: int) -> str:
    """Extract DNA sequence from reference FASTA."""
    try:
        fasta = pysam.FastaFile(fasta_path)

        # Handle consensus naming convention (chr1_consensus vs chr1)
        if chrom in fasta.references:
            fetch_chrom = chrom
        elif f"{chrom}_consensus" in fasta.references:
            fetch_chrom = f"{chrom}_consensus"
        else:
            fasta.close()
            return ""  # Chromosome not found

        # Check if coordinates are valid
        chrom_length = fasta.get_reference_length(fetch_chrom)
        if start >= chrom_length or end > chrom_length:
            fasta.close()
            return ""  # Return empty string for out-of-bounds
        sequence = fasta.fetch(fetch_chrom, start, end)
        fasta.close()
        return sequence.upper()
    except:
        return ""  # Return empty string on any error


def identify_motif(sequence: str) -> List[Tuple[str, int, float]]:
    """
    Identify known motifs in sequence.

    Returns:
        List of (motif_name, position, match_score)
    """
    motifs_found = []

    for motif_name, pattern in MOTIF_PATTERNS.items():
        # Simple substring matching (could use fuzzy matching later)
        pos = sequence.find(pattern)
        if pos != -1:
            match_score = 1.0
            motifs_found.append((motif_name, pos, match_score))

        # Also check reverse complement for some motifs
        if motif_name in ['ALU_CONSENSUS', 'CG_ISLAND']:
            rc_pattern = reverse_complement(pattern)
            pos = sequence.find(rc_pattern)
            if pos != -1:
                motifs_found.append((f"{motif_name}_RC", pos, 1.0))

    # Skip empty sequences
    if len(sequence) == 0:
        return motifs_found

    # Check for poly-A/T runs (relaxed matching)
    if sequence.count('A') / len(sequence) > 0.7:
        motifs_found.append(('POLY_A_RICH', 0, sequence.count('A') / len(sequence)))

    if sequence.count('T') / len(sequence) > 0.7:
        motifs_found.append(('POLY_T_RICH', 0, sequence.count('T') / len(sequence)))

    # Check for GC-rich regions
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
    if gc_content > 0.6:
        motifs_found.append(('GC_RICH', 0, gc_content))

    if gc_content < 0.3:
        motifs_found.append(('AT_RICH', 0, 1.0 - gc_content))

    return motifs_found


def reverse_complement(seq: str) -> str:
    """Return reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(complement.get(base, base) for base in reversed(seq))


def analyze_extreme_chunks(
    extreme_chunks_file: str,
    encoded_genome_path: str,
    reference_fasta: str,
    output_file: str
):
    """
    Analyze extreme chunks to identify motifs and measure density vs signal.

    Args:
        extreme_chunks_file: Path to JSON with extreme chunk data
        encoded_genome_path: Path to H5 encoded genome
        reference_fasta: Path to reference genome FASTA
        output_file: Where to save results
    """
    # Load extreme chunks data
    with open(extreme_chunks_file, 'r') as f:
        extreme_data = json.load(f)

    results = {
        'GC_RICH': [],
        'AT_RICH': [],
        'BALANCED': [],
    }

    # Load chunk keys for genomic coordinate mapping
    logger.info("Loading chunk keys from H5 file...")
    chunk_keys = load_chunk_keys(encoded_genome_path)
    logger.info(f"Loaded {len(chunk_keys)} chunk keys")

    # Open encoded genome
    h5f = h5py.File(encoded_genome_path, 'r')
    all_banks = h5f['all_bank_vectors']

    for category in ['GC_RICH', 'AT_RICH', 'BALANCED']:
        logger.info(f"\nAnalyzing {category} chunks...")
        chunk_indices = extreme_data[category]['chunk_indices'][:50]  # Limit to first 50

        for chunk_idx in chunk_indices:
            # Get genomic position using chunk_keys
            chrom, start, end = chunk_idx_to_genomic_position(chunk_idx, chunk_keys)

            # Extract sequence
            try:
                sequence = extract_sequence(reference_fasta, chrom, start, end)
            except Exception as e:
                logger.warning(f"Could not extract sequence for chunk {chunk_idx}: {e}")
                continue

            # Skip empty sequences (out of bounds)
            if len(sequence) == 0:
                logger.warning(f"Empty sequence for chunk {chunk_idx} at {chrom}:{start}-{end}")
                continue

            # Identify motifs
            motifs = identify_motif(sequence)

            # Get encoded data
            chunk_data = all_banks[chunk_idx, :, :]
            banks = {
                'bank1': chunk_data[0, :],
                'bank2': chunk_data[1, :],
                'bank3': chunk_data[2, :],
            }

            # Compute metrics
            total_zeros = sum(np.sum(bank == 0) for bank in banks.values())
            density = 1 - (total_zeros / (3 * 5120))

            bank_mags = {
                'bank1': np.linalg.norm(banks['bank1']),
                'bank2': np.linalg.norm(banks['bank2']),
                'bank3': np.linalg.norm(banks['bank3']),
            }

            # Calculate detailed nucleotide composition
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
            }

            results[category].append({
                'chunk_idx': chunk_idx,
                'position': f"{chrom}:{start}-{end}",
                'sequence': sequence[:100],  # First 100bp for inspection
                'motifs': [{'name': m[0], 'pos': m[1], 'score': m[2]} for m in motifs],
                'density': float(density),
                'composition': composition,
                'bank_magnitudes': {k: float(v) for k, v in bank_mags.items()},
            })

        logger.info(f"Analyzed {len(results[category])} {category} chunks")

    h5f.close()

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("MOTIF IDENTIFICATION SUMMARY")
    print("="*80)

    for category in ['GC_RICH', 'AT_RICH', 'BALANCED']:
        print(f"\n{category}:")

        # Count motif types
        motif_counts = {}
        for result in results[category]:
            for motif in result['motifs']:
                name = motif['name']
                motif_counts[name] = motif_counts.get(name, 0) + 1

        for motif_name, count in sorted(motif_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {motif_name}: {count} instances")

        if len(results[category]) == 0:
            print(f"  (No chunks found in chr22 range)")
            continue

        # Density vs signal correlation
        densities = [r['density'] for r in results[category]]
        bank2_signals = [r['bank_magnitudes']['bank2'] for r in results[category]]
        gc_contents = [r['composition']['GC_content'] for r in results[category]]
        at_contents = [r['composition']['AT_content'] for r in results[category]]

        print(f"\n  Density range: {min(densities):.4f} - {max(densities):.4f}")
        print(f"  Bank2 signal range: {min(bank2_signals):.2f} - {max(bank2_signals):.2f}")
        print(f"  GC content range: {min(gc_contents):.1f}% - {max(gc_contents):.1f}%")
        print(f"  AT content range: {min(at_contents):.1f}% - {max(at_contents):.1f}%")

    print("="*80)

    return results


if __name__ == '__main__':
    # Paths
    extreme_chunks_file = "/tmp/extreme_motifs_n50.json"
    encoded_genome_path = "genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_3banks.h5"
    reference_fasta = "benchmark_results/enhanced_privacy_k13_phase123_optimized/layer1_consensus/consensus.fa"  # Whole-genome consensus with complete coverage
    output_file = "genomevault/hdv_validation/hdc_experimentation/output/identified_motifs.json"

    # Check if files exist
    if not Path(extreme_chunks_file).exists():
        logger.error(f"ERROR: Extreme chunks file not found: {extreme_chunks_file}")
        logger.error("Run find_extreme_motifs.py first!")
        exit(1)

    if not Path(reference_fasta).exists():
        logger.error(f"ERROR: Reference FASTA not found: {reference_fasta}")
        logger.error("Need reference genome to extract sequences!")
        exit(1)

    # Run analysis
    results = analyze_extreme_chunks(
        extreme_chunks_file,
        encoded_genome_path,
        reference_fasta,
        output_file
    )
