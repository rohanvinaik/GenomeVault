#!/usr/bin/env python3
"""
Shared utilities for HDV validation testing.
"""

import json
import gzip
import numpy as np
import pysam
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# BIOPHYSICAL LENS DEFINITIONS
# =============================================================================

LENS_DEFINITIONS = {
    'AT': {'positive': ('A',), 'negative': ('T',)},
    'GC': {'positive': ('G',), 'negative': ('C',)},
    'PuPy': {'positive': ('A', 'G'), 'negative': ('T', 'C')},
    'AmKe': {'positive': ('A', 'C'), 'negative': ('G', 'T')},
    'StWk': {'positive': ('G', 'C'), 'negative': ('A', 'T')},
}

LENS_THRESHOLDS = {
    'AT': {'signal': 0.1, 'neutral': 0.6028},
    'GC': {'signal': 0.1, 'neutral': 0.4885},
    'PuPy': {'signal': 0.1, 'neutral': 0.3},
    'AmKe': {'signal': 0.1, 'neutral': 0.3},
    'StWk': {'signal': 0.1, 'neutral': 0.3},
}

# Empirically-determined optimal voting thresholds (per quantization type)
# Determined via tune_lens_thresholds.py on 1000 random positions (seed=42)
# Date: November 19, 2025
# Test set: 954 positions with valid ground truth
OPTIMAL_VOTING_THRESHOLDS = {
    'float32': {
        'AT': 0.05,
        'GC': 0.00,   # Threshold-free! Most reliable lens across ALL quantizations
        'PuPy': 0.20,
        'AmKe': 0.20,
        'StWk': 0.20,
        # Accuracy: 99.48% (vs 98.11% threshold-free, +1.37%)
    },
    'int8': {
        'AT': 0.05,
        'GC': 0.00,   # Threshold-free
        'PuPy': 0.10,
        'AmKe': 0.10,
        'StWk': 0.15,
        # Accuracy: 99.69% (BEST!) (vs 98.11% threshold-free, +1.58%)
    },
    'int4': {
        'AT': 0.0028,
        'GC': 0.00,   # Threshold-free
        'PuPy': 0.0083,
        'AmKe': 0.0055,
        'StWk': 0.0083,
        # Accuracy: 99.37% (vs 30.40% uniform 0.1, +68.97%!)
    },
    'binary': {
        'AT': 0.0025,
        'GC': 0.00,   # Threshold-free
        'PuPy': 0.0020,
        'AmKe': 0.0012,
        'StWk': 0.0020,
        # Accuracy: 96.65% (vs 30.40% uniform 0.1, +66.25%!)
    },
}

NUCLEOTIDE_SIGNATURES = {
    'A': {'AT': +1, 'GC': 0, 'PuPy': +1, 'AmKe': +1, 'StWk': -1},
    'T': {'AT': -1, 'GC': 0, 'PuPy': -1, 'AmKe': -1, 'StWk': -1},
    'G': {'AT': 0, 'GC': +1, 'PuPy': +1, 'AmKe': -1, 'StWk': +1},
    'C': {'AT': 0, 'GC': -1, 'PuPy': -1, 'AmKe': +1, 'StWk': +1},
}


def load_validated_n_positions(path: Path) -> List[Dict]:
    """Load validated N positions from JSON."""
    if not path.exists():
        logger.warning(f"Validated N positions file not found: {path}")
        return []
    
    with open(path, 'r') as f:
        data = json.load(f)
        return data.get('positions', [])


def load_gdiff(path: Path) -> Tuple[Dict, Dict]:
    """
    Load GDiff file and return variants and variant index.
    
    Returns:
        (gdiff_data, variant_index)
    """
    with gzip.open(path, 'rt') as f:
        gdiff = json.load(f)
    
    variants = gdiff["differential_variants"]
    variant_index = {}
    for v in variants:
        key = f"{v['chrom']}:{v['pos']}"
        variant_index[key] = v
    
    return gdiff, variant_index


def sample_test_positions(
    chunk_keys: List[str],
    validated_n_positions: List[Dict],
    sample_size: int,
    n_sample_ratio: float = 0.10,
    seed: int = 42,
    genome_wide: bool = True
) -> Tuple[List[Tuple[str, int]], Set[Tuple[str, int]]]:
    """
    Sample test positions with controlled ratio of validated N positions.

    IMPORTANT: Uses genome-wide sampling by default to avoid local sequence context bias.
    Previous version sampled from whatever chunks were in chunk_keys, which could be
    a single linear region (e.g., chr22:10,000,000-10,010,000). This led to statistically
    invalid results where correction signatures were overfitted to one genomic region.

    Args:
        chunk_keys: List of chunk keys from HDF5
        validated_n_positions: List of validated N position dicts
        sample_size: Total number of positions to sample
        n_sample_ratio: Ratio of positions from validated N list (default 0.10)
        seed: Random seed for reproducibility
        genome_wide: If True (default), stratify sampling across all chromosomes

    Returns:
        (test_positions, high_n_positions_set)
    """
    np.random.seed(seed)

    test_positions = []
    high_n_positions = []

    # Sample from validated N positions
    if validated_n_positions:
        n_sample_size = int(sample_size * n_sample_ratio)
        selected_indices = np.random.choice(
            len(validated_n_positions),
            size=min(n_sample_size, len(validated_n_positions)),
            replace=False
        )

        for idx in selected_indices:
            n_pos = validated_n_positions[idx]
            chrom = n_pos['chrom']

            if not chrom.endswith('_consensus'):
                chrom = chrom + '_consensus'

            pos = n_pos['pos']
            test_positions.append((chrom, pos))
            high_n_positions.append((chrom, pos))

    # Sample from general genome
    N = 2000  # Chunk size
    general_sample_size = sample_size - len(high_n_positions)

    if genome_wide:
        # GENOME-WIDE SAMPLING: Stratify across all chromosomes
        from collections import defaultdict

        # Group chunks by chromosome
        chunks_by_chrom = defaultdict(list)
        for i, chunk_key in enumerate(chunk_keys):
            chrom_part = chunk_key.split(':')[0]
            chrom = chrom_part.replace('_consensus', '')
            chunks_by_chrom[chrom].append(i)

        # Calculate samples per chromosome (proportional to chunk count)
        total_chunks = len(chunk_keys)
        chromosomes = sorted(chunks_by_chrom.keys(), key=lambda x: (
            x != 'chrX' and x != 'chrY',
            int(x.replace('chr', '').replace('X', '23').replace('Y', '24'))
        ))

        samples_per_chrom = {}
        for chrom in chromosomes:
            n_chunks = len(chunks_by_chrom[chrom])
            proportion = n_chunks / total_chunks
            samples_per_chrom[chrom] = int(general_sample_size * proportion)

        # Adjust for rounding errors
        total_allocated = sum(samples_per_chrom.values())
        if total_allocated < general_sample_size:
            diff = general_sample_size - total_allocated
            sorted_chroms = sorted(chromosomes, key=lambda c: len(chunks_by_chrom[c]), reverse=True)
            for i in range(diff):
                samples_per_chrom[sorted_chroms[i % len(sorted_chroms)]] += 1

        # Sample from each chromosome
        for chrom in chromosomes:
            chrom_chunk_indices = chunks_by_chrom[chrom]
            n_samples = samples_per_chrom[chrom]

            if n_samples == 0:
                continue

            # Sample chunks from this chromosome
            sampled_chunk_indices = np.random.choice(
                chrom_chunk_indices,
                size=min(n_samples, len(chrom_chunk_indices)),
                replace=False
            )

            for chunk_idx in sampled_chunk_indices:
                chunk_key = chunk_keys[chunk_idx]
                chrom_key, chunk_start_str = chunk_key.split(':')
                chunk_start = int(chunk_start_str)
                pos = chunk_start + np.random.randint(0, N)
                test_positions.append((chrom_key, pos))

    else:
        # LEGACY SAMPLING: Random chunks (may be biased if chunk_keys is limited)
        random_chunk_indices = np.random.randint(0, len(chunk_keys), size=general_sample_size)

        for chunk_idx in random_chunk_indices:
            chunk_key = chunk_keys[chunk_idx]
            chrom, chunk_start_str = chunk_key.split(':')
            chunk_start = int(chunk_start_str)
            pos = chunk_start + np.random.randint(0, N)
            test_positions.append((chrom, pos))

    return test_positions, set(high_n_positions)


def get_ground_truth(
    chrom: str,
    pos: int,
    variant_index: Dict,
    exp_bam: pysam.AlignmentFile,
    gdiff_region_map: Dict
) -> Tuple[str, int, bool]:
    """
    Get ground truth nucleotide for a position.
    
    Returns:
        (ground_truth, guide_idx, has_n)
        - ground_truth: The nucleotide ('A', 'T', 'G', 'C', 'N', or 'UNVALIDATED')
        - guide_idx: Index of the guide strand
        - has_n: Whether this position has N in experimental data
    """
    pos_key = f"{chrom}:{pos}"
    is_variant = pos_key in variant_index
    
    if is_variant:
        v = variant_index[pos_key]
        return v["alt"], v.get("guide_idx", 0), False
    
    # Non-variant position - check experimental BAM
    try:
        pileup = exp_bam.pileup(chrom, pos, pos + 1, truncate=True, min_base_quality=20)
        bases = []
        for pileupcolumn in pileup:
            if pileupcolumn.pos == pos:
                for pileupread in pileupcolumn.pileups:
                    if not pileupread.is_del and not pileupread.is_refskip:
                        base = pileupread.alignment.query_sequence[pileupread.query_position]
                        bases.append(base.upper())
        
        if not bases:
            # No experimental coverage - biophysical recovery needed
            return 'N', 0, True
        
        # Use consensus base
        base_counts = Counter(bases)
        ground_truth = base_counts.most_common(1)[0][0]
        
        # Get guide index from region map
        guide_idx = 0
        for region_key, gidx in gdiff_region_map.items():
            region_chrom, region_range = region_key.split(':')
            if region_chrom == chrom:
                start, end = map(int, region_range.split('-'))
                if start <= pos < end:
                    guide_idx = gidx
                    break
        
        return ground_truth, guide_idx, False
    
    except Exception as e:
        return None, 0, False


def predict_naive_hdc(lens_results: Dict[str, float]) -> Tuple[str, float]:
    """Naive HDC baseline: Use only AT and GC lenses."""
    at_sim = lens_results.get('AT', 0.0)
    gc_sim = lens_results.get('GC', 0.0)

    votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}

    if at_sim > 0.1:
        votes['A'] += 1
    elif at_sim < -0.1:
        votes['T'] += 1

    if gc_sim > 0.1:
        votes['G'] += 1
    elif gc_sim < -0.1:
        votes['C'] += 1

    if sum(votes.values()) == 0:
        return 'A', 0.0

    best_nuc = max(votes, key=votes.get)
    confidence = votes[best_nuc] / 2.0
    return best_nuc, confidence


def compute_signal_strength(lens_results: Dict[str, float]) -> float:
    """
    Compute total signal strength across all lenses.

    This is the sum of absolute values of all lens similarities,
    representing how much biophysical signal is present.

    Args:
        lens_results: Dict of lens similarities

    Returns:
        Total signal strength (sum of absolute values)
    """
    return sum(abs(lens_results.get(lens, 0.0)) for lens in ['AT', 'GC', 'PuPy', 'AmKe', 'StWk'])


def predict_multi_lens_voting(
    lens_results: Dict[str, float],
    threshold: float = None,
    quantization: str = None
) -> Tuple[str, float, Dict[str, int]]:
    """
    Multi-lens voting: Use all 5 lenses with optimal per-lens thresholds.

    Args:
        lens_results: Dict of lens similarities
        threshold: Optional uniform threshold (overrides per-lens thresholds)
        quantization: Quantization type ('float32', 'int8', 'int4', 'binary')
                     If provided, uses empirically-determined optimal thresholds

    Returns:
        (predicted_nucleotide, confidence, vote_counts)
    """
    votes = {nuc: 0 for nuc in 'ATGC'}

    # Determine thresholds to use
    if threshold is not None:
        # Uniform threshold provided (backward compatibility)
        per_lens_thresholds = {ln: threshold for ln in ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']}
    elif quantization and quantization in OPTIMAL_VOTING_THRESHOLDS:
        # Use optimal per-lens thresholds for this quantization
        per_lens_thresholds = OPTIMAL_VOTING_THRESHOLDS[quantization]
    else:
        # Default: threshold-free (all zeros)
        per_lens_thresholds = {ln: 0.0 for ln in ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']}

    for nuc, signature in NUCLEOTIDE_SIGNATURES.items():
        score = 0
        for lens_name, expected_sign in signature.items():
            observed_similarity = lens_results.get(lens_name, 0.0)
            lens_threshold = per_lens_thresholds.get(lens_name, 0.0)

            if expected_sign == 0:
                continue
            elif expected_sign > 0 and observed_similarity > lens_threshold:
                score += 1
            elif expected_sign < 0 and observed_similarity < -lens_threshold:
                score += 1
        votes[nuc] = score

    best_nuc = max(votes, key=votes.get)
    confidence = votes[best_nuc] / 5.0

    return best_nuc, confidence, votes


def check_lens_property(lens_results: Dict[str, float], ground_truth: str) -> Dict[str, bool]:
    """Check if each lens correctly detects its biophysical property."""
    results = {}
    for lens_name, lens_def in LENS_DEFINITIONS.items():
        similarity = lens_results.get(lens_name, 0.0)
        thresholds = LENS_THRESHOLDS[lens_name]
        
        if ground_truth in lens_def['positive']:
            expected_sign = +1
        elif ground_truth in lens_def['negative']:
            expected_sign = -1
        else:
            expected_sign = 0
        
        if expected_sign == 0:
            correct = abs(similarity) < thresholds['neutral']
        elif expected_sign > 0:
            correct = similarity > thresholds['signal']
        else:
            correct = similarity < -thresholds['signal']
        
        results[lens_name] = correct
    
    return results


def save_results(results: Dict, output_path: Path):
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"✓ Results saved to: {output_path}")


def compute_confusion_matrix(predictions: List[str], ground_truths: List[str]) -> Dict:
    """
    Compute confusion matrix for nucleotide predictions.
    
    Returns dict with:
        - matrix: 4x4 confusion matrix (ATGC order)
        - per_class_stats: precision, recall, f1 for each nucleotide
    """
    nucleotides = ['A', 'T', 'G', 'C']
    nuc_to_idx = {nuc: i for i, nuc in enumerate(nucleotides)}
    
    # Initialize confusion matrix
    matrix = np.zeros((4, 4), dtype=int)
    
    for pred, truth in zip(predictions, ground_truths):
        if pred in nuc_to_idx and truth in nuc_to_idx:
            matrix[nuc_to_idx[truth], nuc_to_idx[pred]] += 1
    
    # Compute per-class statistics
    per_class_stats = {}
    for i, nuc in enumerate(nucleotides):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class_stats[nuc] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': int(matrix[i, :].sum())
        }
    
    return {
        'matrix': matrix.tolist(),
        'nucleotides': nucleotides,
        'per_class_stats': per_class_stats
    }
