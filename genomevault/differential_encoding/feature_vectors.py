"""
Feature Vector Construction for Differential Encoding.

This module implements feature vector construction from variant differences,
converting genomic differences into fixed-dimensional numerical representations
suitable for hyperdimensional encoding or machine learning.

Section 6.1: Feature Vector Construction
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import numpy as np

from genomevault.differential_encoding.differences import (
    VariantDifference,
    DifferenceType,
    FunctionalImpact,
)
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


# Feature vector dimensions (total = 95)
DIM_DIFFERENCE_TYPES = 3      # New, missing, genotype
DIM_POSITION_ENCODING = 64    # Sinusoidal position encoding
DIM_ALLELE_COMPOSITION = 8    # Ref + alt nucleotide frequencies (4+4)
DIM_GENOTYPE_DIST = 5         # 0/0, 0/1, 1/1, 1/2, other
DIM_FUNCTIONAL_IMPACT = 10    # Impact scores across categories
DIM_QUALITY_METRICS = 5       # mean, std, min, max, median

TOTAL_FEATURE_DIM = (
    DIM_DIFFERENCE_TYPES +
    DIM_POSITION_ENCODING +
    DIM_ALLELE_COMPOSITION +
    DIM_GENOTYPE_DIST +
    DIM_FUNCTIONAL_IMPACT +
    DIM_QUALITY_METRICS
)


def sinusoidal_position_encoding(
    position: int,
    dimension: int = DIM_POSITION_ENCODING,
    max_wavelength: float = 10000.0
) -> np.ndarray:
    """
    Generate sinusoidal position encoding for a genomic position.

    Uses the Transformer-style position encoding:
        PE(pos, 2i) = sin(pos / max_wavelength^(2i/dim))
        PE(pos, 2i+1) = cos(pos / max_wavelength^(2i/dim))

    This encoding has several useful properties:
    - Unique encoding for each position
    - Smooth similarity for nearby positions
    - Bounded values in [-1, 1]
    - Learnable positional relationships

    Args:
        position: Genomic position (0-based coordinate)
        dimension: Output dimension (must be even)
        max_wavelength: Maximum wavelength for encoding (default 10000)

    Returns:
        Position encoding vector of shape (dimension,)

    Raises:
        ValueError: If dimension is not even or position is negative

    Example:
        >>> encoding = sinusoidal_position_encoding(100000, dimension=64)
        >>> print(encoding.shape)
        (64,)
        >>> print(np.min(encoding), np.max(encoding))
        -1.0 1.0
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension must be even, got {dimension}")
    if position < 0:
        raise ValueError(f"position must be non-negative, got {position}")

    # Create encoding vector
    encoding = np.zeros(dimension, dtype=np.float32)

    # Compute divisors for each dimension
    # div_term = max_wavelength^(2i/dim) for i in [0, dim/2)
    i = np.arange(0, dimension // 2, dtype=np.float32)
    div_term = np.power(max_wavelength, 2 * i / dimension)

    # Apply sin to even indices
    encoding[0::2] = np.sin(position / div_term)

    # Apply cos to odd indices
    encoding[1::2] = np.cos(position / div_term)

    return encoding


def compute_allele_composition(differences: List[VariantDifference]) -> np.ndarray:
    """
    Compute allele composition features from variant differences.

    Counts nucleotide frequencies in reference and alternate alleles,
    normalized to frequencies.

    Features:
    - Ref allele composition: [A, C, G, T] frequencies (4D)
    - Alt allele composition: [A, C, G, T] frequencies (4D)
    Total: 8D

    Args:
        differences: List of variant differences

    Returns:
        8-dimensional allele composition vector

    Example:
        >>> diffs = [VariantDifference(...)]
        >>> composition = compute_allele_composition(diffs)
        >>> print(composition.shape)
        (8,)
    """
    nucleotides = ['A', 'C', 'G', 'T']
    ref_counts = {nuc: 0 for nuc in nucleotides}
    alt_counts = {nuc: 0 for nuc in nucleotides}

    total_ref = 0
    total_alt = 0

    for diff in differences:
        # Count reference alleles
        if diff.exp_ref:
            for nuc in diff.exp_ref.upper():
                if nuc in nucleotides:
                    ref_counts[nuc] += 1
                    total_ref += 1
        elif diff.ref_ref:
            for nuc in diff.ref_ref.upper():
                if nuc in nucleotides:
                    ref_counts[nuc] += 1
                    total_ref += 1

        # Count alternate alleles
        if diff.exp_alt:
            for nuc in diff.exp_alt.upper():
                if nuc in nucleotides:
                    alt_counts[nuc] += 1
                    total_alt += 1
        elif diff.ref_alt:
            for nuc in diff.ref_alt.upper():
                if nuc in nucleotides:
                    alt_counts[nuc] += 1
                    total_alt += 1

    # Normalize to frequencies
    composition = np.zeros(8, dtype=np.float32)

    # Ref frequencies
    if total_ref > 0:
        for i, nuc in enumerate(nucleotides):
            composition[i] = ref_counts[nuc] / total_ref

    # Alt frequencies
    if total_alt > 0:
        for i, nuc in enumerate(nucleotides):
            composition[4 + i] = alt_counts[nuc] / total_alt

    return composition


def compute_genotype_distribution(differences: List[VariantDifference]) -> np.ndarray:
    """
    Compute genotype distribution features from variant differences.

    Counts frequency of different genotype patterns:
    - 0/0: Homozygous reference
    - 0/1: Heterozygous
    - 1/1: Homozygous alternate
    - 1/2: Compound heterozygous
    - other: Other patterns (including missing)

    Args:
        differences: List of variant differences

    Returns:
        5-dimensional genotype distribution vector (normalized frequencies)

    Example:
        >>> diffs = [VariantDifference(...)]
        >>> dist = compute_genotype_distribution(diffs)
        >>> print(dist.shape)
        (5,)
        >>> print(np.sum(dist))  # Should be close to 1.0
        1.0
    """
    genotype_counts = {
        '0/0': 0,
        '0/1': 0,
        '1/1': 0,
        '1/2': 0,
        'other': 0,
    }

    total = 0

    for diff in differences:
        # Get genotype from experimental or reference
        genotype = diff.exp_genotype or diff.ref_genotype

        if genotype:
            # Normalize genotype format (handle both / and |)
            normalized = genotype.replace('|', '/')

            if normalized in genotype_counts:
                genotype_counts[normalized] += 1
            else:
                genotype_counts['other'] += 1

            total += 1
        else:
            genotype_counts['other'] += 1
            total += 1

    # Normalize to frequencies
    distribution = np.zeros(5, dtype=np.float32)

    if total > 0:
        distribution[0] = genotype_counts['0/0'] / total
        distribution[1] = genotype_counts['0/1'] / total
        distribution[2] = genotype_counts['1/1'] / total
        distribution[3] = genotype_counts['1/2'] / total
        distribution[4] = genotype_counts['other'] / total

    return distribution


def compute_functional_impact_vector(differences: List[VariantDifference]) -> np.ndarray:
    """
    Compute functional impact features from variant differences.

    Creates a 10-dimensional vector capturing functional impact distribution
    and characteristics:
    - Impact level frequencies: HIGH, MODERATE, LOW, MODIFIER, UNKNOWN (5D)
    - Average impact score (1D)
    - Max impact score (1D)
    - Fraction with high/moderate impact (1D)
    - Transition/transversion ratio (1D)
    - Indel fraction (1D)

    Args:
        differences: List of variant differences

    Returns:
        10-dimensional functional impact vector

    Example:
        >>> diffs = [VariantDifference(...)]
        >>> impact = compute_functional_impact_vector(diffs)
        >>> print(impact.shape)
        (10,)
    """
    impact_counts = {
        FunctionalImpact.HIGH: 0,
        FunctionalImpact.MODERATE: 0,
        FunctionalImpact.LOW: 0,
        FunctionalImpact.MODIFIER: 0,
        FunctionalImpact.UNKNOWN: 0,
    }

    # Impact scores (for numeric conversion)
    impact_scores = {
        FunctionalImpact.HIGH: 1.0,
        FunctionalImpact.MODERATE: 0.7,
        FunctionalImpact.LOW: 0.3,
        FunctionalImpact.MODIFIER: 0.1,
        FunctionalImpact.UNKNOWN: 0.0,
    }

    total = len(differences)
    if total == 0:
        return np.zeros(10, dtype=np.float32)

    # Count impacts
    scores = []
    transitions = 0  # A<->G, C<->T
    transversions = 0  # Other SNVs
    indels = 0

    for diff in differences:
        impact_counts[diff.functional_impact] += 1
        scores.append(impact_scores[diff.functional_impact])

        # Determine variant type
        ref = diff.exp_ref or diff.ref_ref or ""
        alt = diff.exp_alt or diff.ref_alt or ""

        if len(ref) == 1 and len(alt) == 1:
            # SNV - check transition vs transversion
            if (ref, alt) in [('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')]:
                transitions += 1
            else:
                transversions += 1
        elif len(ref) != len(alt):
            # Indel
            indels += 1

    # Build feature vector
    features = np.zeros(10, dtype=np.float32)

    # Impact level frequencies (5D)
    features[0] = impact_counts[FunctionalImpact.HIGH] / total
    features[1] = impact_counts[FunctionalImpact.MODERATE] / total
    features[2] = impact_counts[FunctionalImpact.LOW] / total
    features[3] = impact_counts[FunctionalImpact.MODIFIER] / total
    features[4] = impact_counts[FunctionalImpact.UNKNOWN] / total

    # Average impact score (1D)
    features[5] = np.mean(scores) if scores else 0.0

    # Max impact score (1D)
    features[6] = np.max(scores) if scores else 0.0

    # Fraction with high/moderate impact (1D)
    high_moderate = impact_counts[FunctionalImpact.HIGH] + impact_counts[FunctionalImpact.MODERATE]
    features[7] = high_moderate / total

    # Transition/transversion ratio (1D)
    # Use Ti/Tv ratio, capped at 10 for numerical stability
    if transversions > 0:
        features[8] = min(transitions / transversions, 10.0) / 10.0  # Normalize to [0, 1]
    else:
        features[8] = 1.0 if transitions > 0 else 0.0

    # Indel fraction (1D)
    features[9] = indels / total

    return features


def compute_quality_metrics(differences: List[VariantDifference]) -> np.ndarray:
    """
    Compute quality metric features from variant differences.

    Computes statistical measures of variant quality scores:
    - Mean quality
    - Standard deviation
    - Minimum quality
    - Maximum quality
    - Median quality

    All values are normalized to [0, 1] assuming quality scores in [0, 100].

    Args:
        differences: List of variant differences

    Returns:
        5-dimensional quality metrics vector

    Example:
        >>> diffs = [VariantDifference(...)]
        >>> metrics = compute_quality_metrics(diffs)
        >>> print(metrics.shape)
        (5,)
    """
    if not differences:
        return np.zeros(5, dtype=np.float32)

    # Collect quality scores (prefer experimental, fallback to reference)
    qualities = []
    for diff in differences:
        if diff.exp_quality is not None and diff.exp_quality > 0:
            qualities.append(diff.exp_quality)
        elif diff.ref_quality is not None and diff.ref_quality > 0:
            qualities.append(diff.ref_quality)

    if not qualities:
        # No quality scores available
        return np.zeros(5, dtype=np.float32)

    qualities = np.array(qualities, dtype=np.float32)

    # Normalize to [0, 1] assuming quality in [0, 100]
    qualities_normalized = np.clip(qualities / 100.0, 0.0, 1.0)

    metrics = np.zeros(5, dtype=np.float32)
    metrics[0] = np.mean(qualities_normalized)
    metrics[1] = np.std(qualities_normalized)
    metrics[2] = np.min(qualities_normalized)
    metrics[3] = np.max(qualities_normalized)
    metrics[4] = np.median(qualities_normalized)

    return metrics


def compute_difference_type_distribution(differences: List[VariantDifference]) -> np.ndarray:
    """
    Compute difference type distribution features.

    Counts frequency of each difference type:
    - New mutations (experimental not in reference)
    - Missing variants (reference not in experimental)
    - Genotype differences (same position, different genotype)

    Args:
        differences: List of variant differences

    Returns:
        3-dimensional difference type distribution vector (normalized frequencies)

    Example:
        >>> diffs = [VariantDifference(...)]
        >>> dist = compute_difference_type_distribution(diffs)
        >>> print(dist.shape)
        (3,)
        >>> print(np.sum(dist))  # Should be close to 1.0
        1.0
    """
    total = len(differences)
    if total == 0:
        return np.zeros(3, dtype=np.float32)

    type_counts = {
        DifferenceType.NEW_MUTATION: 0,
        DifferenceType.MISSING: 0,
        DifferenceType.GENOTYPE_DIFF: 0,
    }

    for diff in differences:
        type_counts[diff.difference_type] += 1

    distribution = np.zeros(3, dtype=np.float32)
    distribution[0] = type_counts[DifferenceType.NEW_MUTATION] / total
    distribution[1] = type_counts[DifferenceType.MISSING] / total
    distribution[2] = type_counts[DifferenceType.GENOTYPE_DIFF] / total

    return distribution


def differences_to_feature_vector(
    differences: List[VariantDifference],
    representative_position: Optional[int] = None,
    position_encoding_dim: int = DIM_POSITION_ENCODING,
) -> np.ndarray:
    """
    Convert variant differences to a fixed-dimensional feature vector.

    Creates a 95-dimensional feature vector from a list of variant differences,
    combining multiple feature types:

    1. Difference type distribution (3D): New, missing, genotype frequencies
    2. Position encoding (64D): Sinusoidal encoding of genomic position
    3. Allele composition (8D): Nucleotide frequencies in ref and alt alleles
    4. Genotype distribution (5D): Frequencies of genotype patterns
    5. Functional impact (10D): Impact scores and variant type statistics
    6. Quality metrics (5D): Statistical measures of variant quality

    Args:
        differences: List of variant differences to encode
        representative_position: Genomic position for encoding (default: median position)
        position_encoding_dim: Dimension for position encoding (default: 64)

    Returns:
        95-dimensional feature vector as numpy array

    Raises:
        ValueError: If differences list is empty or invalid

    Example:
        >>> from genomevault.differential_encoding import (
        ...     compute_variant_differences,
        ...     differences_to_feature_vector
        ... )
        >>> differences = compute_variant_differences(exp_section, ref_section)
        >>> feature_vector = differences_to_feature_vector(differences)
        >>> print(feature_vector.shape)
        (95,)
        >>> print(f"Min: {feature_vector.min():.3f}, Max: {feature_vector.max():.3f}")
        Min: -1.000, Max: 1.000

    Notes:
        - Position encoding uses values in range [-1, 1]
        - All other features are normalized to [0, 1]
        - Empty input returns zero vector
        - Representative position defaults to median of all variant positions
    """
    if not differences:
        logger.warning("Empty differences list, returning zero vector")
        return np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)

    # Determine representative position (use median if not provided)
    if representative_position is None:
        positions = [d.position for d in differences]
        representative_position = int(np.median(positions))

    # Initialize feature vector
    features = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)
    offset = 0

    # 1. Difference type distribution (3D)
    diff_type_dist = compute_difference_type_distribution(differences)
    features[offset:offset + DIM_DIFFERENCE_TYPES] = diff_type_dist
    offset += DIM_DIFFERENCE_TYPES

    # 2. Position encoding (64D)
    pos_encoding = sinusoidal_position_encoding(
        representative_position,
        dimension=position_encoding_dim
    )
    features[offset:offset + position_encoding_dim] = pos_encoding
    offset += position_encoding_dim

    # 3. Allele composition (8D)
    allele_comp = compute_allele_composition(differences)
    features[offset:offset + DIM_ALLELE_COMPOSITION] = allele_comp
    offset += DIM_ALLELE_COMPOSITION

    # 4. Genotype distribution (5D)
    genotype_dist = compute_genotype_distribution(differences)
    features[offset:offset + DIM_GENOTYPE_DIST] = genotype_dist
    offset += DIM_GENOTYPE_DIST

    # 5. Functional impact (10D)
    impact_vector = compute_functional_impact_vector(differences)
    features[offset:offset + DIM_FUNCTIONAL_IMPACT] = impact_vector
    offset += DIM_FUNCTIONAL_IMPACT

    # 6. Quality metrics (5D)
    quality_metrics = compute_quality_metrics(differences)
    features[offset:offset + DIM_QUALITY_METRICS] = quality_metrics
    offset += DIM_QUALITY_METRICS

    assert offset == TOTAL_FEATURE_DIM, f"Feature dimension mismatch: {offset} != {TOTAL_FEATURE_DIM}"

    logger.debug(
        f"Generated {TOTAL_FEATURE_DIM}D feature vector from {len(differences)} differences "
        f"at position {representative_position}"
    )

    return features


def get_feature_names() -> List[str]:
    """
    Get human-readable names for all feature dimensions.

    Returns:
        List of feature names (length 95)

    Example:
        >>> names = get_feature_names()
        >>> print(len(names))
        95
        >>> print(names[:5])
        ['diff_type_new', 'diff_type_missing', 'diff_type_genotype', 'pos_enc_0', 'pos_enc_1']
    """
    names = []

    # Difference types (3)
    names.extend(['diff_type_new', 'diff_type_missing', 'diff_type_genotype'])

    # Position encoding (64)
    names.extend([f'pos_enc_{i}' for i in range(DIM_POSITION_ENCODING)])

    # Allele composition (8)
    for allele_type in ['ref', 'alt']:
        for nuc in ['A', 'C', 'G', 'T']:
            names.append(f'{allele_type}_{nuc}_freq')

    # Genotype distribution (5)
    names.extend(['geno_0/0', 'geno_0/1', 'geno_1/1', 'geno_1/2', 'geno_other'])

    # Functional impact (10)
    names.extend([
        'impact_high_freq',
        'impact_moderate_freq',
        'impact_low_freq',
        'impact_modifier_freq',
        'impact_unknown_freq',
        'impact_avg_score',
        'impact_max_score',
        'impact_high_moderate_frac',
        'impact_ti_tv_ratio',
        'impact_indel_frac',
    ])

    # Quality metrics (5)
    names.extend(['qual_mean', 'qual_std', 'qual_min', 'qual_max', 'qual_median'])

    assert len(names) == TOTAL_FEATURE_DIM, f"Name count mismatch: {len(names)} != {TOTAL_FEATURE_DIM}"

    return names


def describe_feature_vector(feature_vector: np.ndarray) -> Dict[str, any]:
    """
    Create a human-readable description of a feature vector.

    Args:
        feature_vector: 95-dimensional feature vector

    Returns:
        Dictionary with feature descriptions

    Example:
        >>> vector = differences_to_feature_vector(differences)
        >>> description = describe_feature_vector(vector)
        >>> print(description['difference_types'])
        {'new': 0.5, 'missing': 0.3, 'genotype': 0.2}
    """
    if len(feature_vector) != TOTAL_FEATURE_DIM:
        raise ValueError(f"Expected {TOTAL_FEATURE_DIM}D vector, got {len(feature_vector)}D")

    offset = 0

    # Difference types
    diff_types = feature_vector[offset:offset + DIM_DIFFERENCE_TYPES]
    offset += DIM_DIFFERENCE_TYPES

    # Position encoding
    pos_enc = feature_vector[offset:offset + DIM_POSITION_ENCODING]
    offset += DIM_POSITION_ENCODING

    # Allele composition
    allele_comp = feature_vector[offset:offset + DIM_ALLELE_COMPOSITION]
    offset += DIM_ALLELE_COMPOSITION

    # Genotype distribution
    geno_dist = feature_vector[offset:offset + DIM_GENOTYPE_DIST]
    offset += DIM_GENOTYPE_DIST

    # Functional impact
    impact = feature_vector[offset:offset + DIM_FUNCTIONAL_IMPACT]
    offset += DIM_FUNCTIONAL_IMPACT

    # Quality metrics
    quality = feature_vector[offset:offset + DIM_QUALITY_METRICS]
    offset += DIM_QUALITY_METRICS

    return {
        'difference_types': {
            'new': float(diff_types[0]),
            'missing': float(diff_types[1]),
            'genotype': float(diff_types[2]),
        },
        'position_encoding': {
            'dimension': DIM_POSITION_ENCODING,
            'mean': float(np.mean(pos_enc)),
            'std': float(np.std(pos_enc)),
        },
        'allele_composition': {
            'ref': {nuc: float(allele_comp[i]) for i, nuc in enumerate(['A', 'C', 'G', 'T'])},
            'alt': {nuc: float(allele_comp[4 + i]) for i, nuc in enumerate(['A', 'C', 'G', 'T'])},
        },
        'genotype_distribution': {
            '0/0': float(geno_dist[0]),
            '0/1': float(geno_dist[1]),
            '1/1': float(geno_dist[2]),
            '1/2': float(geno_dist[3]),
            'other': float(geno_dist[4]),
        },
        'functional_impact': {
            'high_freq': float(impact[0]),
            'moderate_freq': float(impact[1]),
            'low_freq': float(impact[2]),
            'modifier_freq': float(impact[3]),
            'unknown_freq': float(impact[4]),
            'avg_score': float(impact[5]),
            'max_score': float(impact[6]),
            'high_moderate_frac': float(impact[7]),
            'ti_tv_ratio': float(impact[8]),
            'indel_frac': float(impact[9]),
        },
        'quality_metrics': {
            'mean': float(quality[0]),
            'std': float(quality[1]),
            'min': float(quality[2]),
            'max': float(quality[3]),
            'median': float(quality[4]),
        },
    }
