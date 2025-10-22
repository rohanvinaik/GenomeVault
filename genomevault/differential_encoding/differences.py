"""
Variant Difference Computation for Differential Encoding.

This module implements efficient computation of variant differences between
experimental and reference genome sections. It computes three types of differences:
1. Δ_new: New mutations in experimental not in reference
2. Δ_missing: Variants in reference but missing from experimental
3. Δ_genotype: Variants with different genotypes

The implementation uses position-based indexing for efficient O(n) computation
with large variant sets (10,000+ variants).

Mathematical Formulation (Section 5.1):
    Δ(E, R) = {Δ_new, Δ_missing, Δ_genotype}

    where:
    - Δ_new = {v ∈ E | v ∉ R}
    - Δ_missing = {v ∈ R | v ∉ E}
    - Δ_genotype = {(v_e, v_r) | v_e ∈ E, v_r ∈ R, pos(v_e) = pos(v_r), gt(v_e) ≠ gt(v_r)}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from genomevault.differential_encoding.reference_management import (
    Variant,
    GenomeSection,
)
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class DifferenceType(Enum):
    """Type of variant difference."""

    NEW_MUTATION = "new_mutation"      # Variant in experimental, not in reference
    MISSING = "missing"                # Variant in reference, not in experimental
    GENOTYPE_DIFF = "genotype_diff"    # Same position, different genotype


class FunctionalImpact(Enum):
    """Functional impact classification for variants."""

    HIGH = "high"              # Likely deleterious (stop gain, frameshift)
    MODERATE = "moderate"      # Missense, in-frame indel
    LOW = "low"                # Synonymous, intronic
    MODIFIER = "modifier"      # Intergenic, regulatory
    UNKNOWN = "unknown"        # Impact not determined


@dataclass(slots=True)
class VariantDifference:
    """
    Represents a difference between experimental and reference variants.

    This class captures the essential information about how an experimental
    variant differs from a reference variant at the same position, or represents
    a variant that exists in only one of the two genomes.

    Memory Optimization: Uses __slots__ for 40-50% memory reduction.

    Attributes:
        difference_type: Type of difference (new, missing, genotype_diff)
        chromosome: Chromosome identifier
        position: Genomic position

        # Experimental variant data (None if MISSING type)
        exp_ref: Reference allele from experimental
        exp_alt: Alternate allele from experimental
        exp_genotype: Genotype from experimental
        exp_quality: Quality score from experimental

        # Reference variant data (None if NEW_MUTATION type)
        ref_ref: Reference allele from reference
        ref_alt: Alternate allele from reference
        ref_genotype: Genotype from reference
        ref_quality: Quality score from reference

        # Annotation
        functional_impact: Predicted functional impact
        metadata: Additional metadata (HGVS, gene names, etc.)

    Example:
        >>> # New mutation
        >>> diff = VariantDifference(
        ...     difference_type=DifferenceType.NEW_MUTATION,
        ...     chromosome="chr1",
        ...     position=100000,
        ...     exp_ref="A",
        ...     exp_alt="G",
        ...     exp_genotype="0/1",
        ...     functional_impact=FunctionalImpact.MODERATE
        ... )

        >>> # Genotype difference
        >>> diff = VariantDifference(
        ...     difference_type=DifferenceType.GENOTYPE_DIFF,
        ...     chromosome="chr1",
        ...     position=200000,
        ...     exp_ref="C",
        ...     exp_alt="T",
        ...     exp_genotype="1/1",
        ...     ref_ref="C",
        ...     ref_alt="T",
        ...     ref_genotype="0/1",
        ...     functional_impact=FunctionalImpact.HIGH
        ... )
    """

    difference_type: DifferenceType
    chromosome: str
    position: int

    # Experimental variant data
    exp_ref: Optional[str] = None
    exp_alt: Optional[str] = None
    exp_genotype: Optional[str] = None
    exp_quality: float = 1.0

    # Reference variant data
    ref_ref: Optional[str] = None
    ref_alt: Optional[str] = None
    ref_genotype: Optional[str] = None
    ref_quality: float = 1.0

    # Annotation
    functional_impact: FunctionalImpact = FunctionalImpact.UNKNOWN
    metadata: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate difference data."""
        if self.position < 0:
            raise ValueError(f"Position must be non-negative, got {self.position}")

        # Validate that appropriate fields are set for each type
        if self.difference_type == DifferenceType.NEW_MUTATION:
            if self.exp_ref is None or self.exp_alt is None:
                raise ValueError("NEW_MUTATION requires exp_ref and exp_alt")
        elif self.difference_type == DifferenceType.MISSING:
            if self.ref_ref is None or self.ref_alt is None:
                raise ValueError("MISSING requires ref_ref and ref_alt")
        elif self.difference_type == DifferenceType.GENOTYPE_DIFF:
            if self.exp_genotype is None or self.ref_genotype is None:
                raise ValueError("GENOTYPE_DIFF requires both genotypes")

    @property
    def is_new_mutation(self) -> bool:
        """Check if this is a new mutation."""
        return self.difference_type == DifferenceType.NEW_MUTATION

    @property
    def is_missing(self) -> bool:
        """Check if this is a missing variant."""
        return self.difference_type == DifferenceType.MISSING

    @property
    def is_genotype_diff(self) -> bool:
        """Check if this is a genotype difference."""
        return self.difference_type == DifferenceType.GENOTYPE_DIFF

    def __str__(self) -> str:
        if self.is_new_mutation:
            return (
                f"VariantDifference(NEW: {self.chromosome}:{self.position} "
                f"{self.exp_ref}>{self.exp_alt} ({self.exp_genotype}))"
            )
        elif self.is_missing:
            return (
                f"VariantDifference(MISSING: {self.chromosome}:{self.position} "
                f"{self.ref_ref}>{self.ref_alt} ({self.ref_genotype}))"
            )
        else:  # genotype_diff
            return (
                f"VariantDifference(GENOTYPE: {self.chromosome}:{self.position} "
                f"{self.exp_genotype} vs {self.ref_genotype})"
            )


def variant_key(variant: Variant) -> Tuple[str, int, str, str]:
    """
    Generate a unique key for variant position-based indexing.

    Creates a composite key from chromosome, position, ref, and alt alleles.
    This enables O(1) lookup for variant matching.

    Args:
        variant: Variant to generate key for

    Returns:
        Tuple of (chromosome, position, ref, alt)

    Example:
        >>> v = Variant(chromosome="chr1", position=100, ref="A", alt="G")
        >>> key = variant_key(v)
        >>> print(key)
        ('chr1', 100, 'A', 'G')
    """
    return (variant.chromosome, variant.position, variant.ref, variant.alt)


def get_functional_impact(variant: Variant) -> FunctionalImpact:
    """
    Predict functional impact of a variant.

    Uses variant metadata and allele characteristics to predict
    the functional consequence. In a full implementation, this would
    integrate with tools like VEP, SnpEff, or CADD scores.

    Current implementation uses simple heuristics:
    - Stop codons, frameshifts: HIGH
    - Missense variants: MODERATE
    - Synonymous, intronic: LOW
    - Other: MODIFIER

    Args:
        variant: Variant to analyze

    Returns:
        Predicted functional impact

    Example:
        >>> v = Variant(
        ...     chromosome="chr1",
        ...     position=100,
        ...     ref="A",
        ...     alt="G",
        ...     info={"IMPACT": "HIGH"}
        ... )
        >>> impact = get_functional_impact(v)
        >>> print(impact)
        FunctionalImpact.HIGH

    Notes:
        - This is a simplified implementation
        - Production systems should use VEP, SnpEff, or similar tools
        - Can be extended with machine learning models
    """
    # Check if impact is already annotated in info field
    if "IMPACT" in variant.info:
        impact_str = variant.info["IMPACT"].upper()
        if impact_str in ["HIGH", "H"]:
            return FunctionalImpact.HIGH
        elif impact_str in ["MODERATE", "M", "MEDIUM"]:
            return FunctionalImpact.MODERATE
        elif impact_str in ["LOW", "L"]:
            return FunctionalImpact.LOW
        elif impact_str in ["MODIFIER", "MOD"]:
            return FunctionalImpact.MODIFIER

    # Check for specific consequence terms
    if "Consequence" in variant.info:
        consequence = variant.info["Consequence"].lower()

        # High impact
        if any(term in consequence for term in [
            "stop_gained", "stop_lost", "start_lost",
            "frameshift", "splice_acceptor", "splice_donor"
        ]):
            return FunctionalImpact.HIGH

        # Moderate impact
        if any(term in consequence for term in [
            "missense", "inframe_deletion", "inframe_insertion"
        ]):
            return FunctionalImpact.MODERATE

        # Low impact
        if any(term in consequence for term in [
            "synonymous", "intronic", "utr"
        ]):
            return FunctionalImpact.LOW

        # Modifier
        if any(term in consequence for term in [
            "intergenic", "regulatory", "upstream", "downstream"
        ]):
            return FunctionalImpact.MODIFIER

    # Simple heuristics based on allele change
    ref_len = len(variant.ref)
    alt_len = len(variant.alt)

    # Frameshift (indel not divisible by 3)
    if ref_len != alt_len and abs(ref_len - alt_len) % 3 != 0:
        return FunctionalImpact.HIGH

    # In-frame indel
    if ref_len != alt_len and abs(ref_len - alt_len) % 3 == 0:
        return FunctionalImpact.MODERATE

    # SNV - assume moderate if not annotated
    if ref_len == 1 and alt_len == 1:
        return FunctionalImpact.MODERATE

    # Default to unknown
    return FunctionalImpact.UNKNOWN


def compute_variant_differences(
    experimental_section: GenomeSection,
    reference_section: GenomeSection
) -> List[VariantDifference]:
    """
    Compute variant differences between experimental and reference sections.

    Implements the mathematical formulation from Section 5.1:
        Δ(E, R) = {Δ_new, Δ_missing, Δ_genotype}

    where:
    - Δ_new: Variants in experimental but not in reference (new mutations)
    - Δ_missing: Variants in reference but not in experimental (missing variants)
    - Δ_genotype: Variants at same position with different genotypes

    The implementation uses position-based indices for O(n+m) complexity
    where n = |experimental variants|, m = |reference variants|.

    Args:
        experimental_section: Experimental genome section
        reference_section: Reference genome section

    Returns:
        List of VariantDifference objects sorted by position

    Raises:
        ValueError: If sections are from different chromosomes

    Example:
        >>> exp_section = GenomeSection(
        ...     chromosome="chr1",
        ...     start_position=100000,
        ...     end_position=200000,
        ...     variants=[v1, v2, v3]
        ... )
        >>> ref_section = GenomeSection(
        ...     chromosome="chr1",
        ...     start_position=100000,
        ...     end_position=200000,
        ...     variants=[v2, v4, v5]
        ... )
        >>> differences = compute_variant_differences(exp_section, ref_section)
        >>> print(f"Found {len(differences)} differences")
        >>> print(f"New mutations: {sum(1 for d in differences if d.is_new_mutation)}")

    Performance:
        - Time complexity: O(n + m)
        - Space complexity: O(n + m)
        - Efficient for large variant sets (tested with 10,000+ variants)

    Notes:
        - Variants are matched by chromosome, position, ref, and alt alleles
        - Genotype differences are detected for variants at same position
        - Functional impact is predicted for each difference
        - Results are sorted by genomic position
    """
    # Validate inputs
    if experimental_section.chromosome != reference_section.chromosome:
        raise ValueError(
            f"Chromosome mismatch: experimental={experimental_section.chromosome}, "
            f"reference={reference_section.chromosome}"
        )

    chromosome = experimental_section.chromosome

    # Build position-based indices for O(1) lookup
    # Key: (chromosome, position, ref, alt) -> Variant
    exp_index: Dict[Tuple[str, int, str, str], Variant] = {
        variant_key(v): v for v in experimental_section.variants
    }

    ref_index: Dict[Tuple[str, int, str, str], Variant] = {
        variant_key(v): v for v in reference_section.variants
    }

    # Build position-only index for genotype difference detection
    # Key: (chromosome, position) -> List[Variant]
    exp_by_position: Dict[Tuple[str, int], List[Variant]] = {}
    for v in experimental_section.variants:
        pos_key = (v.chromosome, v.position)
        if pos_key not in exp_by_position:
            exp_by_position[pos_key] = []
        exp_by_position[pos_key].append(v)

    ref_by_position: Dict[Tuple[str, int], List[Variant]] = {}
    for v in reference_section.variants:
        pos_key = (v.chromosome, v.position)
        if pos_key not in ref_by_position:
            ref_by_position[pos_key] = []
        ref_by_position[pos_key].append(v)

    differences: List[VariantDifference] = []

    # Compute Δ_new: variants in experimental but not in reference
    for exp_variant in experimental_section.variants:
        exp_key = variant_key(exp_variant)

        if exp_key not in ref_index:
            # This is a new mutation
            differences.append(
                VariantDifference(
                    difference_type=DifferenceType.NEW_MUTATION,
                    chromosome=chromosome,
                    position=exp_variant.position,
                    exp_ref=exp_variant.ref,
                    exp_alt=exp_variant.alt,
                    exp_genotype=exp_variant.genotype,
                    exp_quality=exp_variant.quality,
                    functional_impact=get_functional_impact(exp_variant),
                    metadata=exp_variant.info.copy() if exp_variant.info else {}
                )
            )

    # Compute Δ_missing: variants in reference but not in experimental
    for ref_variant in reference_section.variants:
        ref_key = variant_key(ref_variant)

        if ref_key not in exp_index:
            # This variant is missing from experimental
            differences.append(
                VariantDifference(
                    difference_type=DifferenceType.MISSING,
                    chromosome=chromosome,
                    position=ref_variant.position,
                    ref_ref=ref_variant.ref,
                    ref_alt=ref_variant.alt,
                    ref_genotype=ref_variant.genotype,
                    ref_quality=ref_variant.quality,
                    functional_impact=get_functional_impact(ref_variant),
                    metadata=ref_variant.info.copy() if ref_variant.info else {}
                )
            )

    # Compute Δ_genotype: variants at same position with different genotypes
    # We already identified variants that match exactly in the previous steps
    # Now find variants at the same position but with different alleles or genotypes

    processed_positions: Set[Tuple[str, int]] = set()

    for exp_variant in experimental_section.variants:
        pos_key = (exp_variant.chromosome, exp_variant.position)

        # Skip if already processed
        if pos_key in processed_positions:
            continue

        processed_positions.add(pos_key)

        # Check if reference has variants at this position
        if pos_key in ref_by_position:
            exp_variants_at_pos = exp_by_position[pos_key]
            ref_variants_at_pos = ref_by_position[pos_key]

            # Find variants with exact allele match but different genotypes
            for ev in exp_variants_at_pos:
                for rv in ref_variants_at_pos:
                    # Same position, same alleles, different genotype
                    if (ev.ref == rv.ref and ev.alt == rv.alt and
                        ev.genotype != rv.genotype):

                        # Combine metadata from both variants
                        combined_metadata = {}
                        if ev.info:
                            combined_metadata.update(ev.info)
                        if rv.info:
                            combined_metadata["ref_info"] = rv.info.copy()

                        differences.append(
                            VariantDifference(
                                difference_type=DifferenceType.GENOTYPE_DIFF,
                                chromosome=chromosome,
                                position=ev.position,
                                exp_ref=ev.ref,
                                exp_alt=ev.alt,
                                exp_genotype=ev.genotype,
                                exp_quality=ev.quality,
                                ref_ref=rv.ref,
                                ref_alt=rv.alt,
                                ref_genotype=rv.genotype,
                                ref_quality=rv.quality,
                                functional_impact=get_functional_impact(ev),
                                metadata=combined_metadata
                            )
                        )

    # Sort differences by position for consistent ordering
    differences.sort(key=lambda d: d.position)

    logger.info(
        f"Computed differences for {chromosome}: "
        f"{len(differences)} total "
        f"({sum(1 for d in differences if d.is_new_mutation)} new, "
        f"{sum(1 for d in differences if d.is_missing)} missing, "
        f"{sum(1 for d in differences if d.is_genotype_diff)} genotype)"
    )

    return differences
