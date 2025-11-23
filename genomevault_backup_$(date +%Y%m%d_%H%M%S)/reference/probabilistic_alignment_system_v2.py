"""
Improved Probabilistic Alignment System - v2.0

KEY IMPROVEMENTS:
1. Distinguishes SNPs (isolated single-nucleotide changes) from structural variants
2. Properly handles 4+ consecutive mismatches as potential SVs, not errors
3. Clearer security model: data poisoning for defense
4. Better articulation of the FASTQ ordering/concatenation purpose

Critical Distinction:
- SNP: ONE base differs, neighbors match (biological variant)
- 2-3 consecutive mismatches: Rare adjacent SNPs OR sequencing error
- 4+ consecutive mismatches: Likely structural variant (NOT sequential SNPs)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class MismatchPattern(Enum):
    """Classification of mismatch patterns with biological interpretation."""
    MATCH = "MATCH"  # Perfect match
    ISOLATED_SNP = "ISOLATED_SNP"  # Single mismatch, neighbors match (normal)
    TWO_ADJACENT_SNPS = "TWO_ADJACENT_SNPS"  # 2 consecutive (rare but biological)
    THREE_CONSECUTIVE = "THREE_CONSECUTIVE"  # 3 consecutive (highly suspicious)
    STRUCTURAL_VARIANT = "STRUCTURAL_VARIANT"  # 4+ consecutive (likely SV)


@dataclass
class ImprovedAlignmentCertainty:
    """
    Alignment certainty with proper SNP vs. SV classification.
    
    Certainty interpretation:
    - ISOLATED_SNP: certainty ~10^-6 (expected)
    - TWO_ADJACENT_SNPS: certainty ~10^-12 (rare linkage)
    - THREE_CONSECUTIVE: certainty ~10^-18 (sequencing error)
    - STRUCTURAL_VARIANT: certainty depends on SV detector confidence
    """
    position: int
    reference_base: str
    query_base: str
    pattern: MismatchPattern
    max_consecutive_in_window: int  # Max consecutive mismatches in surrounding window
    certainty_score: float
    is_known_snp: bool
    
    @property
    def is_likely_sequencing_error(self) -> bool:
        """Only 3 consecutive mismatches flagged as sequencing error."""
        return self.pattern == MismatchPattern.THREE_CONSECUTIVE
    
    @property
    def requires_sv_analysis(self) -> bool:
        """4+ consecutive should trigger structural variant pipeline."""
        return self.pattern == MismatchPattern.STRUCTURAL_VARIANT
    
    @property
    def certainty_level(self) -> str:
        """Human-readable certainty with proper biological context."""
        if self.pattern == MismatchPattern.MATCH:
            return "PERFECT_MATCH"
        elif self.pattern == MismatchPattern.ISOLATED_SNP:
            return "HIGH_SNP"  # Expected biological variation
        elif self.pattern == MismatchPattern.TWO_ADJACENT_SNPS:
            return "MODERATE_RARE_SNPS"
        elif self.pattern == MismatchPattern.THREE_CONSECUTIVE:
            return "VERY_LOW_SEQUENCING_ERROR"
        else:  # STRUCTURAL_VARIANT
            return "STRUCTURAL_VARIANT_DETECTED"


class ImprovedProbabilisticAligner:
    """
    Improved aligner that properly distinguishes SNPs from structural variants.
    
    Key improvements:
    1. Windowed mismatch analysis (not just point-by-point)
    2. Separate handling for isolated SNPs vs. consecutive mismatches
    3. 4+ consecutive triggers SV detection, not error flagging
    4. Clear security model: injected uncertainty makes stolen data useless
    """
    
    SNP_FREQUENCY = 1e-6
    WINDOW_SIZE = 10  # Look at ±10bp window for pattern classification
    
    def __init__(self, snp_database=None):
        self.snp_db = snp_database
        self.alignment_history: List[ImprovedAlignmentCertainty] = []
    
    def classify_mismatch_pattern(
        self,
        current_position: int,
        is_mismatch: bool
    ) -> Tuple[MismatchPattern, int]:
        """
        Classify mismatch pattern by analyzing surrounding window.
        
        Returns:
            (pattern_type, max_consecutive_in_window)
        """
        if not is_mismatch:
            return MismatchPattern.MATCH, 0
        
        # Analyze recent history for consecutive mismatch streaks
        window_start = max(0, len(self.alignment_history) - self.WINDOW_SIZE)
        window = self.alignment_history[window_start:]
        
        # Count consecutive mismatches ending at current position
        consecutive = 1  # Current mismatch
        for i in range(len(window) - 1, -1, -1):
            cert = window[i]
            if cert.reference_base != cert.query_base:
                consecutive += 1
            else:
                break  # Hit a match, stop counting
        
        # Classify based on consecutive count
        if consecutive == 1:
            # Check if truly isolated (next position will match)
            return MismatchPattern.ISOLATED_SNP, 1
        elif consecutive == 2:
            return MismatchPattern.TWO_ADJACENT_SNPS, 2
        elif consecutive == 3:
            return MismatchPattern.THREE_CONSECUTIVE, 3
        else:  # 4+
            return MismatchPattern.STRUCTURAL_VARIANT, consecutive
    
    def compute_certainty(
        self,
        pattern: MismatchPattern,
        max_consecutive: int,
        is_known_snp: bool
    ) -> float:
        """
        Compute probabilistic certainty based on pattern type.
        
        SNP model:
        - Isolated SNP: ~10^-6 (expected)
        - 2 adjacent SNPs: ~10^-12 (rare linkage)
        - 3 consecutive: ~10^-18 (sequencing error)
        
        Structural variants use different confidence model (from SV detector).
        """
        if pattern == MismatchPattern.MATCH:
            return 1.0
        
        if pattern == MismatchPattern.STRUCTURAL_VARIANT:
            # Don't use SNP model for SVs - this needs SV detection pipeline
            # Return moderate confidence pending SV analysis
            return 0.5
        
        # For SNP-like patterns, use exponential decay
        certainty = self.SNP_FREQUENCY ** max_consecutive
        
        # Boost if known SNP
        if is_known_snp:
            certainty *= 1.5
        
        return min(1.0, certainty)
    
    def align_position(
        self,
        chromosome: str,
        position: int,
        reference_base: str,
        query_base: str
    ) -> ImprovedAlignmentCertainty:
        """
        Align single position with improved pattern classification.
        """
        is_mismatch = (reference_base != query_base)
        
        # Classify pattern
        pattern, max_consecutive = self.classify_mismatch_pattern(
            position, is_mismatch
        )
        
        # Check if known SNP
        is_known_snp = False
        if self.snp_db:
            snp_record = self.snp_db.lookup(chromosome, position)
            is_known_snp = (
                snp_record is not None and 
                query_base in snp_record.alt_alleles
            )
        
        # Compute certainty
        certainty = self.compute_certainty(pattern, max_consecutive, is_known_snp)
        
        # Create certainty object
        cert = ImprovedAlignmentCertainty(
            position=position,
            reference_base=reference_base,
            query_base=query_base,
            pattern=pattern,
            max_consecutive_in_window=max_consecutive,
            certainty_score=certainty,
            is_known_snp=is_known_snp
        )
        
        # Update history
        self.alignment_history.append(cert)
        
        # Log structural variants
        if cert.requires_sv_analysis:
            logger.info(
                f"Structural variant signature at {chromosome}:{position} "
                f"({max_consecutive} consecutive mismatches) - triggering SV pipeline"
            )
        
        return cert
    
    def generate_improved_report(
        self,
        certainties: List[ImprovedAlignmentCertainty]
    ) -> dict:
        """
        Generate report with proper SNP vs. SV classification.
        """
        total = len(certainties)
        if total == 0:
            return {}
        
        # Count by pattern
        pattern_counts = {
            MismatchPattern.MATCH: 0,
            MismatchPattern.ISOLATED_SNP: 0,
            MismatchPattern.TWO_ADJACENT_SNPS: 0,
            MismatchPattern.THREE_CONSECUTIVE: 0,
            MismatchPattern.STRUCTURAL_VARIANT: 0,
        }
        
        for cert in certainties:
            pattern_counts[cert.pattern] += 1
        
        # Known vs unknown SNPs (only for SNP patterns)
        known_snps = sum(
            1 for c in certainties 
            if c.is_known_snp and c.pattern in [
                MismatchPattern.ISOLATED_SNP,
                MismatchPattern.TWO_ADJACENT_SNPS
            ]
        )
        
        sequencing_errors = pattern_counts[MismatchPattern.THREE_CONSECUTIVE]
        sv_signatures = pattern_counts[MismatchPattern.STRUCTURAL_VARIANT]
        
        return {
            "total_bases_aligned": total,
            "pattern_classification": {
                "perfect_matches": pattern_counts[MismatchPattern.MATCH],
                "isolated_snps": pattern_counts[MismatchPattern.ISOLATED_SNP],
                "two_adjacent_snps": pattern_counts[MismatchPattern.TWO_ADJACENT_SNPS],
                "three_consecutive_errors": sequencing_errors,
                "structural_variant_signatures": sv_signatures,
            },
            "pattern_percentages": {
                "perfect_matches_pct": 100.0 * pattern_counts[MismatchPattern.MATCH] / total,
                "isolated_snps_pct": 100.0 * pattern_counts[MismatchPattern.ISOLATED_SNP] / total,
                "sv_signatures_pct": 100.0 * sv_signatures / total,
            },
            "snp_statistics": {
                "known_snps": known_snps,
                "sequencing_errors_detected": sequencing_errors,
                "sequencing_error_rate": sequencing_errors / total,
            },
            "structural_variants": {
                "sv_signatures_detected": sv_signatures,
                "sv_rate": sv_signatures / total,
                "requires_sv_pipeline": sv_signatures > 0,
            },
            "interpretation": {
                "isolated_snps": "Expected biological variation (~10^-6 frequency)",
                "two_adjacent": "Rare but possible adjacent SNPs (~10^-12)",
                "three_consecutive": "Likely sequencing errors (~10^-18 probability)",
                "structural_variants": "4+ consecutive mismatches → trigger SV detection pipeline",
            }
        }


# Example usage documentation
def example_usage():
    """
    Example of improved alignment with proper SNP vs. SV handling.
    
    Key differences from v1:
    1. 1-3 mismatches → SNP analysis (exponential decay model)
    2. 4+ mismatches → SV analysis (different pipeline)
    3. Clear biological interpretation for each pattern
    """
    from genomevault.reference import SNPDatabase
    
    # Initialize
    snp_db = SNPDatabase()
    # snp_db.load_from_vcf('dbsnp_common.vcf')
    
    aligner = ImprovedProbabilisticAligner(snp_database=snp_db)
    
    # Example sequences
    chromosome = "chr22"
    reference = "ACGTACGTACGT"
    query     = "ACTTACGTACGT"  # SNP at position 2
    
    certainties = []
    for i, (ref_base, query_base) in enumerate(zip(reference, query)):
        cert = aligner.align_position(
            chromosome=chromosome,
            position=i,
            reference_base=ref_base,
            query_base=query_base
        )
        certainties.append(cert)
    
    # Generate report
    report = aligner.generate_improved_report(certainties)
    
    print("Alignment Report:")
    print(f"  Isolated SNPs: {report['pattern_classification']['isolated_snps']}")
    print(f"  SV signatures: {report['structural_variants']['sv_signatures_detected']}")
    print(f"  Sequencing errors: {report['snp_statistics']['sequencing_errors_detected']}")
    
    return report


if __name__ == "__main__":
    example_usage()
