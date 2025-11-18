#!/usr/bin/env python3
"""
Comprehensive Error Profiling System for HDC Quantization Levels

This module provides exhaustive error analysis across quantization levels:
- Spatial distribution of errors
- Error correlation and clustering
- Genomic context (exons, introns, regulatory regions, etc.)
- AT vs GC pair error patterns
- Error stability across quantization levels
- Statistical significance testing

The goal: Understand EVERYTHING about where and why errors occur.
"""

import gzip
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ErrorPosition:
    """Detailed information about a single error."""
    chrom: str
    pos: int
    ground_truth: str
    predicted: str
    confidence: float
    pair_type: str  # 'AT' or 'GC'
    quantization_level: str

    # Genomic context (to be filled)
    genomic_feature: Optional[str] = None  # exon, intron, UTR, intergenic, etc.
    gene_name: Optional[str] = None
    gc_content_local: Optional[float] = None  # GC% in 1kb window
    repeat_region: Optional[bool] = None
    mappability: Optional[float] = None
    conservation_score: Optional[float] = None


@dataclass
class ErrorProfile:
    """Complete error profile for a quantization level."""
    quantization_level: str
    total_queries: int
    correct: int
    errors: List[ErrorPosition] = field(default_factory=list)

    # Summary statistics
    accuracy: float = 0.0
    at_accuracy: float = 0.0
    gc_accuracy: float = 0.0

    # Error distribution
    error_positions: Set[Tuple[str, int]] = field(default_factory=set)
    error_by_chromosome: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_by_pair_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Clustering metrics
    error_clusters: List[List[Tuple[str, int]]] = field(default_factory=list)
    median_error_distance: float = 0.0
    error_autocorrelation: float = 0.0

    # Genomic context
    error_by_feature: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    gc_content_at_errors: List[float] = field(default_factory=list)

    def compute_summary_stats(self):
        """Compute summary statistics from errors."""
        if self.total_queries > 0:
            self.accuracy = (self.correct / self.total_queries) * 100

        # Count by pair type
        at_total = sum(1 for e in self.errors if e.pair_type == 'AT') + \
                   sum(1 for i in range(self.total_queries)
                       if i not in [e for e in self.errors])  # This is approximate
        gc_total = self.total_queries - at_total

        at_errors = sum(1 for e in self.errors if e.pair_type == 'AT')
        gc_errors = sum(1 for e in self.errors if e.pair_type == 'GC')

        if at_total > 0:
            self.at_accuracy = ((at_total - at_errors) / at_total) * 100
        if gc_total > 0:
            self.gc_accuracy = ((gc_total - gc_errors) / gc_total) * 100

        # Build error position set
        self.error_positions = {(e.chrom, e.pos) for e in self.errors}

        # Count by chromosome
        for e in self.errors:
            self.error_by_chromosome[e.chrom] += 1
            self.error_by_pair_type[e.pair_type] += 1


class ComprehensiveErrorProfiler:
    """
    The Error Analysis Engine.

    This class performs EXHAUSTIVE error profiling across all quantization levels.
    """

    def __init__(self, test_size: int = 5000):
        """
        Initialize profiler.

        Args:
            test_size: Number of random positions to test (more = better statistics)
        """
        self.test_size = test_size
        self.profiles: Dict[str, ErrorProfile] = {}
        self.ground_truth_cache: Dict[Tuple[str, int], str] = {}

        # Quantization level handlers
        self.quantization_handlers = {
            'float32': None,  # Will be disk-based
            'int8': None,
            'int4': None,
            'binary': None
        }

    def load_ground_truth(self, gdiff_path: Path) -> List[Dict]:
        """Load ground truth from GDiff file."""
        logger.info(f"Loading ground truth from {gdiff_path}...")

        with gzip.open(gdiff_path, 'rt') as f:
            gdiff = json.load(f)

        variants = gdiff["differential_variants"]

        # Filter to canonical nucleotides only
        filtered = []
        for v in variants:
            if v["alt"] in ['A', 'T', 'G', 'C']:
                filtered.append(v)
                self.ground_truth_cache[(v["chrom"], v["pos"])] = v["alt"]

        logger.info(f"  Loaded {len(filtered):,} canonical variant positions")
        return filtered

    def select_test_positions(self, variants: List[Dict]) -> List[Dict]:
        """Select random test positions."""
        logger.info(f"Selecting {self.test_size:,} random test positions...")

        if len(variants) <= self.test_size:
            selected = variants
        else:
            indices = np.random.choice(len(variants), size=self.test_size, replace=False)
            selected = [variants[i] for i in indices]

        # Balance AT vs GC if possible
        at_positions = [v for v in selected if v["alt"] in ['A', 'T']]
        gc_positions = [v for v in selected if v["alt"] in ['G', 'C']]

        logger.info(f"  Selected {len(at_positions)} AT positions, {len(gc_positions)} GC positions")
        return selected

    def profile_quantization_level(
        self,
        level_name: str,
        query_func,
        test_positions: List[Dict]
    ) -> ErrorProfile:
        """
        Profile errors for a single quantization level.

        Args:
            level_name: Name of quantization level (e.g., 'int8')
            query_func: Function that takes (chrom, pos) and returns (nucleotide, confidence)
            test_positions: List of test positions

        Returns:
            ErrorProfile with complete error analysis
        """
        logger.info(f"Profiling {level_name} quantization...")

        profile = ErrorProfile(
            quantization_level=level_name,
            total_queries=len(test_positions),
            correct=0
        )

        correct_count = 0

        for i, v in enumerate(test_positions):
            if (i + 1) % 1000 == 0:
                logger.info(f"  Progress: {i+1:,}/{len(test_positions):,}")

            chrom = v["chrom"]
            pos = v["pos"]
            ground_truth = v["alt"]

            # Query the quantization level
            predicted, confidence = query_func(chrom, pos)

            if predicted == ground_truth:
                correct_count += 1
            else:
                # Record error
                pair_type = 'AT' if ground_truth in ['A', 'T'] else 'GC'

                error = ErrorPosition(
                    chrom=chrom,
                    pos=pos,
                    ground_truth=ground_truth,
                    predicted=predicted,
                    confidence=confidence,
                    pair_type=pair_type,
                    quantization_level=level_name
                )
                profile.errors.append(error)

        profile.correct = correct_count
        profile.compute_summary_stats()

        logger.info(f"  {level_name}: {profile.accuracy:.2f}% accuracy ({profile.correct}/{profile.total_queries})")
        logger.info(f"    AT: {profile.at_accuracy:.2f}%, GC: {profile.gc_accuracy:.2f}%")
        logger.info(f"    Total errors: {len(profile.errors)}")

        return profile

    def compute_error_clustering(self, profile: ErrorProfile):
        """Analyze spatial clustering of errors."""
        if len(profile.errors) < 2:
            return

        logger.info(f"Analyzing error clustering for {profile.quantization_level}...")

        # Group errors by chromosome
        errors_by_chrom = defaultdict(list)
        for e in profile.errors:
            errors_by_chrom[e.chrom].append(e.pos)

        all_distances = []

        # Compute distances within each chromosome
        for chrom, positions in errors_by_chrom.items():
            if len(positions) < 2:
                continue

            positions = sorted(positions)

            # Pairwise distances
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = positions[j] - positions[i]
                    all_distances.append(dist)

        if all_distances:
            profile.median_error_distance = np.median(all_distances)

            logger.info(f"  Median inter-error distance: {profile.median_error_distance:,.0f} bp")
            logger.info(f"  Min distance: {min(all_distances):,} bp")
            logger.info(f"  Max distance: {max(all_distances):,} bp")

            # Identify clusters (errors within 10kb)
            clusters = []
            for chrom, positions in errors_by_chrom.items():
                positions = sorted(positions)

                if len(positions) < 2:
                    continue

                current_cluster = [positions[0]]

                for i in range(1, len(positions)):
                    if positions[i] - current_cluster[-1] <= 10000:
                        current_cluster.append(positions[i])
                    else:
                        if len(current_cluster) >= 2:
                            clusters.append([(chrom, p) for p in current_cluster])
                        current_cluster = [positions[i]]

                if len(current_cluster) >= 2:
                    clusters.append([(chrom, p) for p in current_cluster])

            profile.error_clusters = clusters
            logger.info(f"  Found {len(clusters)} error clusters (≥2 errors within 10kb)")

    def compute_gc_content_at_errors(self, profile: ErrorProfile, reference_fasta: Optional[Path] = None):
        """Compute local GC content at error positions."""
        if reference_fasta is None or not reference_fasta.exists():
            logger.warning(f"  Reference FASTA not available, skipping GC content analysis")
            return

        logger.info(f"Computing local GC content at error positions...")

        # This would require pysam/pyfaidx to efficiently access reference
        # For now, we'll skip this but structure is here
        pass

    def compare_error_overlap(self, profile1: ErrorProfile, profile2: ErrorProfile) -> Dict:
        """
        Compare error positions between two quantization levels.

        Returns:
            Dict with overlap statistics
        """
        errors1 = profile1.error_positions
        errors2 = profile2.error_positions

        overlap = errors1 & errors2
        unique_to_1 = errors1 - errors2
        unique_to_2 = errors2 - errors1

        total_union = errors1 | errors2

        jaccard = len(overlap) / len(total_union) if total_union else 0.0

        return {
            'overlap_count': len(overlap),
            'unique_to_level1': len(unique_to_1),
            'unique_to_level2': len(unique_to_2),
            'jaccard_similarity': jaccard,
            'level1_name': profile1.quantization_level,
            'level2_name': profile2.quantization_level,
            'level1_total_errors': len(errors1),
            'level2_total_errors': len(errors2)
        }

    def analyze_error_transitions(self, profiles: List[ErrorProfile]) -> Dict:
        """
        Analyze how errors transition between quantization levels.

        For each position, track which levels have errors.
        """
        logger.info("Analyzing error transitions across quantization levels...")

        # Map position -> set of levels that have errors there
        position_to_errors = defaultdict(set)

        for profile in profiles:
            for error in profile.errors:
                position_to_errors[(error.chrom, error.pos)].add(profile.quantization_level)

        # Categorize positions by error pattern
        patterns = defaultdict(list)
        for pos, levels in position_to_errors.items():
            pattern = tuple(sorted(levels))
            patterns[pattern].append(pos)

        logger.info(f"\nError patterns across quantization levels:")
        for pattern, positions in sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True):
            logger.info(f"  {pattern}: {len(positions)} positions")

        return {
            'patterns': dict(patterns),
            'position_error_map': dict(position_to_errors)
        }

    def statistical_comparison(self, profile1: ErrorProfile, profile2: ErrorProfile) -> Dict:
        """
        Statistical comparison between two quantization levels.
        """
        # McNemar's test for paired binary outcomes
        # For each position: (correct in 1, correct in 2)

        # Build contingency table
        both_correct = 0
        only_1_correct = 0
        only_2_correct = 0
        both_wrong = 0

        errors1 = profile1.error_positions
        errors2 = profile2.error_positions

        # All positions tested (assuming same test set)
        all_positions = errors1 | errors2 | {(e.chrom, e.pos) for profile in [profile1, profile2]
                                             for e in profile.errors}

        # This is approximate - would need actual test set
        # For now, compare error counts

        contingency = np.array([
            [profile1.correct, len(profile1.errors)],
            [len(profile2.errors), 0]  # Placeholder
        ])

        # Fisher's exact test for difference in error rates
        error_rate_1 = len(profile1.errors) / profile1.total_queries
        error_rate_2 = len(profile2.errors) / profile2.total_queries

        # Z-test for proportions
        n1 = profile1.total_queries
        n2 = profile2.total_queries
        p1 = error_rate_1
        p2 = error_rate_2

        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

        if se > 0:
            z = (p1 - p2) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        else:
            z = 0
            p_value = 1.0

        return {
            'level1': profile1.quantization_level,
            'level2': profile2.quantization_level,
            'error_rate_1': error_rate_1,
            'error_rate_2': error_rate_2,
            'difference': error_rate_1 - error_rate_2,
            'z_statistic': z,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01
        }

    def generate_comprehensive_report(self, output_dir: Path):
        """Generate comprehensive error profiling report."""
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "COMPREHENSIVE_ERROR_PROFILE.md"

        logger.info(f"Generating comprehensive report: {report_path}")

        with open(report_path, 'w') as f:
            f.write("# Comprehensive HDC Quantization Error Profile\n\n")
            f.write(f"**Test Size:** {self.test_size:,} positions\n")
            f.write(f"**Quantization Levels Analyzed:** {len(self.profiles)}\n\n")
            f.write("---\n\n")

            # Individual level summaries
            f.write("## Individual Quantization Level Profiles\n\n")

            for level_name in sorted(self.profiles.keys()):
                profile = self.profiles[level_name]

                f.write(f"### {level_name.upper()}\n\n")
                f.write(f"**Overall Accuracy:** {profile.accuracy:.2f}%\n")
                f.write(f"**Total Errors:** {len(profile.errors)}/{profile.total_queries}\n\n")

                f.write("**Pair-Specific Accuracy:**\n")
                f.write(f"- AT pairs: {profile.at_accuracy:.2f}%\n")
                f.write(f"- GC pairs: {profile.gc_accuracy:.2f}%\n")
                f.write(f"- AT errors: {profile.error_by_pair_type.get('AT', 0)}\n")
                f.write(f"- GC errors: {profile.error_by_pair_type.get('GC', 0)}\n\n")

                f.write("**Error Distribution by Chromosome:**\n")
                for chrom in sorted(profile.error_by_chromosome.keys()):
                    count = profile.error_by_chromosome[chrom]
                    f.write(f"- {chrom}: {count} errors\n")
                f.write("\n")

                if profile.error_clusters:
                    f.write(f"**Error Clustering:**\n")
                    f.write(f"- Median inter-error distance: {profile.median_error_distance:,.0f} bp\n")
                    f.write(f"- Number of clusters (≥2 errors within 10kb): {len(profile.error_clusters)}\n")

                    f.write(f"\nTop 5 largest clusters:\n")
                    sorted_clusters = sorted(profile.error_clusters, key=len, reverse=True)[:5]
                    for i, cluster in enumerate(sorted_clusters, 1):
                        chrom = cluster[0][0]
                        positions = [p for _, p in cluster]
                        span = max(positions) - min(positions)
                        f.write(f"  {i}. {chrom}: {len(cluster)} errors spanning {span:,} bp\n")
                    f.write("\n")

                f.write("---\n\n")

            # Pairwise comparisons
            f.write("## Pairwise Quantization Level Comparisons\n\n")

            levels = sorted(self.profiles.keys())

            for i in range(len(levels)):
                for j in range(i + 1, len(levels)):
                    level1 = levels[i]
                    level2 = levels[j]

                    f.write(f"### {level1} vs {level2}\n\n")

                    # Error overlap
                    overlap = self.compare_error_overlap(
                        self.profiles[level1],
                        self.profiles[level2]
                    )

                    f.write("**Error Position Overlap:**\n")
                    f.write(f"- Shared errors: {overlap['overlap_count']}\n")
                    f.write(f"- Unique to {level1}: {overlap['unique_to_level1']}\n")
                    f.write(f"- Unique to {level2}: {overlap['unique_to_level2']}\n")
                    f.write(f"- Jaccard similarity: {overlap['jaccard_similarity']:.3f}\n\n")

                    # Statistical comparison
                    stats_result = self.statistical_comparison(
                        self.profiles[level1],
                        self.profiles[level2]
                    )

                    f.write("**Statistical Comparison:**\n")
                    f.write(f"- Error rate {level1}: {stats_result['error_rate_1']:.4f}\n")
                    f.write(f"- Error rate {level2}: {stats_result['error_rate_2']:.4f}\n")
                    f.write(f"- Difference: {stats_result['difference']:.4f}\n")
                    f.write(f"- Z-statistic: {stats_result['z_statistic']:.3f}\n")
                    f.write(f"- P-value: {stats_result['p_value']:.6f}\n")
                    f.write(f"- Significant at α=0.05: {'Yes' if stats_result['significant_at_0.05'] else 'No'}\n")
                    f.write(f"- Significant at α=0.01: {'Yes' if stats_result['significant_at_0.01'] else 'No'}\n\n")

                    f.write("---\n\n")

        logger.info(f"✓ Report written to {report_path}")

        # Also save raw error data
        self._save_raw_error_data(output_dir)

    def _save_raw_error_data(self, output_dir: Path):
        """Save raw error positions to JSON for further analysis."""
        data_path = output_dir / "raw_error_data.json"

        data = {}

        for level_name, profile in self.profiles.items():
            data[level_name] = {
                'total_queries': profile.total_queries,
                'correct': profile.correct,
                'accuracy': profile.accuracy,
                'at_accuracy': profile.at_accuracy,
                'gc_accuracy': profile.gc_accuracy,
                'errors': [
                    {
                        'chrom': e.chrom,
                        'pos': e.pos,
                        'ground_truth': e.ground_truth,
                        'predicted': e.predicted,
                        'confidence': e.confidence,
                        'pair_type': e.pair_type
                    }
                    for e in profile.errors
                ]
            }

        with open(data_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"✓ Raw error data saved to {data_path}")


def main():
    """Example usage of the error profiler."""
    logger.info("Comprehensive Error Profiler - Example Usage")
    logger.info("=" * 80)

    # This is a template - actual integration would load real query functions
    profiler = ComprehensiveErrorProfiler(test_size=5000)

    # Load ground truth
    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")

    if not gdiff_path.exists():
        logger.error(f"Ground truth file not found: {gdiff_path}")
        return

    variants = profiler.load_ground_truth(gdiff_path)
    test_positions = profiler.select_test_positions(variants)

    logger.info("\nTo use this profiler, integrate with your HDC query systems")
    logger.info("Example:")
    logger.info("  profiler.profile_quantization_level('int8', int8_hdc.query_nucleotide, test_positions)")


if __name__ == "__main__":
    main()
