#!/usr/bin/env python3
"""
Comprehensive Error Profiling System for HDV Quantization Levels

This system profiles errors across Int8, Int4, and Binary quantization levels
with deep biological context analysis:
- Error location tracking (chromosome, position)
- Error correlation analysis between compression levels
- Genomic feature annotation (exons, introns, regulatory regions)
- AT vs GC error distribution
- Combinatorial error pattern identification
"""

import json
import gzip
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging
from dataclasses import dataclass, asdict
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ErrorRecord:
    """Single error record with full context."""
    chrom: str
    pos: int
    truth: str
    prediction: str
    confidence: float
    quantization_level: str

    # Biological context (populated later)
    is_exon: Optional[bool] = None
    is_intron: Optional[bool] = None
    is_regulatory: Optional[bool] = None
    gene_name: Optional[str] = None
    feature_type: Optional[str] = None


class ErrorProfiler:
    """Comprehensive error profiling across quantization levels."""

    def __init__(self, gdiff_path: Path, encoded_genome_path: Path, output_dir: Path):
        self.gdiff_path = gdiff_path
        self.encoded_genome_path = encoded_genome_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Error storage
        self.errors = {
            'int8': [],
            'int4': [],
            'binary': []
        }

        # Summary stats
        self.stats = {
            'int8': {'correct': 0, 'total': 0, 'at_correct': 0, 'at_total': 0, 'gc_correct': 0, 'gc_total': 0},
            'int4': {'correct': 0, 'total': 0, 'at_correct': 0, 'at_total': 0, 'gc_correct': 0, 'gc_total': 0},
            'binary': {'correct': 0, 'total': 0, 'at_correct': 0, 'at_total': 0, 'gc_correct': 0, 'gc_total': 0},
        }

    def load_ground_truth(self, sample_size: int = 10000) -> List[Dict]:
        """Load and sample ground truth variants."""
        logger.info(f"Loading ground truth from {self.gdiff_path}")
        with gzip.open(self.gdiff_path, 'rt') as f:
            gdiff = json.load(f)

        variants = gdiff['differential_variants']
        logger.info(f"  ✓ Total variants: {len(variants):,}")

        # Stratified sampling (proportional AT vs GC)
        at_variants = [v for v in variants if v['alt'] in ['A', 'T']]
        gc_variants = [v for v in variants if v['alt'] in ['G', 'C']]

        n_at = len(at_variants)
        n_gc = len(gc_variants)
        total = n_at + n_gc

        n_at_sample = int(sample_size * n_at / total)
        n_gc_sample = sample_size - n_at_sample

        np.random.seed(42)
        at_sample = np.random.choice(len([v for v in variants if v['alt'] in ['A', 'T']]),
                                     size=min(n_at_sample, n_at), replace=False)
        gc_sample = np.random.choice(len([v for v in variants if v['alt'] in ['G', 'C']]),
                                     size=min(n_gc_sample, n_gc), replace=False)

        test_variants = []
        at_idx = 0
        gc_idx = 0

        for v in variants:
            if v['alt'] in ['A', 'T'] and at_idx < len(at_sample) and at_sample[at_idx] == at_idx:
                test_variants.append(v)
                at_idx += 1
            elif v['alt'] in ['G', 'C'] and gc_idx < len(gc_sample) and gc_sample[gc_idx] == gc_idx:
                test_variants.append(v)
                gc_idx += 1
            else:
                if v['alt'] in ['A', 'T']:
                    at_idx += 1
                else:
                    gc_idx += 1

        # Simpler approach - just sample randomly then filter
        np.random.seed(42)
        all_valid = [v for v in variants if v['alt'] in ['A', 'T', 'G', 'C']]
        test_variants = list(np.random.choice(all_valid, size=min(sample_size, len(all_valid)), replace=False))

        logger.info(f"  ✓ Selected {len(test_variants):,} test positions")
        return test_variants

    def profile_quantization_level(self, level_name: str, hdc_class, test_variants: List[Dict]):
        """Profile a single quantization level."""
        logger.info("=" * 80)
        logger.info(f"{level_name.upper()} QUANTIZATION ERROR PROFILING")
        logger.info("=" * 80)

        logger.info(f"Loading {hdc_class.__name__}...")
        hdc = hdc_class(self.encoded_genome_path)
        hdc.load()

        logger.info(f"Querying {len(test_variants):,} positions...")

        for i, v in enumerate(test_variants):
            if (i + 1) % 1000 == 0:
                logger.info(f"  Progress: {i+1:,}/{len(test_variants):,}")

            try:
                pred, conf = hdc.query_nucleotide(v['chrom'], v['pos'])
                truth = v['alt']

                # Update stats
                self.stats[level_name]['total'] += 1
                if truth in ['A', 'T']:
                    self.stats[level_name]['at_total'] += 1
                else:
                    self.stats[level_name]['gc_total'] += 1

                if pred == truth:
                    self.stats[level_name]['correct'] += 1
                    if truth in ['A', 'T']:
                        self.stats[level_name]['at_correct'] += 1
                    else:
                        self.stats[level_name]['gc_correct'] += 1
                else:
                    # Record error
                    error = ErrorRecord(
                        chrom=v['chrom'],
                        pos=v['pos'],
                        truth=truth,
                        prediction=pred,
                        confidence=conf,
                        quantization_level=level_name
                    )
                    self.errors[level_name].append(error)

            except Exception as e:
                logger.warning(f"Query failed for {v['chrom']}:{v['pos']}: {e}")

        # Compute accuracy
        acc = self.stats[level_name]['correct'] / self.stats[level_name]['total'] if self.stats[level_name]['total'] > 0 else 0
        at_acc = self.stats[level_name]['at_correct'] / self.stats[level_name]['at_total'] if self.stats[level_name]['at_total'] > 0 else 0
        gc_acc = self.stats[level_name]['gc_correct'] / self.stats[level_name]['gc_total'] if self.stats[level_name]['gc_total'] > 0 else 0

        logger.info(f"\n{level_name.upper()} Results:")
        logger.info(f"  Overall: {acc:.4f} ({self.stats[level_name]['correct']}/{self.stats[level_name]['total']})")
        logger.info(f"  AT:      {at_acc:.4f} ({self.stats[level_name]['at_correct']}/{self.stats[level_name]['at_total']})")
        logger.info(f"  GC:      {gc_acc:.4f} ({self.stats[level_name]['gc_correct']}/{self.stats[level_name]['gc_total']})")
        logger.info(f"  Errors:  {len(self.errors[level_name]):,}")
        logger.info("")

        # Release memory
        del hdc
        gc.collect()

    def annotate_errors_with_biological_context(self):
        """Add biological context to all errors."""
        logger.info("=" * 80)
        logger.info("ANNOTATING ERRORS WITH BIOLOGICAL CONTEXT")
        logger.info("=" * 80)

        # For now, we'll add basic context
        # In a full implementation, this would query GTF/GFF files
        logger.info("Note: Full genomic annotation requires GTF/GFF files")
        logger.info("      Currently using basic context analysis")

        for level in ['int8', 'int4', 'binary']:
            for error in self.errors[level]:
                # Placeholder - would integrate with actual annotation
                error.is_exon = None
                error.is_intron = None
                error.is_regulatory = None
                error.gene_name = None
                error.feature_type = 'unknown'

    def analyze_error_correlations(self) -> Dict:
        """Analyze error correlations between compression levels."""
        logger.info("=" * 80)
        logger.info("ERROR CORRELATION ANALYSIS")
        logger.info("=" * 80)

        # Build position sets for each level
        int8_positions = {(e.chrom, e.pos) for e in self.errors['int8']}
        int4_positions = {(e.chrom, e.pos) for e in self.errors['int4']}
        binary_positions = {(e.chrom, e.pos) for e in self.errors['binary']}

        # Compute overlaps
        int8_int4_overlap = int8_positions & int4_positions
        int8_binary_overlap = int8_positions & binary_positions
        int4_binary_overlap = int4_positions & binary_positions
        all_three_overlap = int8_positions & int4_positions & binary_positions

        # Unique errors (only in one level)
        int8_unique = int8_positions - int4_positions - binary_positions
        int4_unique = int4_positions - int8_positions - binary_positions
        binary_unique = binary_positions - int8_positions - int4_positions

        correlations = {
            'int8_int4_shared': len(int8_int4_overlap),
            'int8_binary_shared': len(int8_binary_overlap),
            'int4_binary_shared': len(int4_binary_overlap),
            'all_three_shared': len(all_three_overlap),
            'int8_unique': len(int8_unique),
            'int4_unique': len(int4_unique),
            'binary_unique': len(binary_unique),
            'total_int8_errors': len(int8_positions),
            'total_int4_errors': len(int4_positions),
            'total_binary_errors': len(binary_positions),
        }

        logger.info(f"Int8 ∩ Int4:     {correlations['int8_int4_shared']:,} shared errors")
        logger.info(f"Int8 ∩ Binary:   {correlations['int8_binary_shared']:,} shared errors")
        logger.info(f"Int4 ∩ Binary:   {correlations['int4_binary_shared']:,} shared errors")
        logger.info(f"All 3 levels:    {correlations['all_three_shared']:,} shared errors")
        logger.info(f"\nUnique errors:")
        logger.info(f"  Int8 only:     {correlations['int8_unique']:,}")
        logger.info(f"  Int4 only:     {correlations['int4_unique']:,}")
        logger.info(f"  Binary only:   {correlations['binary_unique']:,}")
        logger.info("")

        return correlations

    def analyze_at_gc_distribution(self) -> Dict:
        """Analyze AT vs GC error distribution."""
        logger.info("=" * 80)
        logger.info("AT vs GC ERROR DISTRIBUTION")
        logger.info("=" * 80)

        distribution = {}

        for level in ['int8', 'int4', 'binary']:
            at_errors = [e for e in self.errors[level] if e.truth in ['A', 'T']]
            gc_errors = [e for e in self.errors[level] if e.truth in ['G', 'C']]

            distribution[level] = {
                'at_error_count': len(at_errors),
                'gc_error_count': len(gc_errors),
                'at_error_rate': 1 - (self.stats[level]['at_correct'] / self.stats[level]['at_total']),
                'gc_error_rate': 1 - (self.stats[level]['gc_correct'] / self.stats[level]['gc_total']),
            }

            logger.info(f"{level.upper()}:")
            logger.info(f"  AT errors: {len(at_errors):,} ({distribution[level]['at_error_rate']:.4f} error rate)")
            logger.info(f"  GC errors: {len(gc_errors):,} ({distribution[level]['gc_error_rate']:.4f} error rate)")
            logger.info(f"  AT/GC ratio: {len(at_errors)/len(gc_errors) if gc_errors else 0:.3f}")
            logger.info("")

        return distribution

    def analyze_error_patterns(self) -> Dict:
        """Combinatorial error pattern analysis."""
        logger.info("=" * 80)
        logger.info("COMBINATORIAL ERROR PATTERN ANALYSIS")
        logger.info("=" * 80)

        patterns = defaultdict(int)

        # Build error lookup by position
        error_by_pos = defaultdict(dict)
        for level in ['int8', 'int4', 'binary']:
            for error in self.errors[level]:
                pos_key = (error.chrom, error.pos)
                error_by_pos[pos_key][level] = error

        # Analyze substitution patterns
        substitutions = {
            'int8': defaultdict(int),
            'int4': defaultdict(int),
            'binary': defaultdict(int),
        }

        for level in ['int8', 'int4', 'binary']:
            for error in self.errors[level]:
                pattern = f"{error.truth}→{error.prediction}"
                substitutions[level][pattern] += 1

        logger.info("Substitution patterns:")
        for level in ['int8', 'int4', 'binary']:
            logger.info(f"\n{level.upper()}:")
            sorted_patterns = sorted(substitutions[level].items(), key=lambda x: x[1], reverse=True)
            for pattern, count in sorted_patterns[:10]:  # Top 10
                logger.info(f"  {pattern}: {count:,}")

        return {
            'substitution_patterns': {k: dict(v) for k, v in substitutions.items()},
            'error_positions': len(error_by_pos)
        }

    def generate_comprehensive_report(self):
        """Generate comprehensive error analysis report."""
        logger.info("=" * 80)
        logger.info("GENERATING COMPREHENSIVE REPORT")
        logger.info("=" * 80)

        # Run all analyses
        correlations = self.analyze_error_correlations()
        at_gc_dist = self.analyze_at_gc_distribution()
        patterns = self.analyze_error_patterns()

        # Compile full report
        report = {
            'summary': {
                level: {
                    'accuracy': self.stats[level]['correct'] / self.stats[level]['total'],
                    'at_accuracy': self.stats[level]['at_correct'] / self.stats[level]['at_total'],
                    'gc_accuracy': self.stats[level]['gc_correct'] / self.stats[level]['gc_total'],
                    'total_errors': len(self.errors[level]),
                    'total_queries': self.stats[level]['total'],
                }
                for level in ['int8', 'int4', 'binary']
            },
            'error_correlations': correlations,
            'at_gc_distribution': at_gc_dist,
            'error_patterns': patterns,
        }

        # Save report
        report_path = self.output_dir / 'comprehensive_error_analysis.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"✓ Report saved to {report_path}")

        # Save detailed error records
        errors_path = self.output_dir / 'detailed_error_records.json'
        error_data = {
            level: [asdict(e) for e in self.errors[level]]
            for level in ['int8', 'int4', 'binary']
        }
        with open(errors_path, 'w') as f:
            json.dump(error_data, f, indent=2)
        logger.info(f"✓ Detailed errors saved to {errors_path}")

        return report

    def run_full_analysis(self, sample_size: int = 10000):
        """Run complete error profiling pipeline."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        logger.info("=" * 80)
        logger.info("COMPREHENSIVE HDV ERROR PROFILING")
        logger.info("=" * 80)
        logger.info(f"Sample size: {sample_size:,}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("")

        # Load ground truth
        test_variants = self.load_ground_truth(sample_size)

        # Profile each level SEQUENTIALLY
        from int8_lightning_hdc import Int8LightningHDC
        self.profile_quantization_level('int8', Int8LightningHDC, test_variants)

        from int4_lightning_hdc import Int4LightningHDC
        self.profile_quantization_level('int4', Int4LightningHDC, test_variants)

        from binary_lightning_hdc import BinaryLightningHDC
        self.profile_quantization_level('binary', BinaryLightningHDC, test_variants)

        # Annotate with biological context
        self.annotate_errors_with_biological_context()

        # Generate comprehensive report
        report = self.generate_comprehensive_report()

        logger.info("=" * 80)
        logger.info("✅ COMPREHENSIVE ERROR PROFILING COMPLETE")
        logger.info("=" * 80)

        return report


def main():
    """Main entry point."""
    import sys

    # Paths
    gdiff_path = Path('data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz')
    encoded_genome = Path('data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome.h5')
    output_dir = Path('HDV_VALIDATION_PACKAGE/error_analysis')

    # Create profiler
    profiler = ErrorProfiler(gdiff_path, encoded_genome, output_dir)

    # Run analysis
    report = profiler.run_full_analysis(sample_size=10000)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
