#!/usr/bin/env python3
"""
Compare HDV encoding across quantization levels using the SAME query set.

This ensures apple-to-apple comparison by testing all quantization modes
on identical positions.

Usage:
    python compare_quantizations.py --sample-size 1000 --seed 42
    python compare_quantizations.py --quantizations float32 int8 binary
"""

import argparse
import logging
import sys
import time
import json
import gzip
import h5py
import numpy as np
import pysam
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

# Import from hdv_validation package
from genomevault.hdv_validation.query_engine import PreEncodedMultiLensHDV, predict_theoretical_multi_lens_voting
from genomevault.hdv_validation.validation_utils import (
    load_validated_n_positions,
    load_gdiff,
    sample_test_positions,
    get_ground_truth,
    predict_multi_lens_voting,
    check_lens_property,
    save_results,
    compute_confusion_matrix
)
from genomevault.hdv_validation.signature_correction import analyze_with_signatures
from genomevault.hdv_validation.generate_collision_beds import generate_collision_beds

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def compare_quantizations_same_queries(
    quantizations=['float32', 'int8', 'int4', 'binary'],
    sample_size=1000,
    seed=42,
    output_dir=None,
    generate_report=False,
    generate_beds=False,
    n_sample_ratio=0.10
):
    """
    Compare all quantization levels using the same query positions.

    This allows direct comparison of how each quantization mode handles
    the exact same positions, revealing systematic differences.

    Args:
        quantizations: List of quantization modes to compare
        sample_size: Total number of positions to sample
        seed: Random seed for reproducibility
        output_dir: Output directory for results
        generate_report: Whether to auto-generate markdown report
        generate_beds: Whether to generate BED files for collision testing
        n_sample_ratio: Ratio of positions to sample from validated N positions (0.0-1.0)
    """
    if output_dir is None:
        output_dir = Path("genomevault/hdv_validation/results/comparison_results")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("QUANTIZATION COMPARISON WITH SAME QUERY SET")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Quantization modes: {', '.join(quantizations)}")
    logger.info(f"Sample size: {sample_size:,}")
    logger.info(f"N position sampling ratio: {n_sample_ratio:.1%}")
    logger.info(f"Random seed: {seed}")
    logger.info("")
    
    # Helper function to get correct H5 file path for quantization mode
    def get_h5_path(quantization: str) -> Path:
        base_dir = Path("data/experimental_strands/ERR3239334/hdv_encoding")
        if quantization == 'float32':
            return base_dir / "encoded_genome_5lenses_3d.h5"
        elif quantization == 'int8':
            return base_dir / "encoded_genome_5lenses_3d_int8.h5"
        elif quantization == 'int4':
            return base_dir / "encoded_genome_5lenses_3d_int4.h5"
        elif quantization == 'binary':
            return base_dir / "encoded_genome_5lenses_3d_binary.h5"
        else:
            raise ValueError(f"Unknown quantization mode: {quantization}")

    # Paths
    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    guide_fasta_dir = Path("/Volumes/1TBStorage/guide_strands")
    validated_n_path = Path("HDV_VALIDATION_PACKAGE/validated_n_positions.json")
    exp_bam_path = Path("data/experimental_strands/ERR3239334/alignment/k11_bams/experimental_vs_ref1.sorted.bam")

    # Load data once (shared across all quantizations)
    logger.info("Loading shared data...")

    # Load GDiff
    gdiff, variant_index = load_gdiff(gdiff_path)
    logger.info(f"  ✓ Loaded {len(variant_index):,} variants from GDiff")

    # Load validated N positions
    validated_n_positions = load_validated_n_positions(validated_n_path)
    logger.info(f"  ✓ Loaded {len(validated_n_positions):,} validated N positions")

    # Sample test positions (same for all quantizations!)
    logger.info("")
    logger.info("Sampling test positions...")

    # Get chunk keys from first quantization's H5 file (all should have same chunks)
    reference_h5_path = get_h5_path(quantizations[0])
    with h5py.File(reference_h5_path, 'r') as f:
        chunk_keys_bytes = f['chunk_keys'][:]
        chunk_keys = [k.decode('utf-8') for k in chunk_keys_bytes]
    
    test_positions, high_n_set = sample_test_positions(
        chunk_keys,
        validated_n_positions,
        sample_size,
        n_sample_ratio=n_sample_ratio,
        seed=seed
    )
    
    logger.info(f"  ✓ Sampled {len(test_positions):,} positions")
    logger.info(f"    - Validated N positions: {len(high_n_set)} ({len(high_n_set)/len(test_positions)*100:.1f}%)")
    logger.info(f"    - General genome: {len(test_positions) - len(high_n_set)}")
    logger.info("")
    
    # Open experimental BAM (shared)
    exp_bam = pysam.AlignmentFile(str(exp_bam_path), 'rb') if exp_bam_path.exists() else None
    region_map = gdiff.get("region_guide_map", {})
    
    # Get ground truth for all positions (shared)
    logger.info("Computing ground truth for all positions...")
    ground_truths = []
    valid_positions = []
    n_positions_count = 0

    for chrom, pos in test_positions:
        gt, guide_idx, has_n = get_ground_truth(chrom, pos, variant_index, exp_bam, region_map)

        # Accept both ATGC (normal) and N (no coverage) positions
        if gt and gt in 'ATGCN':
            ground_truths.append({
                'chrom': chrom,
                'pos': pos,
                'nucleotide': gt,
                'guide_idx': guide_idx,
                'has_n': has_n,
                'is_high_n': (chrom, pos) in high_n_set
            })
            valid_positions.append((chrom, pos))
            if has_n:
                n_positions_count += 1

    logger.info(f"  ✓ {len(ground_truths):,} positions with valid ground truth")
    logger.info(f"    - N positions (no coverage): {n_positions_count}")
    logger.info(f"    - ATGC positions (has coverage): {len(ground_truths) - n_positions_count}")
    logger.info("")
    
    if exp_bam:
        exp_bam.close()
    
    # Now test each quantization on the same positions
    results_by_quant = {}
    predictions_by_quant = {}
    timing_by_quant = {}
    
    for quant in quantizations:
        logger.info("=" * 80)
        logger.info(f"TESTING: {quant.upper()}")
        logger.info("=" * 80)
        logger.info("")
        
        start_time = time.time()
        
        # Load HDV for this quantization
        logger.info(f"Loading {quant} HDV system...")
        quant_h5_path = get_h5_path(quant)
        logger.info(f"  H5 file: {quant_h5_path}")
        hdv = PreEncodedMultiLensHDV(
            quant_h5_path,
            guide_fasta_dir=guide_fasta_dir,
            quantization=quant
        )
        load_time = time.time() - start_time
        logger.info(f"  ✓ Loaded in {load_time:.2f}s")
        logger.info(f"  Using OPTIMAL per-lens thresholds for {quant}")
        logger.info("")

        # Test all positions
        predictions = []
        lens_results_all = []
        query_start_time = time.time()

        for i, gt in enumerate(ground_truths):
            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i+1}/{len(ground_truths)}")

            chrom = gt['chrom']
            pos = gt['pos']

            # Query all 5 lenses
            lens_results = hdv.query_position_all_lenses(chrom, pos)

            # Multi-lens prediction with optimal thresholds
            pred, conf, votes = predict_multi_lens_voting(lens_results, quantization=quant)
            
            predictions.append({
                'position': f"{chrom}:{pos}",
                'ground_truth': gt['nucleotide'],
                'predicted': pred,
                'confidence': conf,
                'correct': pred == gt['nucleotide'],
                'has_n': gt['has_n'],
                'is_high_n': gt['is_high_n'],
                'lens_results': lens_results,
                'votes': votes
            })
            lens_results_all.append(lens_results)
        
        query_time = time.time() - query_start_time
        total_time = time.time() - start_time
        
        logger.info("")
        logger.info(f"✓ Queries completed in {query_time:.2f}s")
        logger.info(f"  Load time: {load_time:.2f}s")
        logger.info(f"  Query time: {query_time:.2f}s")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Time per query: {(query_time / len(ground_truths)) * 1000:.3f}ms")
        logger.info("")
        
        # Compute statistics
        # Only compute accuracy for positions with experimental coverage (not N)
        observed_correct = sum(1 for p in predictions if p['correct'] and not p['has_n'])
        observed_total = sum(1 for p in predictions if not p['has_n'])
        observed_acc = observed_correct / observed_total if observed_total > 0 else 0

        # For N positions, compute theoretical predictions using only 3 non-complementary lenses
        n_predictions = [p for p in predictions if p['has_n']]
        unvalidated_predictions = len(n_predictions)

        # Count high-confidence theoretical predictions (confidence >= 0.8)
        high_conf_unvalidated = 0
        for p in n_predictions:
            # For N positions, use only PuPy, AmKe, StWk lenses for theoretical prediction
            _, conf, _ = predict_theoretical_multi_lens_voting(p['lens_results'])
            if conf >= 0.8:
                high_conf_unvalidated += 1

        # Combined theoretical accuracy = (observed_correct + high_conf_unvalidated) / observed_total
        combined_theoretical_correct = observed_correct + high_conf_unvalidated
        combined_theoretical_acc = combined_theoretical_correct / observed_total if observed_total > 0 else 0

        logger.info(f"Observed Accuracy: {observed_acc*100:.2f}% ({observed_correct}/{observed_total})")
        logger.info(f"Combined Theoretical Accuracy: {combined_theoretical_acc*100:.2f}% ({combined_theoretical_correct}/{observed_total})")
        logger.info(f"  = Observed + {high_conf_unvalidated} high-confidence predictions from N sites")
        logger.info(f"Unvalidated Biophysical Recovery: {unvalidated_predictions} predictions made")
        logger.info(f"  (N positions - no experimental coverage, no ground truth available)")
        logger.info("")
        
        # Store results
        predictions_by_quant[quant] = predictions
        timing_by_quant[quant] = {
            'load_time_seconds': load_time,
            'query_time_seconds': query_time,
            'total_time_seconds': total_time,
            'time_per_query_ms': (query_time / len(ground_truths)) * 1000,
            'queries_per_second': len(ground_truths) / query_time
        }
        
        # Compute confusion matrix
        pred_list = [p['predicted'] for p in predictions]
        truth_list = [p['ground_truth'] for p in predictions]
        confusion = compute_confusion_matrix(pred_list, truth_list)
        
        results_by_quant[quant] = {
            'observed_accuracy': observed_acc,
            'combined_theoretical_accuracy': combined_theoretical_acc,
            'high_confidence_unvalidated': high_conf_unvalidated,
            'total_positions': len(predictions),
            'observed_positions': observed_total,
            'unvalidated_predictions': unvalidated_predictions,
            'confusion_matrix': confusion,
            'timing': timing_by_quant[quant]
        }
        
        # Close HDV
        hdv.close()
    
    # Analyze differences between quantizations
    logger.info("=" * 80)
    logger.info("CROSS-QUANTIZATION ANALYSIS")
    logger.info("=" * 80)
    logger.info("")
    
    # Find positions where predictions differ
    disagreement_analysis = analyze_disagreements(predictions_by_quant, quantizations)
    
    # Compute pairwise agreement
    pairwise_agreement = compute_pairwise_agreement(predictions_by_quant, quantizations)
    
    # Error correlation
    error_correlation = compute_error_correlation(predictions_by_quant, quantizations)
    
    # Generate comprehensive report
    report = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'quantizations': quantizations,
            'sample_size': sample_size,
            'n_sample_ratio': n_sample_ratio,
            'seed': seed,
            'test_positions_count': len(ground_truths),
            'n_positions_count': n_positions_count
        },
        'results_by_quantization': results_by_quant,
        'disagreement_analysis': disagreement_analysis,
        'pairwise_agreement': pairwise_agreement,
        'error_correlation': error_correlation,
        'ground_truth_positions': ground_truths[:100]  # Save first 100 for reference
    }
    
    # Save report
    report_file = output_dir / "quantization_comparison_same_queries.json"
    save_results(report, report_file)
    
    # Save detailed predictions for each quantization
    for quant in quantizations:
        pred_file = output_dir / f"{quant}_predictions_detailed.json"
        with open(pred_file, 'w') as f:
            json.dump(predictions_by_quant[quant], f, indent=2)
        logger.info(f"✓ Saved {quant} predictions to: {pred_file}")

    logger.info("")

    # Categorize errors by precision level
    logger.info("Categorizing errors by precision level...")
    high_precision_errors, low_precision_errors, common_errors = categorize_errors_by_precision(
        predictions_by_quant, quantizations
    )

    # Save error categorization files
    error_files = {
        'high_precision_errors': (high_precision_errors, "Errors common to float32 and int8"),
        'low_precision_errors': (low_precision_errors, "Errors common to int4 and binary"),
        'common_errors': (common_errors, "Errors found in ALL quantizations")
    }

    for filename, (error_list, description) in error_files.items():
        error_file = output_dir / f"{filename}.json"
        error_data = {
            'description': description,
            'count': len(error_list),
            'errors': error_list
        }
        with open(error_file, 'w') as f:
            json.dump(error_data, f, indent=2)
        logger.info(f"✓ Saved {filename}: {len(error_list)} errors to {error_file}")

    logger.info("")

    # Generate BED files for collision testing (if requested)
    if generate_beds and len(high_precision_errors) > 0:
        logger.info("=" * 80)
        logger.info("GENERATING COLLISION BED FILES")
        logger.info("=" * 80)
        logger.info("")

        try:
            bed_output_dir = output_dir.parent / "bed_files"
            high_precision_error_file = output_dir / "high_precision_errors.json"

            hdc_bed, bam_bed = generate_collision_beds(
                input_json=high_precision_error_file,
                output_dir=bed_output_dir,
                quantization='float32',
                fallback_quantization='int8'
            )

        except Exception as e:
            logger.error(f"Failed to generate BED files: {e}")
            logger.warning("Continuing without BED files...")

    # Apply adaptive threshold correction to all quantizations
    logger.info("=" * 80)
    logger.info("ADAPTIVE THRESHOLD CORRECTION ANALYSIS")
    logger.info("=" * 80)
    logger.info("")

    correction_results_by_quant = {}

    for quant in quantizations:
        logger.info(f"Analyzing {quant} with signature-based correction...")

        # Use exhaustive ALL_CORRECT signatures (tested on ALL 9K+ correct predictions)
        signatures_path = output_dir.parent / "signature_corrections/exhaustive_ALL_CORRECT" / f"{quant}_exhaustive_search_results.json"
        logger.info(f"  Using exhaustive ALL_CORRECT signatures: {signatures_path}")

        # Apply signature-based correction
        correction_analysis = analyze_with_signatures(
            predictions_by_quant[quant],
            signatures_path=signatures_path,
            quantization=quant
        )

        correction_results_by_quant[quant] = correction_analysis

        # Save corrected predictions
        corrected_pred_file = output_dir / f"{quant}_predictions_corrected.json"
        with open(corrected_pred_file, 'w') as f:
            json.dump(correction_analysis['corrected_predictions'], f, indent=2)
        logger.info(f"  ✓ Saved corrected predictions to: {corrected_pred_file}")

        # Save correction statistics
        correction_stats_file = output_dir / f"{quant}_correction_stats.json"
        with open(correction_stats_file, 'w') as f:
            json.dump(correction_analysis['statistics'], f, indent=2)
        logger.info(f"  ✓ Saved correction statistics to: {correction_stats_file}")

        # Log correction summary
        stats = correction_analysis['statistics']
        logger.info(f"  Signatures loaded: {stats['signatures_loaded']}")
        logger.info(f"  Baseline accuracy: {stats.get('baseline_accuracy', 0)*100:.2f}%")
        logger.info(f"  Corrected accuracy: {stats.get('corrected_accuracy', 0)*100:.2f}%")
        logger.info(f"  Improvement: {stats.get('improvement', 0)*100:+.2f}%")
        logger.info(f"  Corrections applied: {stats['corrections_applied']}")
        logger.info(f"    - Fixed errors: {stats['corrections_that_fixed_errors']}")
        logger.info(f"    - Introduced errors: {stats['corrections_that_introduced_errors']}")
        if stats.get('transforms_used'):
            logger.info(f"  Top transforms used:")
            sorted_transforms = sorted(stats['transforms_used'].items(), key=lambda x: x[1], reverse=True)[:5]
            for transform, count in sorted_transforms:
                logger.info(f"    - {transform}: {count} times")
        logger.info("")

    # Print summary
    print_comparison_summary(results_by_quant, correction_results_by_quant, pairwise_agreement, quantizations)

    # Generate markdown report if requested
    if generate_report:
        logger.info("")
        logger.info("=" * 80)
        logger.info("GENERATING COMPREHENSIVE MARKDOWN REPORT")
        logger.info("=" * 80)
        logger.info("")

        try:
            from genomevault.hdv_validation.generate_report import generate_markdown_report

            # Prepare paths for report generation
            summary_path = output_dir / "quantization_comparison_same_queries.json"
            detailed_paths = {
                quant: output_dir / f"{quant}_predictions_detailed.json"
                for quant in quantizations
            }

            # Generate report
            report_output_dir = output_dir.parent.parent / "reports"
            report_output_path = report_output_dir / "quantization_validation_report.md"

            generate_markdown_report(summary_path, detailed_paths, report_output_path)

            logger.info("")
            logger.info(f"✓ Markdown report generated: {report_output_path}")
            logger.info("")

        except ImportError as e:
            logger.warning(f"Could not import report generator: {e}")
            logger.warning("Skipping report generation")
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            logger.error("Report generation failed, but validation results are saved")

    return report


def analyze_disagreements(predictions_by_quant: Dict, quantizations: List[str]) -> Dict:
    """Analyze positions where different quantizations disagree."""
    n_positions = len(predictions_by_quant[quantizations[0]])
    
    disagreements = []
    
    for i in range(n_positions):
        preds = {q: predictions_by_quant[q][i] for q in quantizations}
        
        # Check if all agree
        predicted_nucs = [p['predicted'] for p in preds.values()]
        if len(set(predicted_nucs)) > 1:
            # Disagreement found
            disagreements.append({
                'position': preds[quantizations[0]]['position'],
                'ground_truth': preds[quantizations[0]]['ground_truth'],
                'predictions': {q: preds[q]['predicted'] for q in quantizations},
                'confidences': {q: preds[q]['confidence'] for q in quantizations},
                'correctness': {q: preds[q]['correct'] for q in quantizations}
            })
    
    # Categorize disagreements
    categories = {
        'all_wrong': 0,
        'some_correct': 0,
        'all_correct_but_differ': 0
    }
    
    for d in disagreements:
        correct_count = sum(d['correctness'].values())
        if correct_count == 0:
            categories['all_wrong'] += 1
        elif correct_count == len(quantizations):
            categories['all_correct_but_differ'] += 1
        else:
            categories['some_correct'] += 1
    
    return {
        'total_disagreements': len(disagreements),
        'disagreement_rate': len(disagreements) / n_positions,
        'categories': categories,
        'examples': disagreements[:20]  # First 20 examples
    }


def compute_pairwise_agreement(predictions_by_quant: Dict, quantizations: List[str]) -> Dict:
    """Compute pairwise agreement between quantization modes."""
    n_positions = len(predictions_by_quant[quantizations[0]])
    
    agreement = {}
    
    for i, q1 in enumerate(quantizations):
        for q2 in quantizations[i+1:]:
            pair_key = f"{q1}_vs_{q2}"
            
            agrees = sum(
                1 for j in range(n_positions)
                if predictions_by_quant[q1][j]['predicted'] == predictions_by_quant[q2][j]['predicted']
            )
            
            agreement[pair_key] = {
                'agreement_count': agrees,
                'agreement_rate': agrees / n_positions,
                'disagreement_count': n_positions - agrees
            }
    
    return agreement


def compute_error_correlation(predictions_by_quant: Dict, quantizations: List[str]) -> Dict:
    """Analyze correlation of errors between quantization modes."""
    n_positions = len(predictions_by_quant[quantizations[0]])
    
    # Build error vectors (1 if wrong, 0 if correct)
    error_vectors = {}
    for q in quantizations:
        error_vectors[q] = np.array([
            0 if predictions_by_quant[q][i]['correct'] else 1
            for i in range(n_positions)
        ])
    
    # Compute correlation matrix
    corr_matrix = {}
    for q1 in quantizations:
        for q2 in quantizations:
            if q1 == q2:
                corr_matrix[f"{q1}_vs_{q2}"] = 1.0
            else:
                # Compute correlation coefficient
                corr = np.corrcoef(error_vectors[q1], error_vectors[q2])[0, 1]
                corr_matrix[f"{q1}_vs_{q2}"] = float(corr)
    
    # Find positions where all fail or all succeed
    all_correct = sum(1 for i in range(n_positions) if all(
        predictions_by_quant[q][i]['correct'] for q in quantizations
    ))
    
    all_wrong = sum(1 for i in range(n_positions) if all(
        not predictions_by_quant[q][i]['correct'] for q in quantizations
    ))
    
    return {
        'correlation_matrix': corr_matrix,
        'all_correct': all_correct,
        'all_wrong': all_wrong,
        'all_correct_rate': all_correct / n_positions,
        'all_wrong_rate': all_wrong / n_positions
    }


def categorize_errors_by_precision(
    predictions_by_quant: Dict,
    quantizations: List[str]
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Categorize errors into three groups:
    - high_precision_errors: errors common to both float32 and int8
    - low_precision_errors: errors common to both int4 and binary
    - common_errors: errors found in all quantization levels

    Returns:
        (high_precision_errors, low_precision_errors, common_errors)
    """
    n_positions = len(predictions_by_quant[quantizations[0]])

    high_precision_errors = []
    low_precision_errors = []
    common_errors = []

    for i in range(n_positions):
        # Get error status for each quantization
        float32_wrong = 'float32' in quantizations and not predictions_by_quant['float32'][i]['correct']
        int8_wrong = 'int8' in quantizations and not predictions_by_quant['int8'][i]['correct']
        int4_wrong = 'int4' in quantizations and not predictions_by_quant['int4'][i]['correct']
        binary_wrong = 'binary' in quantizations and not predictions_by_quant['binary'][i]['correct']

        # Build error entry with full details
        error_entry = {
            'position': predictions_by_quant[quantizations[0]][i]['position'],
            'ground_truth': predictions_by_quant[quantizations[0]][i]['ground_truth'],
            'predictions': {},
            'confidences': {},
            'lens_results': {},
            'votes': {}
        }

        for q in quantizations:
            pred_data = predictions_by_quant[q][i]
            error_entry['predictions'][q] = pred_data['predicted']
            error_entry['confidences'][q] = pred_data['confidence']
            error_entry['lens_results'][q] = pred_data['lens_results']
            error_entry['votes'][q] = pred_data['votes']

        # Check if all quantizations wrong (common error)
        all_wrong = all(
            not predictions_by_quant[q][i]['correct'] for q in quantizations
        )

        if all_wrong:
            common_errors.append(error_entry)

        # High precision errors: both float32 AND int8 wrong
        if float32_wrong and int8_wrong:
            high_precision_errors.append(error_entry)

        # Low precision errors: both int4 AND binary wrong
        if int4_wrong and binary_wrong:
            low_precision_errors.append(error_entry)

    return high_precision_errors, low_precision_errors, common_errors


def print_comparison_summary(results: Dict, correction_results: Dict, pairwise: Dict, quantizations: List[str]):
    """Print formatted comparison summary with adaptive correction results."""
    logger.info("=" * 80)
    logger.info("SUMMARY: QUANTIZATION COMPARISON")
    logger.info("=" * 80)
    logger.info("")

    # Accuracy table with baseline, theoretical, and corrected
    logger.info("Accuracy Comparison:")
    logger.info(f"{'Quantization':<12} {'Observed':<10} {'Theoretical':<12} {'Corrected':<11} {'Improvement':<12} {'High-Conf N':<12} {'Time/Query':<12}")
    logger.info("-" * 110)

    for q in quantizations:
        r = results[q]
        c = correction_results[q]['statistics']
        improvement = c.get('improvement', c.get('accuracy_improvement', 0))

        logger.info(
            f"{q:<12} "
            f"{r['observed_accuracy']*100:>9.2f}% "
            f"{r['combined_theoretical_accuracy']*100:>11.2f}% "
            f"{c['corrected_accuracy']*100:>10.2f}% "
            f"{improvement*100:>+11.2f}% "
            f"{r['high_confidence_unvalidated']:>11} "
            f"{r['timing']['time_per_query_ms']:>11.3f}ms"
        )

    logger.info("")
    logger.info("Pairwise Agreement:")
    for pair, stats in pairwise.items():
        logger.info(f"  {pair}: {stats['agreement_rate']*100:.2f}% agreement")

    logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description='Compare HDV quantizations with same query set'
    )
    parser.add_argument(
        '--quantizations',
        nargs='+',
        default=['float32', 'int8', 'int4', 'binary'],
        choices=['float32', 'int8', 'int4', 'binary'],
        help='Quantization modes to compare'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1000,
        help='Number of test positions'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Automatically generate comprehensive markdown report after comparison'
    )
    parser.add_argument(
        '--generate-beds',
        action='store_true',
        help='Generate BED files for UCSC collision testing from high-precision errors'
    )
    parser.add_argument(
        '--n-sample-ratio',
        type=float,
        default=0.10,
        help='Ratio of positions to sample from validated N positions (0.0-1.0, default: 0.10)'
    )

    args = parser.parse_args()

    compare_quantizations_same_queries(
        quantizations=args.quantizations,
        sample_size=args.sample_size,
        seed=args.seed,
        output_dir=args.output_dir,
        generate_report=args.generate_report,
        generate_beds=args.generate_beds,
        n_sample_ratio=args.n_sample_ratio
    )


if __name__ == '__main__':
    main()
