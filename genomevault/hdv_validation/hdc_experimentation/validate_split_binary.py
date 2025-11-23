#!/usr/bin/env python3
"""
Validate 6-Bank Split Binary Architecture

Tests the new within-lens splitting approach:
- 3 ternary banks → 6 binary banks
- Hydrophobic: A bank + T bank
- MajorGroove: G bank + C bank
- Hinge: pos bank + neg bank

Outputs reports to: genomevault/hdv_validation/hdc_experimentation/docs/
"""

import h5py
import numpy as np
import json
import logging
import argparse
import pysam
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

# Import from main validation system
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from genomevault.hdv_validation.validation_utils import (
    load_gdiff,
    sample_test_positions,
    get_ground_truth,
    compute_confusion_matrix
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class SplitBinaryMultiLensHDV:
    """
    Query system for 6-bank split binary architecture.

    Architecture:
        - Hydrophobic_A (bank 0): +1 for A nucleotides
        - Hydrophobic_T (bank 1): +1 for T nucleotides
        - MajorGroove_G (bank 2): +1 for G nucleotides
        - MajorGroove_C (bank 3): +1 for C nucleotides
        - Hinge_pos (bank 4): +1 for positive hinge (G, C)
        - Hinge_neg (bank 5): +1 for negative hinge (A, T)
    """

    def __init__(self, hdf5_path: Path, D=10240, N=2000, seed=42):
        self.D = D
        self.N = N
        self.hdf5_path = hdf5_path

        # Open H5 file
        self.h5_file = h5py.File(hdf5_path, 'r')
        h5_dataset = self.h5_file['binary_bank_vectors']

        # Get metadata before loading
        self.bank_names = [name.decode('utf-8') if isinstance(name, bytes) else name
                          for name in h5_dataset.attrs['bank_names']]
        self.num_banks = h5_dataset.attrs['num_banks']
        self.dimension = h5_dataset.attrs['dimension']

        logger.info(f"  Banks: {self.bank_names}")
        logger.info(f"  Shape: {h5_dataset.shape}")
        logger.info(f"  Dimension: {self.dimension}")

        # Load entire dataset into RAM for instant queries (20 GB uncompressed)
        logger.info(f"  Loading entire dataset into RAM...")
        import time
        t0 = time.time()
        self.bank_vectors = h5_dataset[:]  # Load all data into numpy array
        t1 = time.time()
        logger.info(f"  ✓ Loaded {self.bank_vectors.nbytes / (1024**3):.2f} GB in {t1-t0:.1f}s")

        # Load chunk keys
        chunk_keys_bytes = self.h5_file.get('chunk_keys', None)
        if chunk_keys_bytes is None:
            # Try to load from source file (use h5_dataset for attrs, not numpy array)
            source_file = h5_dataset.attrs.get('source_file', None)
            if source_file:
                logger.info(f"  Loading chunk keys from source: {source_file}")
                with h5py.File(source_file, 'r') as src:
                    chunk_keys_bytes = src['chunk_keys'][:]
            else:
                raise ValueError("No chunk_keys dataset found!")

        self.chunk_keys = [k.decode('utf-8') for k in chunk_keys_bytes]
        self.chunk_index = {key: idx for idx, key in enumerate(self.chunk_keys)}
        self.total_chunks = len(self.chunk_keys)

        logger.info(f"  Total chunks: {self.total_chunks:,}")

        # Generate positional vectors
        np.random.seed(seed)
        self.pos_vectors = np.random.choice([-1, 1], size=(N, D)).astype(np.int8)

    def query_position(self, chrom: str, pos: int) -> Dict[str, float]:
        """
        Query a single position and compute lens similarities.

        Returns dict with 'A', 'T', 'G', 'C' similarity scores derived from split banks.
        """
        # Try exact match first
        chunk_key = f"{chrom}:{pos}"

        if chunk_key in self.chunk_index:
            chunk_idx = self.chunk_index[chunk_key]
            offset = pos % self.N
        else:
            # Try finding the chunk that contains this position
            # Chunk keys are in format "chrX:start-end"
            chunk_idx = None
            for key, idx in self.chunk_index.items():
                if not key.startswith(f"{chrom}:"):
                    continue

                # Parse range
                _, pos_part = key.split(':')
                if '-' in pos_part:
                    start, end = map(int, pos_part.split('-'))
                    if start <= pos < end:
                        chunk_idx = idx
                        offset = (pos - start) % self.N
                        break

            if chunk_idx is None:
                return {'A': 0.0, 'T': 0.0, 'G': 0.0, 'C': 0.0}

        # Read all 6 banks (shape: [6, D])
        banks = self.bank_vectors[chunk_idx, :, :]

        # Extract individual banks
        A_bank = banks[0, :]  # Hydrophobic_A
        T_bank = banks[1, :]  # Hydrophobic_T
        G_bank = banks[2, :]  # MajorGroove_G
        C_bank = banks[3, :]  # MajorGroove_C
        pos_bank = banks[4, :]  # Hinge_pos
        neg_bank = banks[5, :]  # Hinge_neg

        # Get query vector
        query = self.pos_vectors[offset, :]

        # Compute similarities for each nucleotide
        # Positive query positions → match with binary 1s
        # Negative query positions → match with binary 0s

        def binary_similarity(bank: np.ndarray, query: np.ndarray) -> float:
            """
            Compute similarity between binary bank and bipolar query.

            Logic:
              - Where query is +1: reward binary 1s, penalize binary 0s
              - Where query is -1: reward binary 0s, penalize binary 1s

            This maps to: sum(bank * query) / D
            But since bank is {0,1} and query is {-1,+1}:
              - bank=1, query=+1 → +1 (match)
              - bank=0, query=+1 → 0 (neutral)
              - bank=1, query=-1 → -1 (anti-match)
              - bank=0, query=-1 → 0 (neutral)

            Actually: (2*bank - 1) * query gives proper bipolar correlation
            """
            bipolar_bank = 2 * bank.astype(np.float32) - 1  # {0,1} → {-1,+1}
            sim = np.dot(bipolar_bank, query) / self.D
            return float(sim)

        similarities = {
            'A': binary_similarity(A_bank, query),
            'T': binary_similarity(T_bank, query),
            'G': binary_similarity(G_bank, query),
            'C': binary_similarity(C_bank, query)
        }

        return similarities

    def batch_query_positions(self, test_positions: List[Tuple[str, int, str]]) -> List[Dict[str, float]]:
        """
        Batch query multiple positions at once (much faster than one-by-one).

        Args:
            test_positions: List of (chrom, pos, ground_truth) tuples

        Returns:
            List of similarity dicts, one per position
        """
        # Map positions to chunks
        chunk_indices = []
        offsets = []
        valid_positions = []

        for chrom, pos, gt in test_positions:
            # Try exact match first
            chunk_key = f"{chrom}:{pos}"
            if chunk_key in self.chunk_index:
                chunk_idx = self.chunk_index[chunk_key]
                offset = pos % self.N
                chunk_indices.append(chunk_idx)
                offsets.append(offset)
                valid_positions.append((chrom, pos, gt))
                continue

            # Try finding the range that contains this position
            found = False
            for key, idx in self.chunk_index.items():
                key_chrom, pos_part = key.split(':')
                if key_chrom != chrom:
                    continue
                if '-' in pos_part:
                    start, end = map(int, pos_part.split('-'))
                    if start <= pos < end:
                        chunk_idx = idx
                        offset = (pos - start) % self.N
                        chunk_indices.append(chunk_idx)
                        offsets.append(offset)
                        valid_positions.append((chrom, pos, gt))
                        found = True
                        break

        if not chunk_indices:
            return []

        # Load unique chunks
        unique_chunks = list(set(chunk_indices))
        chunk_data = {}
        for chunk_idx in unique_chunks:
            chunk_data[chunk_idx] = self.bank_vectors[chunk_idx, :, :]  # (6, 10240)

        logger.info(f"  ✓ Data loaded, computing similarities...")

        # Compute similarities for all positions
        results = []
        for i, (chunk_idx, offset) in enumerate(zip(chunk_indices, offsets)):
            banks = chunk_data[chunk_idx]
            A_bank = banks[0, :]
            T_bank = banks[1, :]
            G_bank = banks[2, :]
            C_bank = banks[3, :]

            query = self.pos_vectors[offset, :]

            def binary_similarity(bank: np.ndarray, query: np.ndarray) -> float:
                bipolar_bank = 2 * bank.astype(np.float32) - 1
                sim = np.dot(bipolar_bank, query) / self.D
                return float(sim)

            similarities = {
                'A': binary_similarity(A_bank, query),
                'T': binary_similarity(T_bank, query),
                'G': binary_similarity(G_bank, query),
                'C': binary_similarity(C_bank, query)
            }
            results.append(similarities)

        # Report spatial locality after queries complete
        locality_pct = 100.0 * len(unique_chunks) / len(chunk_indices) if chunk_indices else 0
        logger.info(f"  ✓ Spatial locality: {len(chunk_indices)} queries span {len(unique_chunks)} unique chunks ({locality_pct:.1f}% unique)")

        return results

    def close(self):
        """Close H5 file."""
        if self.h5_file:
            self.h5_file.close()


def predict_from_split_banks(similarities: Dict[str, float]) -> Tuple[str, float]:
    """
    Predict nucleotide from split bank similarities.

    Pure argmax - no thresholds. Each nucleotide has a dedicated bank,
    so we simply choose the one with the highest similarity.

    Args:
        similarities: Dict with 'A', 'T', 'G', 'C' similarity scores

    Returns:
        (predicted_nucleotide, confidence)
    """
    # Pure argmax: winner = max(similarities)
    best_nuc = max(similarities, key=similarities.get)

    # Confidence = absolute difference from second-best
    sorted_sims = sorted(similarities.values(), reverse=True)
    confidence = abs(sorted_sims[0] - sorted_sims[1]) if len(sorted_sims) > 1 else abs(sorted_sims[0])

    return best_nuc, confidence


def validate_split_binary(
    h5_path: Path,
    sample_size: int = 1000,
    seed: int = 42,
    output_dir: Path = None
):
    """
    Validate split binary architecture on random genomic positions.
    """
    if output_dir is None:
        output_dir = Path("genomevault/hdv_validation/hdc_experimentation/docs")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("SPLIT BINARY ARCHITECTURE VALIDATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"H5 file: {h5_path}")
    logger.info(f"Sample size: {sample_size:,}")
    logger.info(f"Random seed: {seed}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Load HDV system
    logger.info("Loading split binary HDV system...")
    hdv = SplitBinaryMultiLensHDV(h5_path, seed=seed)
    logger.info("")

    # Load GDiff for ground truth
    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    logger.info("Loading GDiff for ground truth...")
    gdiff, variant_index = load_gdiff(gdiff_path)
    logger.info(f"  ✓ Loaded {len(variant_index):,} variants")
    logger.info("")

    # Sample test positions using genome-wide sampling
    # Adapted from validation_utils.py sample_test_positions() for range-format chunk keys
    logger.info("Sampling test positions...")
    np.random.seed(seed)

    # Parse chunk keys to extract chunk start positions
    # Chunk key format: "chrX:start-end" → we use the start position for each chunk
    parsed_chunks = []
    for chunk_key in hdv.chunk_keys:
        try:
            chrom_part, pos_part = chunk_key.split(':')
            if '-' in pos_part:
                start_str, _ = pos_part.split('-')
                chunk_start = int(start_str)
            else:
                chunk_start = int(pos_part)
            parsed_chunks.append((chrom_part, chunk_start))
        except (ValueError, AttributeError):
            continue

    # Group chunks by chromosome for genome-wide sampling
    from collections import defaultdict
    chunks_by_chrom = defaultdict(list)
    for chrom, chunk_start in parsed_chunks:
        # Remove _consensus suffix if present for grouping
        chrom_clean = chrom.replace('_consensus', '')
        chunks_by_chrom[chrom_clean].append((chrom, chunk_start))

    # Calculate samples per chromosome (proportional to chunk count)
    total_chunks = len(parsed_chunks)
    chromosomes = sorted(chunks_by_chrom.keys(), key=lambda x: (
        x != 'chrX' and x != 'chrY',
        int(x.replace('chr', '').replace('X', '23').replace('Y', '24'))
    ))

    samples_per_chrom = {}
    for chrom in chromosomes:
        n_chunks = len(chunks_by_chrom[chrom])
        proportion = n_chunks / total_chunks
        samples_per_chrom[chrom] = int(sample_size * proportion)

    # Adjust for rounding errors
    total_allocated = sum(samples_per_chrom.values())
    if total_allocated < sample_size:
        diff = sample_size - total_allocated
        sorted_chroms = sorted(chromosomes, key=lambda c: len(chunks_by_chrom[c]), reverse=True)
        for i in range(diff):
            samples_per_chrom[sorted_chroms[i % len(sorted_chroms)]] += 1

    # Sample positions from each chromosome
    test_positions = []
    N = 2000  # Chunk size for position offset sampling

    for chrom in chromosomes:
        chrom_chunks = chunks_by_chrom[chrom]
        n_samples = samples_per_chrom[chrom]

        if n_samples == 0:
            continue

        # Sample random chunks from this chromosome
        sampled_chunks = [chrom_chunks[i] for i in np.random.choice(
            len(chrom_chunks),
            size=min(n_samples, len(chrom_chunks)),
            replace=False
        )]

        # For each sampled chunk, pick a random position within the chunk
        for chrom_key, chunk_start in sampled_chunks:
            pos = chunk_start + np.random.randint(0, N)
            test_positions.append((chrom_key, pos))

    logger.info(f"  ✓ Sampled {len(test_positions):,} positions (genome-wide stratified)")
    logger.info(f"    Chromosome distribution: {len(chromosomes)} chromosomes")
    logger.info("")

    # Open BAM for ground truth validation
    exp_bam_path = Path("data/experimental_strands/ERR3239334/alignment/k11_bams/experimental_vs_ref1.sorted.bam")
    exp_bam = pysam.AlignmentFile(str(exp_bam_path), 'rb') if exp_bam_path.exists() else None
    region_map = gdiff.get("region_guide_map", {})

    # Compute ground truth
    logger.info("Computing ground truth...")
    ground_truths = []
    for chrom, pos in test_positions:
        # Add _consensus suffix if not present (BAM uses chr1_consensus format)
        chrom_for_bam = chrom if chrom.endswith('_consensus') else f"{chrom}_consensus"

        gt, guide_idx, has_n = get_ground_truth(chrom_for_bam, pos, variant_index, exp_bam, region_map)

        if gt and gt in 'ATGC':  # Only valid nucleotides
            ground_truths.append({
                'chrom': chrom,  # Store original chrom for HDV query
                'chrom_bam': chrom_for_bam,  # Store BAM chrom for reference
                'pos': pos,
                'nucleotide': gt
            })

    logger.info(f"  ✓ {len(ground_truths):,} positions with valid ground truth")
    logger.info("")

    if exp_bam:
        exp_bam.close()

    # Query all positions using batch mode (much faster!)
    logger.info("Querying split binary banks (batch mode)...")

    # Prepare test positions for batch query
    test_pos_batch = [(gt['chrom'], gt['pos'], gt['nucleotide']) for gt in ground_truths]

    # Batch query all positions at once
    all_similarities = hdv.batch_query_positions(test_pos_batch)

    logger.info(f"  ✓ Batch query complete, processing predictions...")

    predictions = []
    for i, (gt, similarities) in enumerate(zip(ground_truths, all_similarities)):
        chrom = gt['chrom']
        pos = gt['pos']

        # Predict nucleotide
        pred, conf = predict_from_split_banks(similarities)

        predictions.append({
            'position': f"{chrom}:{pos}",
            'ground_truth': gt['nucleotide'],
            'predicted': pred,
            'confidence': conf,
            'correct': pred == gt['nucleotide'],
            'similarities': similarities
        })

    logger.info("")
    logger.info(f"✓ Queries completed")
    logger.info("")

    # Compute statistics
    logger.info("=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info("")

    correct_count = sum(1 for p in predictions if p['correct'])
    total = len(predictions)
    accuracy = correct_count / total if total > 0 else 0

    logger.info(f"Accuracy: {accuracy*100:.2f}% ({correct_count}/{total})")
    logger.info("")

    # Per-nucleotide performance
    logger.info("Per-Nucleotide Performance:")
    logger.info("")

    by_nucleotide = defaultdict(lambda: {'correct': 0, 'total': 0})
    for p in predictions:
        nuc = p['ground_truth']
        by_nucleotide[nuc]['total'] += 1
        if p['correct']:
            by_nucleotide[nuc]['correct'] += 1

    logger.info(f"{'Nucleotide':<12} {'Accuracy':<10} {'Count':<10}")
    logger.info("-" * 35)
    for nuc in 'ATGC':
        stats = by_nucleotide[nuc]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            logger.info(f"{nuc:<12} {acc*100:>9.2f}% {stats['total']:>9}")

    logger.info("")

    # Confusion matrix
    pred_list = [p['predicted'] for p in predictions]
    truth_list = [p['ground_truth'] for p in predictions]
    confusion = compute_confusion_matrix(pred_list, truth_list)

    logger.info("Confusion Matrix:")
    logger.info("")
    logger.info(f"{'':>10} {'A':>8} {'T':>8} {'G':>8} {'C':>8}")
    for true_nuc in 'ATGC':
        row = [str(confusion.get(f"{true_nuc}_as_{pred}", 0)) for pred in 'ATGC']
        logger.info(f"{true_nuc:>10} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8}")

    logger.info("")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'h5_path': str(h5_path),
            'sample_size': sample_size,
            'seed': seed,
            'total_predictions': total
        },
        'accuracy': accuracy,
        'per_nucleotide': {
            nuc: {
                'accuracy': by_nucleotide[nuc]['correct'] / by_nucleotide[nuc]['total']
                           if by_nucleotide[nuc]['total'] > 0 else 0,
                'count': by_nucleotide[nuc]['total']
            }
            for nuc in 'ATGC'
        },
        'confusion_matrix': confusion,
        'predictions': predictions[:100]  # Save first 100 for inspection
    }

    # Save JSON
    json_path = output_dir / "split_binary_validation_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ Results saved to: {json_path}")
    logger.info("")

    # Generate markdown report
    generate_split_binary_report(results, predictions, output_dir)

    # Close HDV
    hdv.close()

    return results


def generate_split_binary_report(results: Dict, predictions: List[Dict], output_dir: Path):
    """Generate markdown report for split binary validation."""

    report_path = output_dir / "split_binary_validation_report.md"

    with open(report_path, 'w') as f:
        f.write("# Split Binary Architecture Validation Report\n\n")
        f.write("**6-Bank Within-Lens Splitting Analysis**\n\n")
        f.write(f"**Generated:** {results['timestamp']}\n")
        f.write(f"**Sample Size:** {results['configuration']['total_predictions']:,} positions\n\n")

        f.write("## Architecture Overview\n\n")
        f.write("**Split Mapping:**\n")
        f.write("- **Hydrophobic:** A bank + T bank\n")
        f.write("- **MajorGroove:** G bank + C bank\n")
        f.write("- **Hinge:** pos bank + neg bank\n\n")

        f.write("**Total:** 6 binary banks, 10,240 dimensions each\n\n")

        f.write("## Performance Summary\n\n")
        f.write(f"**Overall Accuracy:** {results['accuracy']*100:.2f}%\n\n")

        f.write("### Per-Nucleotide Accuracy\n\n")
        f.write("| Nucleotide | Accuracy | Count |\n")
        f.write("|------------|----------|-------|\n")
        for nuc in 'ATGC':
            stats = results['per_nucleotide'][nuc]
            f.write(f"| **{nuc}** | {stats['accuracy']*100:.2f}% | {stats['count']:,} |\n")

        f.write("\n### Confusion Matrix\n\n")
        f.write("```\n")
        f.write(f"{'':>10} {'A':>8} {'T':>8} {'G':>8} {'C':>8}\n")
        cm = results['confusion_matrix']
        for true_nuc in 'ATGC':
            row = [str(cm.get(f"{true_nuc}_as_{pred}", 0)) for pred in 'ATGC']
            f.write(f"{true_nuc:>10} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8}\n")
        f.write("```\n\n")

        # Example predictions
        f.write("## Example Predictions\n\n")
        for i, pred in enumerate(predictions[:10]):
            status = "✓" if pred['correct'] else "✗"
            f.write(f"**{status} {pred['position']}**\n")
            f.write(f"- Ground truth: {pred['ground_truth']}\n")
            f.write(f"- Predicted: {pred['predicted']}\n")
            f.write(f"- Confidence: {pred['confidence']:.4f}\n")
            f.write(f"- Similarities: {pred['similarities']}\n\n")

        f.write("## File Information\n\n")
        f.write(f"**H5 File:** `{results['configuration']['h5_path']}`\n")
        f.write(f"**Random Seed:** {results['configuration']['seed']}\n")

    logger.info(f"✓ Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate split binary architecture'
    )
    parser.add_argument(
        '--h5-path',
        type=str,
        default='genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_6banks_split_binary.h5',
        help='Path to split binary H5 file'
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
        help='Random seed'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='genomevault/hdv_validation/hdc_experimentation/docs',
        help='Output directory for reports'
    )

    args = parser.parse_args()

    validate_split_binary(
        h5_path=Path(args.h5_path),
        sample_size=args.sample_size,
        seed=args.seed,
        output_dir=Path(args.output_dir)
    )


if __name__ == '__main__':
    main()
