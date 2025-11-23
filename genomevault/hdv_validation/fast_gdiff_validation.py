#!/usr/bin/env python3
"""
Fast GDiff-only validation for signature discovery.

Directly samples from GDiff variants, no BAM lookups needed.
Should complete in seconds, not hours.
"""

import h5py
import gzip
import json
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def load_gdiff_variants(gdiff_path: Path) -> Dict[str, str]:
    """Load GDiff variants into position -> nucleotide dict."""
    logger.info(f"Loading GDiff from: {gdiff_path}")

    with gzip.open(gdiff_path, 'rt') as f:
        gdiff = json.load(f)

    variants = {}
    for v in gdiff["differential_variants"]:
        chrom = v['chrom']

        # Add _consensus suffix if needed
        if not chrom.endswith('_consensus'):
            chrom = chrom + '_consensus'

        pos = v['pos']
        key = f"{chrom}:{pos}"

        # Use ALT as ground truth (experimental nucleotide)
        variants[key] = v['alt']

    logger.info(f"  ✓ Loaded {len(variants):,} variants")
    return variants


def sample_genome_wide_variants(variants: Dict[str, str], sample_size: int, seed: int = 42) -> List[Tuple[str, int, str]]:
    """Sample variants genome-wide (stratified by chromosome)."""
    np.random.seed(seed)

    # Group by chromosome
    by_chrom = defaultdict(list)
    for key, nt in variants.items():
        chrom, pos_str = key.split(':')
        pos = int(pos_str)
        chrom_clean = chrom.replace('_consensus', '')
        by_chrom[chrom_clean].append((chrom, pos, nt))

    # Calculate proportional samples per chromosome
    total_variants = len(variants)
    chromosomes = sorted(by_chrom.keys(), key=lambda x: (
        x != 'chrX' and x != 'chrY',
        int(x.replace('chr', '').replace('X', '23').replace('Y', '24'))
    ))

    samples_per_chrom = {}
    for chrom in chromosomes:
        n_vars = len(by_chrom[chrom])
        proportion = n_vars / total_variants
        samples_per_chrom[chrom] = int(sample_size * proportion)

    # Adjust for rounding
    total_allocated = sum(samples_per_chrom.values())
    if total_allocated < sample_size:
        diff = sample_size - total_allocated
        sorted_chroms = sorted(chromosomes, key=lambda c: len(by_chrom[c]), reverse=True)
        for i in range(diff):
            samples_per_chrom[sorted_chroms[i % len(sorted_chroms)]] += 1

    # Sample from each chromosome
    sampled = []
    logger.info("Sampling strategy:")
    for chrom in chromosomes:
        chrom_vars = by_chrom[chrom]
        n_samples = samples_per_chrom[chrom]

        if n_samples == 0:
            continue

        indices = np.random.choice(len(chrom_vars), size=min(n_samples, len(chrom_vars)), replace=False)
        chrom_sampled = [chrom_vars[i] for i in indices]
        sampled.extend(chrom_sampled)

        logger.info(f"  {chrom:6s}: {len(chrom_sampled):>6,} variants sampled")

    logger.info(f"\n✓ Total sampled: {len(sampled):,} variants")
    return sampled


def validate_hdv_on_variants(
    h5_path: Path,
    sampled_variants: List[Tuple[str, int, str]],
    quantization: str
) -> Dict:
    """Validate HDV predictions on sampled variants."""

    logger.info(f"\nValidating {quantization} on {len(sampled_variants):,} variants...")

    # Load HDV data
    with h5py.File(h5_path, 'r') as f:
        chunk_keys_bytes = f['chunk_keys'][:]
        chunk_keys = [k.decode('utf-8') for k in chunk_keys_bytes]

        # Build chunk index
        chunk_index = {}
        for i, key in enumerate(chunk_keys):
            chunk_index[key] = i

        # Voting thresholds (empirically optimized)
        if quantization == 'float32':
            thresholds = {'AT': 0.05, 'GC': 0.00, 'PuPy': 0.20, 'AmKe': 0.20, 'StWk': 0.20}
        elif quantization == 'int8':
            thresholds = {'AT': 0.05, 'GC': 0.00, 'PuPy': 0.20, 'AmKe': 0.20, 'StWk': 0.20}
        elif quantization == 'int4':
            thresholds = {'AT': 0.05, 'GC': 0.00, 'PuPy': 0.20, 'AmKe': 0.20, 'StWk': 0.20}
        else:  # binary
            thresholds = {'AT': 0.05, 'GC': 0.00, 'PuPy': 0.20, 'AmKe': 0.20, 'StWk': 0.20}

        N = 2000
        results = []
        errors = []

        for chrom, pos, ground_truth in sampled_variants:
            # Find chunk
            chunk_start = (pos // N) * N
            chunk_key = f"{chrom}:{chunk_start}"

            if chunk_key not in chunk_index:
                continue

            chunk_idx = chunk_index[chunk_key]
            offset = pos - chunk_start

            # Get lens similarities
            lens_sims = {}
            for lens in ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']:
                data = f[lens][chunk_idx, offset]
                lens_sims[lens] = float(data)

            # Vote
            votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}

            for lens, sim in lens_sims.items():
                if abs(sim) < thresholds[lens]:
                    continue

                if lens == 'AT':
                    votes['A' if sim > 0 else 'T'] += 1
                elif lens == 'GC':
                    votes['G' if sim > 0 else 'C'] += 1
                elif lens == 'PuPy':
                    votes['A' if sim > 0 else 'C'] += 1
                elif lens == 'AmKe':
                    votes['A' if sim > 0 else 'G'] += 1
                elif lens == 'StWk':
                    votes['G' if sim > 0 else 'A'] += 1

            predicted = max(votes, key=votes.get)
            confidence = votes[predicted] / 5.0
            correct = predicted == ground_truth

            result = {
                'position': f"{chrom}:{pos}",
                'ground_truth': ground_truth,
                'predicted': predicted,
                'correct': correct,
                'confidence': confidence,
                'lens_results': lens_sims
            }

            results.append(result)

            if not correct:
                errors.append(result)

    accuracy = sum(1 for r in results if r['correct']) / len(results)

    logger.info(f"  Accuracy: {accuracy*100:.2f}% ({sum(1 for r in results if r['correct'])}/{len(results)})")
    logger.info(f"  Errors: {len(errors)}")

    return {
        'quantization': quantization,
        'total': len(results),
        'correct': sum(1 for r in results if r['correct']),
        'accuracy': accuracy,
        'errors': errors,
        'all_results': results
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fast GDiff-only validation')
    parser.add_argument('--quantizations', nargs='+', default=['float32', 'int8', 'int4', 'binary'])
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='HDV_VALIDATION_PACKAGE/architecture_testing/genome_wide_gdiff')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("FAST GDIFF-ONLY VALIDATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Quantizations: {', '.join(args.quantizations)}")
    logger.info(f"Sample size: {args.samples:,}")
    logger.info(f"Seed: {args.seed}")
    logger.info("")

    # Load GDiff
    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    variants = load_gdiff_variants(gdiff_path)

    # Sample genome-wide
    logger.info("")
    sampled = sample_genome_wide_variants(variants, args.samples, seed=args.seed)

    # Validate each quantization
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for quant in args.quantizations:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"QUANTIZATION: {quant.upper()}")
        logger.info("=" * 80)

        if quant == 'float32':
            h5_path = Path("data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome_5lenses_3d.h5")
        elif quant == 'int8':
            h5_path = Path("data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome_5lenses_3d_int8.h5")
        elif quant == 'int4':
            h5_path = Path("data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome_5lenses_3d_int4.h5")
        else:  # binary
            h5_path = Path("data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome_5lenses_3d_binary.h5")

        results = validate_hdv_on_variants(h5_path, sampled, quant)

        # Save results
        out_file = output_dir / f"{quant}_predictions_detailed.json"
        with open(out_file, 'w') as f:
            json.dump(results['all_results'], f, indent=2)

        logger.info(f"\n✓ Saved to: {out_file}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
