#!/usr/bin/env python3
"""
Find and query a variant from later in the genome (high position number)
"""

import gzip
import json
import time
from pathlib import Path

def load_and_query_late_variant(gdiff_path: Path):
    """
    Load GDiff file and find variant from late in genome (high position number)
    """
    print("Loading GDiff file (this may take a moment)...")
    start_time = time.time()

    with gzip.open(gdiff_path, 'rt') as f:
        gdiff_data = json.load(f)

    load_time = time.time() - start_time
    print(f"✓ Loaded in {load_time:.2f}s")

    variants = gdiff_data.get('differential_variants', [])
    total_variants = len(variants)
    print(f"✓ Total variants in file: {total_variants:,}")

    # Find variants with high position numbers (later in genome)
    # Target: position > 100,000,000 (100 Mbp)
    late_variants = []
    for i, var in enumerate(variants):
        pos = var.get('pos', 0)
        if pos > 100000000:  # 100 Mbp
            late_variants.append((i, var))
            if len(late_variants) >= 10:  # Collect 10 candidates
                break

    if not late_variants:
        # If no variants past 100M, take the highest position we have
        print("\nNo variants found past position 100M, selecting from highest available...")
        # Get last 100 variants and find one with highest position
        for i in range(len(variants) - 100, len(variants)):
            if i >= 0:
                late_variants.append((i, variants[i]))

    # Select the first late variant found
    selected_index, selected_variant = late_variants[0]

    print(f"\n{'='*80}")
    print(f"SELECTED VARIANT FROM LATER IN GENOME")
    print(f"{'='*80}")
    print(f"  Variant index: {selected_index:,} / {total_variants:,}")
    print(f"  Chromosome: {selected_variant.get('chrom', 'N/A')}")
    print(f"  Position: {selected_variant.get('pos', 0):,}")
    print(f"  Reference: {selected_variant.get('ref', 'N/A')}")
    print(f"  Alternate: {selected_variant.get('alt', 'N/A')}")
    print(f"  Quality: {selected_variant.get('quality_metrics', {}).get('read_depth', 'N/A')}")
    print(f"  Differential type: {selected_variant.get('differential_context', {}).get('diff_type', 'N/A')}")
    print(f"  Confidence: {selected_variant.get('differential_context', {}).get('confidence', 'N/A'):.4f}")

    return selected_variant, selected_index, total_variants


def validate_against_reference(variant: dict):
    """Validate variant structure and plausibility"""
    print(f"\n{'='*80}")
    print("VALIDATION")
    print(f"{'='*80}")

    chrom = variant.get('chrom', '')
    pos = variant.get('pos', 0)
    ref = variant.get('ref', '')
    alt = variant.get('alt', '')

    validations = []

    # Chromosome validation
    if 'chr' in chrom:
        validations.append(f"✓ Chromosome format valid: {chrom}")
    else:
        validations.append(f"⚠️  Unexpected chromosome format: {chrom}")

    # Position validation
    if 1 <= pos <= 300000000:
        validations.append(f"✓ Position plausible: {pos:,} bp")
    else:
        validations.append(f"⚠️  Position outside typical range: {pos:,} bp")

    # Allele validation
    valid_bases = {'A', 'C', 'G', 'T', 'N'}
    if all(b.upper() in valid_bases for b in ref) and all(b.upper() in valid_bases for b in alt):
        validations.append(f"✓ Alleles valid: {ref} → {alt}")
    else:
        validations.append(f"⚠️  Non-standard alleles: {ref} → {alt}")

    # Quality metrics validation
    quality_metrics = variant.get('quality_metrics', {})
    if quality_metrics:
        read_depth = quality_metrics.get('read_depth', 0)
        mapping_quality = quality_metrics.get('mapping_quality', 0)
        validations.append(f"✓ Quality metrics present (depth={read_depth}, mapq={mapping_quality})")
    else:
        validations.append("⚠️  No quality metrics found")

    # Differential context validation
    diff_context = variant.get('differential_context', {})
    if diff_context:
        diff_type = diff_context.get('diff_type', 'unknown')
        confidence = diff_context.get('confidence', 0)
        validations.append(f"✓ Differential context: {diff_type} (confidence={confidence:.2f})")
    else:
        validations.append("⚠️  No differential context found")

    print("\nValidation Results:")
    for v in validations:
        print(f"  {v}")

    all_valid = all('✓' in v for v in validations)
    print(f"\n{'✓' if all_valid else '⚠️'} Overall: {'VALID' if all_valid else 'WARNINGS PRESENT'}")

    return validations


def main():
    """Main execution"""
    gdiff_path = Path("benchmark_results/k3_whole_genome_benchmark/experimental.gdiff.gz")

    if not gdiff_path.exists():
        print(f"ERROR: GDiff file not found at {gdiff_path}")
        return

    print(f"{'='*80}")
    print("LATE-GENOME VARIANT QUERY AND VALIDATION")
    print(f"{'='*80}")
    print(f"GDiff file: {gdiff_path}")
    print(f"File size: {gdiff_path.stat().st_size / (1024**3):.2f} GB\n")

    # Load and select variant
    start = time.time()
    selected_variant, variant_index, total_variants = load_and_query_late_variant(gdiff_path)
    selection_time = time.time() - start

    # Validate
    validations = validate_against_reference(selected_variant)

    # Save results
    results = {
        "timestamp": time.time(),
        "selection_time_s": selection_time,
        "variant_index": variant_index,
        "total_variants": total_variants,
        "selected_variant": selected_variant,
        "validations": validations
    }

    output_path = Path("benchmark_results/k3_whole_genome_benchmark/late_variant_query_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ Results saved to: {output_path}")
    print(f"{'='*80}")

    return selected_variant


if __name__ == "__main__":
    main()
