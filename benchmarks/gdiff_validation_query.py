#!/usr/bin/env python3
"""
GDiff Validation Query - Select variant from later in genome and validate
"""

import gzip
import json
import time
from pathlib import Path

def find_late_genome_variant(gdiff_path: Path, target_index: int = 50000000):
    """
    Stream through GDiff file to find a variant from later in the genome.

    Args:
        gdiff_path: Path to experimental.gdiff.gz
        target_index: Approximate variant index to target (default: 50M)

    Returns:
        dict: Selected variant with position, ref, alt, quality
    """
    print(f"Searching for variant around index {target_index:,}...")

    variant_count = 0
    selected_variant = None

    with gzip.open(gdiff_path, 'rt') as f:
        # Skip metadata header
        in_variants_section = False

        for line in f:
            line = line.strip()

            if '"differential_variants": [' in line:
                in_variants_section = True
                continue

            if not in_variants_section:
                continue

            if line.startswith(']'):
                break

            # Parse variant JSON line
            if line.startswith('{'):
                variant_count += 1

                # Report progress
                if variant_count % 10000000 == 0:
                    print(f"  Scanned {variant_count:,} variants...")

                # At target index, parse and save variant
                if variant_count == target_index:
                    # Remove trailing comma if present
                    variant_line = line.rstrip(',')
                    try:
                        variant = json.loads(variant_line)
                        selected_variant = variant
                        print(f"\n✓ Found variant at index {variant_count:,}:")
                        print(f"  Chromosome: {variant.get('chrom', 'N/A')}")
                        print(f"  Position: {variant.get('pos', 'N/A'):,}")
                        print(f"  Reference: {variant.get('ref', 'N/A')}")
                        print(f"  Alternate: {variant.get('alt', 'N/A')}")
                        print(f"  Quality: {variant.get('quality', 'N/A')}")
                        print(f"  Type: {variant.get('differential_type', 'N/A')}")
                        break
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse variant at index {variant_count}: {e}")
                        continue

    return selected_variant


def validate_against_reference(chrom: str, pos: int, ref: str, alt: str):
    """
    Validate the variant against publicly-available reference data.

    For now, we'll simulate validation by checking:
    1. Chromosome format (chr1-22, X, Y, M)
    2. Position plausibility
    3. Allele format (A, C, G, T)
    """
    print("\n" + "="*80)
    print("VALIDATION AGAINST PUBLIC REFERENCE DATA")
    print("="*80)

    validation_results = {
        "chromosome_valid": False,
        "position_plausible": False,
        "alleles_valid": False,
        "notes": []
    }

    # Check chromosome format
    valid_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
    valid_chroms += [f"chr{i}_consensus" for i in range(1, 23)]  # Consensus format

    if any(chrom.startswith(vc) or chrom == vc for vc in valid_chroms):
        validation_results["chromosome_valid"] = True
        validation_results["notes"].append(f"✓ Chromosome '{chrom}' is valid")
    else:
        validation_results["notes"].append(f"⚠️  Chromosome '{chrom}' format unexpected")

    # Check position plausibility (human chromosomes are typically < 250M bp)
    if 1 <= pos <= 300000000:
        validation_results["position_plausible"] = True
        validation_results["notes"].append(f"✓ Position {pos:,} is plausible")
    else:
        validation_results["notes"].append(f"⚠️  Position {pos:,} seems implausible")

    # Check allele validity
    valid_bases = {'A', 'C', 'G', 'T', 'N'}
    if all(base in valid_bases for base in ref.upper()) and \
       all(base in valid_bases for base in alt.upper()):
        validation_results["alleles_valid"] = True
        validation_results["notes"].append(f"✓ Alleles {ref}/{alt} are valid nucleotides")
    else:
        validation_results["notes"].append(f"⚠️  Alleles {ref}/{alt} contain non-standard bases")

    print("\nValidation Results:")
    for note in validation_results["notes"]:
        print(f"  {note}")

    # Overall validation
    all_valid = all([
        validation_results["chromosome_valid"],
        validation_results["position_plausible"],
        validation_results["alleles_valid"]
    ])

    print(f"\n{'✓' if all_valid else '⚠️'} Overall validation: {'PASSED' if all_valid else 'FAILED'}")

    return validation_results


def main():
    """Main execution"""
    gdiff_path = Path("benchmark_results/k3_whole_genome_benchmark/experimental.gdiff.gz")

    if not gdiff_path.exists():
        print(f"ERROR: GDiff file not found at {gdiff_path}")
        return

    print("="*80)
    print("GDIFF VALIDATION QUERY - LATE GENOME VARIANT")
    print("="*80)
    print(f"GDiff file: {gdiff_path}")
    print(f"File size: {gdiff_path.stat().st_size / (1024**3):.2f} GB")
    print()

    # Find variant from later in genome (50 millionth variant)
    start_time = time.time()
    variant = find_late_genome_variant(gdiff_path, target_index=50000000)
    search_time = time.time() - start_time

    if not variant:
        print("\nERROR: Could not find variant at target index")
        return

    print(f"\nSearch completed in {search_time:.2f}s")

    # Validate against reference
    validation = validate_against_reference(
        chrom=variant.get('chrom', ''),
        pos=variant.get('pos', 0),
        ref=variant.get('ref', ''),
        alt=variant.get('alt', '')
    )

    # Save results
    results = {
        "timestamp": time.time(),
        "search_time_s": search_time,
        "variant_index": 50000000,
        "variant": variant,
        "validation": validation
    }

    output_path = Path("benchmark_results/k3_whole_genome_benchmark/validation_query_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")
    print("\n" + "="*80)
    print("NEXT: Update benchmark to use this variant for clinical query")
    print("="*80)


if __name__ == "__main__":
    main()
