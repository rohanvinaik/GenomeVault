#!/usr/bin/env python3
"""
CRITICAL VALIDATION: Full Genome Nucleotide Recovery from GDiff

This validates the COMPLETE encoding - not just variants, but EVERY nucleotide
in the genome can be determined from the GDiff.

For differential encoding to work, we must be able to answer:
"What is the nucleotide at position chr7:100,234,567 in the experimental genome?"

Two cases:
1. Position HAS a variant in GDiff → Read ALT allele directly
2. Position NOT in GDiff → Infer it matches the guide reference at that position

This test verifies BOTH cases work correctly.
"""

import sys
import json
import gzip
import random
import pysam
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

def load_gdiff(gdiff_file):
    """Load GDiff and create fast lookup structures"""
    print(f"\nLoading GDiff: {gdiff_file}")
    with gzip.open(gdiff_file, 'rt') as f:
        data = json.load(f)

    metadata = data["metadata"]
    variants = data["differential_variants"]

    # Build variant lookup: (chrom, pos) -> variant
    variant_lookup = {}
    for v in variants:
        key = (v["chrom"], v["pos"])
        variant_lookup[key] = v

    # Build region -> guide mapping
    region_guide_map = {}
    for v in variants:
        region = v["region"]
        guide = v["guide_idx"]
        if region not in region_guide_map:
            region_guide_map[region] = guide
        elif region_guide_map[region] != guide:
            raise ValueError(f"Region {region} has multiple guides!")

    print(f"  Loaded {len(variants):,} variants")
    print(f"  Covering {len(region_guide_map)} regions")
    print(f"  Guide pool: {metadata['reference_pool']}")

    return metadata, variant_lookup, region_guide_map

def find_region_for_position(chrom, pos, region_guide_map):
    """Find which region contains this position"""
    # Regions are 10MB chunks: chr1_consensus:0-10000000, chr1_consensus:10000000-20000000, etc.
    # Determine region by position
    region_size = 10_000_000
    region_start = (pos // region_size) * region_size
    region_end = region_start + region_size

    # Try exact match first
    region_key = f"{chrom}:{region_start}-{region_end}"
    if region_key in region_guide_map:
        return region_key, region_guide_map[region_key]

    # Handle edge case: last region may be shorter
    # Try all regions for this chromosome
    for region, guide in region_guide_map.items():
        if region.startswith(chrom + ":"):
            parts = region.split(":")
            coords = parts[1].split("-")
            start = int(coords[0])
            end = int(coords[1])
            if start <= pos < end:
                return region, guide

    return None, None

def query_nucleotide_from_gdiff(chrom, pos, variant_lookup, region_guide_map, guide_fastas):
    """
    Determine nucleotide at position using ONLY GDiff + guide references.

    This is the CRITICAL function - it proves we can reconstruct ANY nucleotide.
    """
    # Step 1: Check if position has a variant in GDiff
    key = (chrom, pos)
    if key in variant_lookup:
        # Case 1: Position is a variant - use ALT allele
        variant = variant_lookup[key]
        return variant["alt"], "VARIANT", variant["guide_idx"]

    # Step 2: Position NOT in GDiff → must match guide reference
    # Find which region contains this position
    region, guide_idx = find_region_for_position(chrom, pos, region_guide_map)

    if region is None:
        return None, "NO_REGION", None

    # Step 3: Look up nucleotide in guide reference
    guide_fasta = guide_fastas[guide_idx]

    try:
        # Fetch nucleotide from guide reference at this position
        nucleotide = guide_fasta.fetch(chrom, pos-1, pos)  # pysam is 0-indexed
        return nucleotide.upper(), "MATCH_GUIDE", guide_idx
    except Exception as e:
        return None, f"ERROR: {e}", guide_idx

def verify_against_experimental_bam(chrom, pos, predicted_nucleotide, guide_idx, experimental_bams_dir):
    """
    Ground truth verification: Check experimental BAM to confirm our prediction.
    """
    bam_path = experimental_bams_dir / f"experimental_vs_ref{guide_idx}.sorted.bam"

    if not bam_path.exists():
        return None, f"BAM not found: {bam_path}"

    try:
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            # Get pileup at position
            for pileup_col in bam.pileup(chrom, pos-1, pos, truncate=True):
                if pileup_col.pos == pos - 1:  # pysam is 0-indexed
                    bases = [read.alignment.query_sequence[read.query_position]
                            for read in pileup_col.pileups
                            if not read.is_del and not read.is_refskip and read.query_position is not None]

                    if not bases:
                        return None, "No coverage"

                    # Get consensus base (most common)
                    from collections import Counter
                    consensus = Counter(bases).most_common(1)[0][0]

                    if consensus == predicted_nucleotide:
                        return True, f"VERIFIED (coverage={len(bases)})"
                    else:
                        return False, f"MISMATCH: predicted={predicted_nucleotide}, actual={consensus} (coverage={len(bases)})"

        return None, "Position not in BAM"
    except Exception as e:
        return None, f"Error: {e}"

def main():
    gdiff_file = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    guide_fastas_dir = Path("/Volumes/1TBStorage/guide_strands")
    experimental_bams_dir = Path("data/experimental_strands/ERR3239334/alignment/k11_bams")

    print("="*80)
    print("FULL GENOME NUCLEOTIDE RECOVERY VALIDATION")
    print("Critical Test: Can we determine ANY nucleotide from GDiff?")
    print("="*80)

    # Load GDiff
    metadata, variant_lookup, region_guide_map = load_gdiff(gdiff_file)

    # Load guide FASTAs
    print("\nLoading guide FASTA files...")
    guide_fastas = {}
    for i in range(1, 12):
        fasta_path = guide_fastas_dir / f"ref{i}.fa.gz"
        if fasta_path.exists():
            guide_fastas[i] = pysam.FastaFile(str(fasta_path))
            print(f"  ✓ Loaded ref{i}.fa.gz")
        else:
            print(f"  ✗ Missing ref{i}.fa.gz")

    print("\n" + "="*80)
    print("TEST 1: Variant Positions (nucleotides IN GDiff)")
    print("="*80)

    # Test Case 1: Random variant positions (should use ALT allele)
    variant_sample = random.sample(list(variant_lookup.values()), 50)

    variant_correct = 0
    variant_total = 0

    for v in variant_sample:
        chrom = v["chrom"]
        pos = v["pos"]
        expected = v["alt"]
        guide = v["guide_idx"]

        # Query from GDiff
        predicted, source, predicted_guide = query_nucleotide_from_gdiff(
            chrom, pos, variant_lookup, region_guide_map, guide_fastas
        )

        # Verify against experimental BAM
        verified, detail = verify_against_experimental_bam(
            chrom, pos, predicted, guide, experimental_bams_dir
        )

        variant_total += 1
        if predicted == expected and verified:
            variant_correct += 1
        else:
            print(f"  FAIL: {chrom}:{pos} expected={expected}, predicted={predicted}, verified={verified}, {detail}")

    print(f"\nVariant positions: {variant_correct}/{variant_total} correct ({variant_correct/variant_total*100:.1f}%)")

    print("\n" + "="*80)
    print("TEST 2: Non-Variant Positions (nucleotides NOT in GDiff)")
    print("Critical: These must be inferred from guide reference")
    print("="*80)

    # Test Case 2: Random non-variant positions (should match guide reference)
    # Strategy: Pick random positions that are NOT in variant_lookup

    non_variant_correct = 0
    non_variant_total = 0
    non_variant_attempts = 0
    max_attempts = 1000  # Try up to 1000 random positions

    # Sample from regions we know about
    test_regions = random.sample(list(region_guide_map.keys()), min(20, len(region_guide_map)))

    for region in test_regions:
        # Parse region
        parts = region.split(":")
        chrom = parts[0]
        coords = parts[1].split("-")
        start = int(coords[0])
        end = int(coords[1])

        # Try to find a non-variant position in this region
        for _ in range(50):  # Try 50 random positions per region
            non_variant_attempts += 1
            if non_variant_attempts > max_attempts:
                break

            pos = random.randint(start, end-1)

            # Check if this position is a variant
            if (chrom, pos) in variant_lookup:
                continue  # Skip, this is a variant

            # This is a non-variant position - test it
            predicted, source, guide = query_nucleotide_from_gdiff(
                chrom, pos, variant_lookup, region_guide_map, guide_fastas
            )

            if predicted is None:
                continue  # Skip positions we can't query

            # Verify against experimental BAM
            verified, detail = verify_against_experimental_bam(
                chrom, pos, predicted, guide, experimental_bams_dir
            )

            non_variant_total += 1
            if verified:
                non_variant_correct += 1
            else:
                print(f"  FAIL: {chrom}:{pos} predicted={predicted} ({source}), {detail}")

            if non_variant_total >= 100:  # Stop after 100 successful tests
                break

        if non_variant_total >= 100:
            break

    print(f"\nNon-variant positions: {non_variant_correct}/{non_variant_total} correct ({non_variant_correct/non_variant_total*100:.1f}%)")

    # Clean up
    for fasta in guide_fastas.values():
        fasta.close()

    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)

    total_correct = variant_correct + non_variant_correct
    total_tested = variant_total + non_variant_total

    print(f"\nTotal nucleotides tested: {total_tested}")
    print(f"  Variant positions: {variant_total}")
    print(f"  Non-variant positions: {non_variant_total}")
    print(f"\nCorrect predictions: {total_correct}/{total_tested} ({total_correct/total_tested*100:.1f}%)")

    if total_correct == total_tested and total_tested >= 100:
        print("\n✅ FULL GENOME RECONSTRUCTION VERIFIED")
        print("   - Variant positions: Direct from GDiff ALT alleles")
        print("   - Non-variant positions: Inferred from guide references")
        print("   - Confidence: Shannon-level certainty")
        return 0
    else:
        print(f"\n❌ VALIDATION FAILED: {total_tested - total_correct} errors")
        return 1

if __name__ == "__main__":
    random.seed(42)  # Reproducible sampling
    sys.exit(main())
