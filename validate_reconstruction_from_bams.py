#!/usr/bin/env python3
"""
EMPIRICAL VALIDATION: Full Genome Reconstruction Using BAMs

Validate that non-variant positions can be inferred by showing:
1. GDiff contains variants (where experimental differs from guide)
2. Non-variant positions (not in GDiff) = experimental matches guide

We'll use the experimental BAMs which contain BOTH:
- Variant positions (differ from guide reference)
- Non-variant positions (match guide reference)

Then verify that GDiff correctly identifies ONLY the differing positions.
"""

import sys
import json
import gzip
import random
import pysam
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent))

def load_gdiff(gdiff_file):
    """Load GDiff"""
    print(f"\nLoading GDiff: {gdiff_file.name}")
    with gzip.open(gdiff_file, 'rt') as f:
        data = json.load(f)

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

    print(f"  {len(variants):,} variants")
    print(f"  {len(region_guide_map)} regions")

    return variant_lookup, region_guide_map

def find_region_for_position(chrom, pos, region_guide_map):
    """Find which guide was used for a genomic region"""
    for region, guide in region_guide_map.items():
        if region.startswith(chrom + ":"):
            parts = region.split(":")
            coords = parts[1].split("-")
            start = int(coords[0])
            end = int(coords[1])
            if start <= pos < end:
                return region, guide
    return None, None

def get_consensus_base_from_bam(bam_path, chrom, pos):
    """Get consensus base from BAM pileup"""
    try:
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for pileup_col in bam.pileup(chrom, pos-1, pos, truncate=True):
                if pileup_col.pos == pos - 1:
                    bases = [read.alignment.query_sequence[read.query_position]
                            for read in pileup_col.pileups
                            if not read.is_del and not read.is_refskip and read.query_position is not None]

                    if not bases:
                        return None, 0

                    consensus = Counter(bases).most_common(1)[0][0]
                    return consensus, len(bases)

        return None, 0
    except Exception as e:
        return None, 0

def main():
    gdiff_file = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    experimental_bams_dir = Path("data/experimental_strands/ERR3239334/alignment/k11_bams")
    guide_bams_dir = Path("/Volumes/1TBStorage/guide_strands")

    # CRITICAL: Must use _gdiff.bam files (guide reads aligned to guide FASTA coords)
    # NOT .sorted.bam files (guide reads aligned to consensus coords)

    print("="*80)
    print("EMPIRICAL FULL GENOME RECONSTRUCTION VALIDATION")
    print("="*80)
    print("\nStrategy:")
    print("1. For variant positions: GDiff contains them → use ALT")
    print("2. For non-variant positions: NOT in GDiff → experimental == guide")
    print("\nWe validate #2 by confirming positions NOT in GDiff truly match guide.")
    print("="*80)

    # Load GDiff
    variant_lookup, region_guide_map = load_gdiff(gdiff_file)

    # Sample regions for testing
    test_regions = random.sample(list(region_guide_map.keys()), min(10, len(region_guide_map)))

    print("\n" + "="*80)
    print("TEST: NON-VARIANT POSITIONS")
    print("Hypothesis: Positions NOT in GDiff match guide reference")
    print("="*80)

    total_tested = 0
    total_matched = 0
    total_mismatched = 0
    mismatches = []

    for region in test_regions:
        parts = region.split(":")
        chrom = parts[0]
        coords = parts[1].split("-")
        start = int(coords[0])
        end = int(coords[1])
        guide = region_guide_map[region]

        exp_bam = experimental_bams_dir / f"experimental_vs_ref{guide}.sorted.bam"
        guide_bam = guide_bams_dir / f"ref{guide}_gdiff.bam"  # FIXED: Use _gdiff.bam (guide FASTA coords)

        if not exp_bam.exists() or not guide_bam.exists():
            print(f"\n  Skipping {region} - BAMs not found")
            continue

        print(f"\n  Testing {region} (guide {guide})...")

        # Sample random positions in this region
        tested_in_region = 0
        for _ in range(200):  # Try 200 random positions per region
            pos = random.randint(start + 100000, end - 100000)

            # Skip if this is a variant
            if (chrom, pos) in variant_lookup:
                continue

            # This is a NON-variant position
            # Get base from experimental BAM
            exp_base, exp_cov = get_consensus_base_from_bam(exp_bam, chrom, pos)
            if exp_base is None or exp_cov < 10:  # Need at least 10× coverage
                continue

            # Get base from guide BAM
            guide_base, guide_cov = get_consensus_base_from_bam(guide_bam, chrom, pos)
            if guide_base is None or guide_cov < 10:
                continue

            # TEST: Do they match?
            total_tested += 1
            tested_in_region += 1

            if exp_base == guide_base:
                total_matched += 1
                if total_tested <= 10:  # Show first 10
                    print(f"    ✓ {chrom}:{pos} = {exp_base} (exp_cov={exp_cov}, guide_cov={guide_cov})")
            else:
                total_mismatched += 1
                mismatches.append({
                    "chrom": chrom,
                    "pos": pos,
                    "exp": exp_base,
                    "guide": guide_base,
                    "exp_cov": exp_cov,
                    "guide_cov": guide_cov,
                    "region": region
                })
                print(f"    ✗ MISMATCH: {chrom}:{pos} exp={exp_base}({exp_cov}×) vs guide={guide_base}({guide_cov}×)")

            if tested_in_region >= 50:  # Test up to 50 per region
                break

        print(f"    Tested {tested_in_region} non-variant positions in this region")

        if total_tested >= 100:  # Stop after 100 total
            break

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print(f"\nNon-variant positions tested: {total_tested}")
    print(f"  Matched guide: {total_matched} ({total_matched/total_tested*100:.1f}%)")
    print(f"  Mismatched: {total_mismatched} ({total_mismatched/total_tested*100:.1f}%)")

    if mismatches:
        print(f"\nMismatches (first 20):")
        for mm in mismatches[:20]:
            print(f"  {mm['chrom']}:{mm['pos']} exp={mm['exp']}({mm['exp_cov']}×) guide={mm['guide']}({mm['guide_cov']}×)")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    match_rate = total_matched / total_tested if total_tested > 0 else 0

    print(f"\nMatch rate: {match_rate*100:.1f}%")

    if match_rate >= 0.95:
        print("\n✅ VALIDATION PASSED")
        print("\nConclusion:")
        print("  Positions NOT in GDiff truly match the guide reference.")
        print("  Therefore, full genome reconstruction is possible:")
        print("    - Variant positions: Use ALT from GDiff")
        print("    - Non-variant positions: Fetch from guide reference")
        print("\n  Empirical confidence: Shannon-grade")
        return 0
    elif match_rate >= 0.90:
        print("\n⚠️  VALIDATION MOSTLY PASSED")
        print(f"\n  {match_rate*100:.1f}% match rate suggests differential encoding is working")
        print("  Small mismatch rate may be due to:")
        print("    - Sequencing errors")
        print("    - Low coverage positions")
        print("    - Alignment artifacts")
        return 0
    else:
        print("\n❌ VALIDATION FAILED")
        print(f"\n  {match_rate*100:.1f}% match rate too low")
        print("  This suggests systematic issues in the encoding")
        return 1

if __name__ == "__main__":
    random.seed(42)
    sys.exit(main())
