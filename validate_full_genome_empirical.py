#!/usr/bin/env python3
"""
EMPIRICAL VALIDATION: Full Genome Nucleotide Recovery

Actually test non-variant positions by decompressing guide FASTAs on-the-fly.
No bgzip required - we'll use samtools faidx for random access.
"""

import sys
import json
import gzip
import random
import subprocess
import pysam
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

def load_gdiff(gdiff_file):
    """Load GDiff and create lookup structures"""
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

    print(f"  Loaded {len(variants):,} variants")
    print(f"  Covering {len(region_guide_map)} regions")

    return metadata, variant_lookup, region_guide_map

def find_region_for_position(chrom, pos, region_guide_map):
    """Find which region contains this position"""
    region_size = 10_000_000
    region_start = (pos // region_size) * region_size
    region_end = region_start + region_size

    # Try exact match first
    region_key = f"{chrom}:{region_start}-{region_end}"
    if region_key in region_guide_map:
        return region_key, region_guide_map[region_key]

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

def fetch_nucleotide_from_guide(guide_idx, chrom, pos, guide_fastas_dir):
    """
    Fetch nucleotide from guide FASTA using samtools faidx.
    This works with gzip-compressed files.
    """
    fasta_path = guide_fastas_dir / f"ref{guide_idx}.fa.gz"

    # Use samtools faidx to extract single base
    # Format: samtools faidx file.fa.gz chr:pos-pos
    cmd = ["samtools", "faidx", str(fasta_path), f"{chrom}:{pos}-{pos}"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            # First line is header, second line is sequence
            nucleotide = lines[1].strip().upper()
            return nucleotide
        return None
    except subprocess.CalledProcessError as e:
        return None

def verify_against_experimental_bam(chrom, pos, predicted_nucleotide, guide_idx, experimental_bams_dir):
    """Ground truth verification"""
    bam_path = experimental_bams_dir / f"experimental_vs_ref{guide_idx}.sorted.bam"

    if not bam_path.exists():
        return None, f"BAM not found"

    try:
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for pileup_col in bam.pileup(chrom, pos-1, pos, truncate=True):
                if pileup_col.pos == pos - 1:
                    bases = [read.alignment.query_sequence[read.query_position]
                            for read in pileup_col.pileups
                            if not read.is_del and not read.is_refskip and read.query_position is not None]

                    if not bases:
                        return None, "No coverage"

                    from collections import Counter
                    consensus = Counter(bases).most_common(1)[0][0]
                    coverage = len(bases)

                    if consensus == predicted_nucleotide:
                        return True, f"coverage={coverage}"
                    else:
                        return False, f"predicted={predicted_nucleotide}, actual={consensus}, coverage={coverage}"

        return None, "Position not in BAM"
    except Exception as e:
        return None, f"Error: {e}"

def main():
    gdiff_file = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    guide_fastas_dir = Path("/Volumes/1TBStorage/guide_strands")
    experimental_bams_dir = Path("data/experimental_strands/ERR3239334/alignment/k11_bams")

    print("="*80)
    print("EMPIRICAL FULL GENOME RECONSTRUCTION VALIDATION")
    print("Testing: Can we determine ANY nucleotide from GDiff + guides?")
    print("="*80)

    # Load GDiff
    metadata, variant_lookup, region_guide_map = load_gdiff(gdiff_file)

    # Check samtools is available
    try:
        subprocess.run(["samtools", "--version"], capture_output=True, check=True)
        print("\n✓ samtools available")
    except:
        print("\n✗ samtools not found - install with: brew install samtools")
        return 1

    print("\n" + "="*80)
    print("TEST 1: VARIANT POSITIONS (nucleotides IN GDiff)")
    print("="*80)

    variant_sample = random.sample(list(variant_lookup.values()), 50)
    variant_correct = 0
    variant_total = 0

    for v in variant_sample:
        chrom = v["chrom"]
        pos = v["pos"]
        expected_alt = v["alt"]
        guide = v["guide_idx"]

        # The prediction is simple: use ALT from GDiff
        predicted = expected_alt

        # Verify against experimental BAM
        verified, detail = verify_against_experimental_bam(
            chrom, pos, predicted, guide, experimental_bams_dir
        )

        variant_total += 1
        if verified:
            variant_correct += 1
            if variant_total <= 5:  # Show first 5
                print(f"  ✓ {chrom}:{pos} {v['ref']}>{expected_alt} (guide {guide}, {detail})")
        else:
            print(f"  ✗ FAIL: {chrom}:{pos} {v['ref']}>{expected_alt} - {detail}")

    print(f"\n  Result: {variant_correct}/{variant_total} ({variant_correct/variant_total*100:.1f}%)")

    print("\n" + "="*80)
    print("TEST 2: NON-VARIANT POSITIONS (nucleotides NOT in GDiff)")
    print("CRITICAL: Must infer from guide reference")
    print("="*80)

    non_variant_correct = 0
    non_variant_total = 0
    non_variant_failed = []

    # Sample regions
    test_regions = random.sample(list(region_guide_map.keys()), min(20, len(region_guide_map)))

    for region in test_regions:
        parts = region.split(":")
        chrom = parts[0]
        coords = parts[1].split("-")
        start = int(coords[0])
        end = int(coords[1])
        guide = region_guide_map[region]

        # Try to find non-variant positions in this region
        attempts = 0
        while non_variant_total < 100 and attempts < 200:
            attempts += 1
            pos = random.randint(start + 100000, end - 100000)  # Stay away from edges

            # Skip if this is a variant
            if (chrom, pos) in variant_lookup:
                continue

            # This is a non-variant position - fetch from guide FASTA
            predicted = fetch_nucleotide_from_guide(guide, chrom, pos, guide_fastas_dir)

            if predicted is None:
                continue

            # Verify against experimental BAM
            verified, detail = verify_against_experimental_bam(
                chrom, pos, predicted, guide, experimental_bams_dir
            )

            non_variant_total += 1

            if verified:
                non_variant_correct += 1
                if non_variant_total <= 5:  # Show first 5
                    print(f"  ✓ {chrom}:{pos} = {predicted} (guide {guide}, {detail})")
            elif verified is False:
                non_variant_failed.append({
                    "chrom": chrom,
                    "pos": pos,
                    "predicted": predicted,
                    "guide": guide,
                    "detail": detail
                })
                print(f"  ✗ FAIL: {chrom}:{pos} - {detail}")

            if non_variant_total >= 100:
                break

        if non_variant_total >= 100:
            break

    print(f"\n  Result: {non_variant_correct}/{non_variant_total} ({non_variant_correct/non_variant_total*100:.1f}%)")

    if non_variant_failed:
        print(f"\n  Failures: {len(non_variant_failed)}")
        for fail in non_variant_failed[:10]:
            print(f"    {fail['chrom']}:{fail['pos']} - {fail['detail']}")

    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)

    total_correct = variant_correct + non_variant_correct
    total_tested = variant_total + non_variant_total

    print(f"\nTotal positions tested: {total_tested}")
    print(f"  Variant positions: {variant_total} ({variant_correct} correct)")
    print(f"  Non-variant positions: {non_variant_total} ({non_variant_correct} correct)")
    print(f"\nAccuracy: {total_correct}/{total_tested} ({total_correct/total_tested*100:.1f}%)")

    # Success criteria
    variant_accuracy = variant_correct / variant_total if variant_total > 0 else 0
    non_variant_accuracy = non_variant_correct / non_variant_total if non_variant_total > 0 else 0

    if variant_accuracy >= 0.95 and non_variant_accuracy >= 0.95 and total_tested >= 100:
        print("\n✅ FULL GENOME RECONSTRUCTION VALIDATED")
        print("   - Variant positions: Direct from GDiff (empirically verified)")
        print("   - Non-variant positions: Inferred from guides (empirically verified)")
        print("   - Confidence: Shannon-level certainty (empirical data)")
        return 0
    else:
        print(f"\n⚠️  VALIDATION INCOMPLETE")
        print(f"   - Variant accuracy: {variant_accuracy*100:.1f}% (need ≥95%)")
        print(f"   - Non-variant accuracy: {non_variant_accuracy*100:.1f}% (need ≥95%)")
        return 1

if __name__ == "__main__":
    random.seed(42)
    sys.exit(main())
