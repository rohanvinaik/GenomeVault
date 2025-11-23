#!/usr/bin/env python3
"""
Debug Genomic Sequence Construction

Check what nucleotides we're actually encoding vs what we should be encoding.
"""

import gzip
import json
import bisect
from pathlib import Path
import pysam
from collections import Counter

# Test parameters
gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
guide_fasta_dir = Path("/Volumes/1TBStorage/guide_strands")
experimental_bam = Path("data/experimental_strands/ERR3239334/alignment/k11_bams/experimental_vs_ref2.sorted.bam")

# Load GDiff
print("Loading GDiff...")
with gzip.open(gdiff_path, 'rt') as f:
    gdiff = json.load(f)

region_guide_map = gdiff.get("region_guide_map", {})
print(f"✓ Loaded region→guide map: {len(region_guide_map)} regions")

# Index variants
print("Indexing variants...")
variants_by_chrom = {}
for variant in gdiff["differential_variants"]:
    chrom = variant["chrom"]
    if chrom not in variants_by_chrom:
        variants_by_chrom[chrom] = []
    variants_by_chrom[chrom].append(variant)

for chrom in variants_by_chrom:
    variants_by_chrom[chrom].sort(key=lambda v: v["pos"])

total_variants = sum(len(v) for v in variants_by_chrom.values())
print(f"✓ Indexed {total_variants:,} variants")

# Load guide FASTAs
print("Loading guide FASTAs...")
guide_fastas = {}
for i in range(1, 12):
    fasta_path = guide_fasta_dir / f"ref{i}.fa.gz"
    if fasta_path.exists():
        guide_fastas[f"ref{i}"] = pysam.FastaFile(str(fasta_path))
print(f"✓ Loaded {len(guide_fastas)} guide FASTAs")

# Open experimental BAM
bam = pysam.AlignmentFile(str(experimental_bam), 'rb')
print(f"✓ Opened experimental BAM")
print()

# Pick a test position with a variant
test_variant = gdiff["differential_variants"][1000]  # Arbitrary variant
chrom = test_variant["chrom"]
pos = test_variant["pos"]
alt = test_variant["alt"]

print("=" * 80)
print(f"TEST POSITION: {chrom}:{pos}")
print("=" * 80)
print()

# Get guide for this region
region_key = f"{chrom}:{pos // 10000}"
guide_id = region_guide_map.get(region_key, 'ref1')
print(f"Region key: {region_key}")
print(f"Assigned guide: {guide_id}")
print()

# Get nucleotide from GDiff (variant alt)
print(f"GDiff variant alt: {alt}")

# Get nucleotide from guide FASTA
if guide_id in guide_fastas:
    try:
        guide_nuc = guide_fastas[guide_id].fetch(chrom, pos, pos + 1).upper()
        print(f"Guide FASTA nucleotide: {guide_nuc}")
    except Exception as e:
        print(f"Error fetching from guide FASTA: {e}")
        guide_nuc = 'N'
else:
    print(f"Guide {guide_id} not found")
    guide_nuc = 'N'

# Get ground truth from experimental BAM
print()
print("Getting ground truth from experimental BAM...")
bases = []
try:
    for pileup_col in bam.pileup(chrom, pos, pos + 1, stepper='nofilter', truncate=True):
        if pileup_col.pos == pos:
            for pileup_read in pileup_col.pileups:
                if not pileup_read.is_del and not pileup_read.is_refskip:
                    base = pileup_read.alignment.query_sequence[pileup_read.query_position]
                    bases.append(base.upper())
            break
except Exception as e:
    print(f"Error: {e}")

if bases:
    ground_truth = Counter(bases).most_common(1)[0][0]
    print(f"Experimental BAM ground truth: {ground_truth} (from {len(bases)} reads)")
    print(f"All bases: {Counter(bases)}")
else:
    print("No coverage at this position")
    ground_truth = 'N'

print()
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"GDiff alt:           {alt}")
print(f"Guide FASTA:         {guide_nuc}")
print(f"Experimental BAM:    {ground_truth}")
print()

# Expected behavior:
# - GDiff alt should match experimental BAM (this is the variant)
# - Guide FASTA might be different (it's the reference)

# What we SHOULD encode:
print("What we SHOULD encode:")
print(f"  If this position has a variant: use alt nucleotide ({alt})")
print(f"  If this position has no variant: use guide FASTA nucleotide ({guide_nuc})")
print()

# Check if there's actually a variant at this position
chrom_variants = variants_by_chrom.get(chrom, [])
idx = bisect.bisect_left(chrom_variants, pos, key=lambda v: v["pos"])
has_variant = (idx < len(chrom_variants) and chrom_variants[idx]["pos"] == pos)

if has_variant:
    variant_at_pos = chrom_variants[idx]
    print(f"✓ Variant found at position: ref={variant_at_pos.get('ref', 'N/A')}, alt={variant_at_pos['alt']}")
    print(f"  Should encode: {variant_at_pos['alt']}")
    print(f"  Expected match: Experimental BAM")
else:
    print(f"✗ No variant at this position (unexpected - we selected from variant list)")

print()

# Cleanup
bam.close()
for fasta in guide_fastas.values():
    fasta.close()
