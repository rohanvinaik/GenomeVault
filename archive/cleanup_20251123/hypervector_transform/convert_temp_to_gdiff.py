#!/usr/bin/env python3
"""
SAFE conversion of temp_variants.pkl.gz to final GDiff file.

This script ONLY reads the temp file and writes the GDiff.
It does NOT run any alignment or encoding - just file format conversion.
"""

import sys
import json
import gzip as gzip_module
import pickle
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime

# Add genomevault to path
sys.path.insert(0, str(Path(__file__).parent))

from genomevault.differential_encoding.gdiff.schema import (
    GDiffDocument,
    GDIFF_SCHEMA_VERSION,
    GDiffMetadata,
    AlignmentParams,
)

def main():
    # Input/output paths
    temp_file = Path("data/experimental_strands/ERR3239334/encoding/temp_variants.pkl.gz")
    gdiff_file = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")

    print(f"Converting {temp_file} to {gdiff_file}")
    print(f"Temp file size: {temp_file.stat().st_size / (1024**2):.2f} MB")

    # Create metadata (matches the pipeline configuration)
    metadata = GDiffMetadata(
        query_id="ERR3239334",
        reference_pool=[f"ref{i}" for i in range(1, 12)],  # ref1-ref11
        k_anonymity=12,  # k=12 (11 guides + 1 experimental)
        alignment_params=AlignmentParams(
            kmer=21,
            window=11,
            scoring="match=2,mismatch=-4,gap_open=-6",
            entropy_bits=512.0,
        ),
        genome_build="GRCh38",
        timestamp=datetime.utcnow().isoformat() + "Z",
        gdiff_version=GDIFF_SCHEMA_VERSION,
    )

    print("\nLoading variants from temp file...")
    total_variants = 0

    # Write GDiff file with streaming
    with gzip_module.open(gdiff_file, 'wt') as out_f:
        # Write header
        out_f.write('{\n')
        out_f.write(f'  "schema_version": "{GDIFF_SCHEMA_VERSION}",\n')
        out_f.write('  "metadata": ')
        json.dump(asdict(metadata), out_f)
        out_f.write(',\n')
        out_f.write('  "differential_variants": [\n')

        # Stream variants from temp file
        variant_idx = 0
        with gzip_module.open(temp_file, 'rb') as temp_f:
            while True:
                try:
                    variant_batch = pickle.load(temp_f)
                    for variant in variant_batch:
                        if variant_idx > 0:
                            out_f.write(',\n')
                        json.dump(variant, out_f, indent=2)
                        variant_idx += 1
                        total_variants += 1

                        # Progress indicator
                        if total_variants % 100000 == 0:
                            print(f"  Processed {total_variants:,} variants...")
                except EOFError:
                    break

        # Write footer
        out_f.write('\n  ]\n')
        out_f.write('}\n')

    print(f"\n✓ Conversion complete!")
    print(f"  Input: {temp_file} ({temp_file.stat().st_size / (1024**2):.2f} MB)")
    print(f"  Output: {gdiff_file} ({gdiff_file.stat().st_size / (1024**2):.2f} MB)")
    print(f"  Total variants: {total_variants:,}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
