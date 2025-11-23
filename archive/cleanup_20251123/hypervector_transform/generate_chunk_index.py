#!/usr/bin/env python3
"""
Generate chunk_index.parquet from encoded_genome.h5
"""
import h5py
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_chunk_index(encoded_genome_path: Path, output_path: Path):
    """Generate chunk index parquet from encoded genome H5 file."""
    logger.info(f"Reading encoded genome: {encoded_genome_path}")

    with h5py.File(encoded_genome_path, 'r') as f:
        # Check if chunk_index dataset exists
        if 'chunk_index' in f:
            # Load from H5 dataset
            logger.info("Found chunk_index dataset in H5 file")
            chunk_data = {
                'chunk_id': f['chunk_index']['chunk_id'][:],
                'chrom': [s.decode() if isinstance(s, bytes) else s for s in f['chunk_index']['chrom'][:]],
                'start': f['chunk_index']['start'][:],
                'end': f['chunk_index']['end'][:]
            }
        else:
            # Generate from metadata if available
            logger.info("No chunk_index dataset - generating from metadata")
            if 'metadata' not in f:
                raise ValueError("No metadata found in H5 file - cannot generate chunk index")

            metadata_group = f['metadata']
            chunk_data = {
                'chunk_id': [],
                'chrom': [],
                'start': [],
                'end': []
            }

            # Iterate through chromosome groups
            for chrom in metadata_group.keys():
                chrom_group = metadata_group[chrom]
                for chunk_id_str in chrom_group.keys():
                    chunk_meta = chrom_group[chunk_id_str]
                    chunk_data['chunk_id'].append(int(chunk_id_str))
                    chunk_data['chrom'].append(chrom)
                    chunk_data['start'].append(chunk_meta.attrs['start'])
                    chunk_data['end'].append(chunk_meta.attrs['end'])

    # Create DataFrame
    df = pd.DataFrame(chunk_data)
    df = df.sort_values('chunk_id').reset_index(drop=True)

    logger.info(f"Generated index with {len(df):,} chunks")
    logger.info(f"Chromosomes: {sorted(df['chrom'].unique())}")

    # Save as parquet
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved chunk index to: {output_path}")

    return df


if __name__ == '__main__':
    encoded_genome = Path('data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome.h5')
    output = Path('data/experimental_strands/ERR3239334/hdv_encoding/chunk_index.parquet')

    df = generate_chunk_index(encoded_genome, output)

    print(f"\n✓ Generated chunk index:")
    print(f"  Total chunks: {len(df):,}")
    print(f"  Chromosomes: {len(df['chrom'].unique())}")
    print(f"  Output: {output}")
