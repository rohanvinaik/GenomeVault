#!/usr/bin/env python3
"""
Download new diverse samples not already in the project

Selects 2 samples from each ancestry pool that haven't been used yet.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Samples already in use (from validation and acquisition plans)
USED_SAMPLES = {
    'ERR3239276', 'ERR3239285', 'ERR3239334', 'ERR3239363', 'ERR3239372',
    'ERR3239398', 'ERR3239401', 'ERR3239421', 'ERR3239428', 'ERR3239445',
    'ERR3239454', 'ERR3239475', 'ERR3239489', 'ERR3239512', 'ERR3239534',
    'ERR3239567', 'ERR3239601', 'ERR3239608', 'ERR3239615', 'ERR3239622',
    'ERR3239629', 'ERR3239636', 'ERR3239643', 'ERR3239650', 'ERR3239657',
    'ERR3239664', 'ERR3239671', 'ERR3239678', 'ERR3239685', 'ERR3239701',
    'ERR3239708', 'ERR3239715', 'ERR3239722', 'ERR3239729', 'ERR3239736',
    'ERR3239743', 'ERR3239750', 'ERR3239757', 'ERR3239764', 'ERR3239771',
    'ERR3239778', 'ERR3239785', 'ERR3239801', 'ERR3239808', 'ERR3239815',
    'ERR3239822', 'ERR3239829', 'ERR3239836', 'ERR3239843', 'ERR3239850',
    'ERR3239857', 'ERR3239864', 'ERR3239871', 'ERR3239878', 'ERR3239885',
    'ERR4295224', 'ERR4295225', 'ERR4295226'
}

# New samples for each pool (carefully selected to avoid used ones)
NEW_SAMPLES = {
    'european': {
        'samples': ['ERR3239520', 'ERR3239548'],  # Different from used European samples
        'description': 'European ancestry (UK/Europe) - NEW'
    },
    'east_asian': {
        'samples': ['ERR3239590', 'ERR3239620'],  # Different from used East Asian samples
        'description': 'East Asian ancestry (China/Japan/Korea) - NEW'
    },
    'african': {
        'samples': ['ERR3239790', 'ERR3239812'],  # Different from used African samples
        'description': 'African ancestry (Sub-Saharan Africa) - NEW'
    },
    'south_asian': {
        'samples': ['ERR3239920', 'ERR3239945'],  # Different from used South Asian samples
        'description': 'South Asian ancestry (India/Pakistan/Bangladesh) - NEW'
    }
}

def main():
    parser = argparse.ArgumentParser(description='Download diverse genomic samples')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt')
    args = parser.parse_args()

    print("="*80)
    print("🧬 GenomeVault - Diverse Sample Download")
    print("="*80)
    print(f"\n📋 Downloading 2 new samples from each of 4 ancestry pools")
    print(f"   Total: 8 samples (~320 GB)")
    print(f"\n✅ All selected samples are NEW (not in existing validation data)")
    print("="*80)
    print()

    # Show selection
    print("📦 Selected Samples:")
    print()
    for pool, data in NEW_SAMPLES.items():
        print(f"  {pool.upper().replace('_', ' ')}:")
        for sample in data['samples']:
            status = "✅ NEW" if sample not in USED_SAMPLES else "⚠️  ALREADY USED"
            print(f"    - {sample} {status}")
        print()

    # Confirm
    if not args.yes:
        response = input("Start download? This will take several hours and use ~320 GB. (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    print("\n" + "="*80)
    print("Starting downloads...")
    print("="*80 + "\n")

    # Download each sample
    successful = []
    failed = []

    for pool, data in NEW_SAMPLES.items():
        for sample in data['samples']:
            print(f"\n{'='*80}")
            print(f"📥 Downloading: {sample} ({pool})")
            print(f"{'='*80}\n")

            try:
                result = subprocess.run(
                    [
                        'python', 'scripts/download_genomic_data_automated.py',
                        '--accession', sample,
                        '--output-dir', f'data/downloaded/fastq'
                    ],
                    check=True
                )
                successful.append(sample)
                print(f"\n✅ {sample} completed successfully")
            except subprocess.CalledProcessError as e:
                failed.append(sample)
                print(f"\n❌ {sample} failed: {e}")
                continue

    # Summary
    print("\n" + "="*80)
    print("📊 Download Summary")
    print("="*80)
    print(f"Total samples: 8")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print(f"\n✅ Successfully downloaded:")
        for sample in successful:
            print(f"   - {sample}")

    if failed:
        print(f"\n❌ Failed downloads:")
        for sample in failed:
            print(f"   - {sample}")
    else:
        print(f"\n✅ All downloads completed successfully!")

    print("="*80)

if __name__ == '__main__':
    main()
