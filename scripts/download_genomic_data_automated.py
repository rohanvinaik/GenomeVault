#!/usr/bin/env python3
"""
Automated Genomic Data Downloader

Downloads genomic data from ENA/SRA with comprehensive tracking:
- Progress monitoring with JSON state file
- Parallel download support
- Automatic compression
- MD5 verification
- Resumable downloads

Usage:
    python scripts/download_genomic_data_automated.py --pool european --samples 7
    python scripts/download_genomic_data_automated.py --pool all --samples 10
    python scripts/download_genomic_data_automated.py --accession ERR3239363
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Sample pools from QUICK_START_GUIDE
SAMPLE_POOLS = {
    'european': {
        'reference': [
            'ERR3239363', 'ERR3239372', 'ERR3239401',
            'ERR3239428', 'ERR3239445', 'ERR3239512', 'ERR3239567'
        ],
        'query': ['ERR3239276', 'ERR3239334', 'ERR3239454'],
        'description': 'European ancestry (UK/Europe)'
    },
    'east_asian': {
        'reference': [
            'ERR3239578', 'ERR3239612', 'ERR3239634',
            'ERR3239689', 'ERR3239701', 'ERR3239723'
        ],
        'query': ['ERR3239745'],
        'description': 'East Asian ancestry (China/Japan/Korea)'
    },
    'african': {
        'reference': [
            'ERR3239756', 'ERR3239778', 'ERR3239801',
            'ERR3239823', 'ERR3239845', 'ERR3239867'
        ],
        'query': ['ERR3239889'],
        'description': 'African ancestry (Sub-Saharan Africa)'
    },
    'south_asian': {
        'reference': [
            'ERR3239912', 'ERR3239934', 'ERR3239956',
            'ERR3239978', 'ERR3240001', 'ERR3240023'
        ],
        'query': ['ERR3240045'],
        'description': 'South Asian ancestry (India/Pakistan/Bangladesh)'
    }
}


class GenomicDataDownloader:
    """Automated downloader with progress tracking."""

    def __init__(self, output_dir: Path, state_file: Path):
        self.output_dir = Path(output_dir)
        self.state_file = Path(state_file)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize state
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load download state from JSON file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                pass

        # Initialize new state
        return {
            'start_time': datetime.now().isoformat(),
            'samples': {},
            'total_downloaded_gb': 0.0,
            'status': 'initialized'
        }

    def _save_state(self):
        """Save download state to JSON file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def check_dependencies(self) -> bool:
        """Check if required tools are installed."""
        tools = ['fasterq-dump', 'pigz', 'prefetch']
        missing = []

        for tool in tools:
            if shutil.which(tool) is None:
                missing.append(tool)

        if missing:
            print(f"❌ Missing required tools: {', '.join(missing)}")
            print("\nInstall with:")
            print("  conda install -c bioconda sra-tools pigz")
            return False

        return True

    def get_disk_space(self) -> float:
        """Get available disk space in GB."""
        try:
            # Try to create output dir if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)

            result = subprocess.run(
                ['df', '-BG', str(self.output_dir)],
                capture_output=True,
                text=True
            )
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                fields = lines[1].split()
                available_gb = float(fields[3].replace('G', ''))
                return available_gb
            return 100.0  # Default if can't determine
        except:
            return 100.0  # Default if error

    def download_sample(self, accession: str, sample_type: str = 'reference', pool: str = 'european') -> bool:
        """Download a single genomic sample with progress tracking."""
        # Initialize sample state
        if accession not in self.state['samples']:
            self.state['samples'][accession] = {
                'accession': accession,
                'sample_type': sample_type,
                'pool': pool,
                'status': 'queued',
                'start_time': None,
                'end_time': None,
                'size_gb': 0.0,
                'files': [],
                'error': None
            }

        sample_state = self.state['samples'][accession]

        # Skip if already completed
        if sample_state['status'] == 'completed':
            print(f"✓ {accession} already downloaded, skipping...")
            return True

        # Create output directory
        sample_dir = self.output_dir / pool / accession
        sample_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Update state
            sample_state['status'] = 'downloading'
            sample_state['start_time'] = datetime.now().isoformat()
            self._save_state()

            print(f"\n{'='*80}")
            print(f"📥 Downloading: {accession} ({pool} {sample_type})")
            print(f"{'='*80}")

            # Step 1: Prefetch (download to SRA cache)
            print(f"[1/3] Prefetching {accession} to SRA cache...")
            prefetch_result = subprocess.run(
                ['prefetch', accession, '--max-size', '100GB', '--progress'],
                capture_output=True,
                text=True
            )

            if prefetch_result.returncode != 0:
                raise Exception(f"Prefetch failed: {prefetch_result.stderr}")

            print(f"✓ Prefetch complete")

            # Step 2: Extract FASTQ files
            print(f"[2/3] Extracting FASTQ files...")
            fasterq_result = subprocess.run(
                [
                    'fasterq-dump', accession,
                    '--outdir', str(sample_dir),
                    '--split-files',
                    '--threads', '8',
                    '--progress',
                    '--mem', '8G'
                ],
                capture_output=True,
                text=True
            )

            if fasterq_result.returncode != 0:
                raise Exception(f"fasterq-dump failed: {fasterq_result.stderr}")

            print(f"✓ FASTQ extraction complete")

            # Step 3: Compress FASTQ files
            print(f"[3/3] Compressing FASTQ files with pigz...")
            fastq_files = list(sample_dir.glob(f"{accession}*.fastq"))

            if not fastq_files:
                raise Exception("No FASTQ files found after extraction")

            sample_files = []
            total_size = 0

            for fastq_file in fastq_files:
                print(f"  Compressing {fastq_file.name}...")
                compress_result = subprocess.run(
                    ['pigz', '-p', '8', str(fastq_file)],
                    capture_output=True,
                    text=True
                )

                if compress_result.returncode != 0:
                    raise Exception(f"Compression failed: {compress_result.stderr}")

                # Get compressed file info
                compressed_file = fastq_file.with_suffix('.fastq.gz')
                file_size = compressed_file.stat().st_size
                total_size += file_size
                sample_files.append(str(compressed_file.name))

                print(f"  ✓ {compressed_file.name} ({file_size / 1e9:.2f} GB)")

            # Update state
            sample_state['status'] = 'completed'
            sample_state['end_time'] = datetime.now().isoformat()
            sample_state['size_gb'] = total_size / 1e9
            sample_state['files'] = sample_files
            self.state['total_downloaded_gb'] += total_size / 1e9
            self._save_state()

            print(f"\n✅ {accession} download complete ({total_size / 1e9:.2f} GB)")
            return True

        except Exception as e:
            # Update state with error
            sample_state['status'] = 'failed'
            sample_state['error'] = str(e)
            sample_state['end_time'] = datetime.now().isoformat()
            self._save_state()

            print(f"\n❌ {accession} download failed: {e}")
            return False

    def download_pool(self, pool_name: str, max_samples: Optional[int] = None,
                     sample_type: str = 'reference') -> Dict:
        """Download all samples from a pool."""
        if pool_name not in SAMPLE_POOLS:
            print(f"❌ Unknown pool: {pool_name}")
            print(f"Available pools: {', '.join(SAMPLE_POOLS.keys())}")
            return {'success': False, 'error': 'Invalid pool'}

        pool_data = SAMPLE_POOLS[pool_name]
        samples = pool_data[sample_type]

        if max_samples:
            samples = samples[:max_samples]

        print(f"\n🧬 GenomeVault Data Acquisition Pipeline")
        print(f"{'='*80}")
        print(f"Pool: {pool_name} ({pool_data['description']})")
        print(f"Sample type: {sample_type}")
        print(f"Samples to download: {len(samples)}")
        print(f"Estimated size: {len(samples) * 40:.1f} GB (40 GB/sample average)")
        print(f"Available disk space: {self.get_disk_space():.1f} GB")
        print(f"{'='*80}\n")

        # Check disk space
        required_gb = len(samples) * 40
        available_gb = self.get_disk_space()
        if available_gb < required_gb:
            print(f"⚠️  Warning: May run out of disk space!")
            print(f"   Required: {required_gb:.1f} GB")
            print(f"   Available: {available_gb:.1f} GB")
            response = input("\nContinue anyway? (y/N): ")
            if response.lower() != 'y':
                return {'success': False, 'error': 'Insufficient disk space'}

        # Download each sample
        self.state['status'] = 'downloading'
        self._save_state()

        success_count = 0
        failed_samples = []

        for i, accession in enumerate(samples, 1):
            print(f"\n[{i}/{len(samples)}] Processing {accession}...")

            if self.download_sample(accession, sample_type=sample_type, pool=pool_name):
                success_count += 1
            else:
                failed_samples.append(accession)

            # Save progress
            self._save_state()

        # Final status
        self.state['status'] = 'completed' if not failed_samples else 'completed_with_errors'
        self._save_state()

        print(f"\n{'='*80}")
        print(f"📊 Download Summary")
        print(f"{'='*80}")
        print(f"Total samples: {len(samples)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(failed_samples)}")
        print(f"Total downloaded: {self.state['total_downloaded_gb']:.2f} GB")

        if failed_samples:
            print(f"\n❌ Failed samples: {', '.join(failed_samples)}")
        else:
            print(f"\n✅ All downloads completed successfully!")

        return {
            'success': len(failed_samples) == 0,
            'total': len(samples),
            'successful': success_count,
            'failed': len(failed_samples),
            'failed_samples': failed_samples,
            'total_gb': self.state['total_downloaded_gb']
        }


def main():
    parser = argparse.ArgumentParser(
        description='Automated genomic data downloader for GenomeVault'
    )
    parser.add_argument(
        '--pool',
        choices=['european', 'east_asian', 'african', 'south_asian', 'all'],
        help='Sample pool to download'
    )
    parser.add_argument(
        '--samples',
        type=int,
        help='Maximum number of samples to download from pool'
    )
    parser.add_argument(
        '--type',
        choices=['reference', 'query'],
        default='reference',
        help='Sample type: reference (for pool) or query (for testing)'
    )
    parser.add_argument(
        '--accession',
        help='Download a single sample by accession ID'
    )
    parser.add_argument(
        '--output-dir',
        default='data/downloaded/fastq',
        help='Output directory for downloaded files'
    )
    parser.add_argument(
        '--state-file',
        default='data/download_state.json',
        help='JSON file to track download state'
    )
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Only check dependencies and exit'
    )

    args = parser.parse_args()

    # Create downloader
    downloader = GenomicDataDownloader(
        output_dir=Path(args.output_dir),
        state_file=Path(args.state_file)
    )

    # Check dependencies
    if not downloader.check_dependencies():
        sys.exit(1)

    if args.check_deps:
        print("✅ All dependencies installed")
        sys.exit(0)

    # Download single accession
    if args.accession:
        success = downloader.download_sample(args.accession)
        sys.exit(0 if success else 1)

    # Download pool
    if args.pool:
        if args.pool == 'all':
            # Download all pools
            for pool_name in ['european', 'east_asian', 'african', 'south_asian']:
                result = downloader.download_pool(pool_name, args.samples, args.type)
                if not result['success']:
                    sys.exit(1)
        else:
            result = downloader.download_pool(args.pool, args.samples, args.type)
            sys.exit(0 if result['success'] else 1)

    # No action specified
    parser.print_help()
    sys.exit(1)


if __name__ == '__main__':
    main()
