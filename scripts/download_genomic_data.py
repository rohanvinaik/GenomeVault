#!/usr/bin/env python3
"""
GenomeVault Genomic Data Downloader

Downloads real, open-access genomic data from public repositories:
- 1000 Genomes Project (diverse populations, whole genomes)
- GIAB (Genome in a Bottle) - gold standard reference samples
- SRA (Sequence Read Archive) - thousands of public datasets

Supports:
- FASTQ files (raw sequencing reads)
- VCF files (variant calls)
- Reference genomes
- Diverse genomic variations (SNPs, indels, CNVs, structural variants)

Usage:
    # Quick start - download curated test dataset
    python scripts/download_genomic_data.py --preset quick-test
    
    # Download specific data types
    python scripts/download_genomic_data.py --source giab --sample NA12878
    python scripts/download_genomic_data.py --source 1000genomes --population EUR --samples 3
    
    # Download via SRA accession
    python scripts/download_genomic_data.py --sra-accession SRR000001
    
    # Custom configuration
    python scripts/download_genomic_data.py --config my_downloads.yaml
"""

import argparse
import logging
import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import urllib.request
import hashlib
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class GenomicDataset:
    """Represents a genomic dataset to download."""
    name: str
    source: str
    data_type: str  # fastq, vcf, bam
    url: str
    md5: Optional[str] = None
    description: str = ""
    size_mb: Optional[float] = None
    coverage: Optional[str] = None
    platform: Optional[str] = None


class GenomicDataDownloader:
    """
    Downloads and organizes genomic data from public repositories.
    """
    
    # Curated datasets - high-quality, well-characterized samples
    CURATED_DATASETS = {
        # GIAB Reference Samples - Gold Standard
        'giab_na12878_illumina': GenomicDataset(
            name='NA12878_Illumina_2x250_chr22',
            source='giab',
            data_type='fastq',
            url='ftp://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/NA12878/Garvan_NA12878_HG001_HiSeq_Exome/NIST7035_TAAGGCGA_L001_R1_001.fastq.gz',
            description='GIAB NA12878 (HG001) - CEU female, gold standard reference',
            size_mb=2500,
            coverage='30x',
            platform='Illumina HiSeq'
        ),
        
        # 1000 Genomes - Diverse Populations
        '1kg_hg00096_exome': GenomicDataset(
            name='HG00096_GBR_exome',
            source='1000genomes',
            data_type='fastq',
            url='ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/exome_alignment/HG00096.chrom22.ILLUMINA.bwa.GBR.exome.20121211.bam',
            description='1000 Genomes HG00096 - British (GBR) sample, chr22 exome',
            size_mb=150,
            coverage='60x',
            platform='Illumina'
        ),
        
        '1kg_na19238_wgs': GenomicDataset(
            name='NA19238_YRI_chr22',
            source='1000genomes',
            data_type='bam',
            url='ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/NA19238/alignment/NA19238.chrom22.ILLUMINA.bwa.YRI.low_coverage.20130415.bam',
            description='1000 Genomes NA19238 - Yoruba (YRI) sample, chr22',
            size_mb=200,
            coverage='5x',
            platform='Illumina'
        ),
        
        # Platinum Genomes - High-quality, deep coverage
        'platinum_na12877': GenomicDataset(
            name='NA12877_PlatinumGenomes',
            source='platinum',
            data_type='fastq',
            url='https://storage.googleapis.com/brain-genomics-public/research/sequencing/fastq/platinum/NA12877/H06HDADXX130110.1.ATCACGAT.20k_reads_1.fastq.gz',
            description='Illumina Platinum Genomes NA12877 - high-quality reference',
            size_mb=100,
            coverage='50x',
            platform='Illumina HiSeq'
        ),
    }
    
    # Reference genomes
    REFERENCE_GENOMES = {
        'hg38_chr22': {
            'url': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz',
            'size_mb': 15,
            'description': 'Human reference genome hg38, chromosome 22'
        },
        'hg38_full': {
            'url': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz',
            'size_mb': 900,
            'description': 'Complete human reference genome hg38'
        },
        'grch38_chr22': {
            'url': 'https://ftp.ensembl.org/pub/release-104/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz',
            'size_mb': 15,
            'description': 'GRCh38 chromosome 22 from Ensembl'
        }
    }
    
    # Variant call datasets (VCF)
    VCF_DATASETS = {
        '1kg_phase3_chr22': {
            'url': 'http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz',
            'size_mb': 300,
            'samples': 2504,
            'variants': '~1M SNPs + indels',
            'description': '1000 Genomes Phase 3, chr22, all populations'
        },
        'giab_na12878_vcf': {
            'url': 'ftp://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/NA12878_HG001/NISTv4.2.1/GRCh38/HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz',
            'size_mb': 450,
            'description': 'GIAB NA12878 high-confidence calls, GRCh38'
        }
    }
    
    def __init__(self, output_dir: Path, use_aria2: bool = True):
        """
        Initialize the downloader.
        
        Args:
            output_dir: Base directory for downloaded data
            use_aria2: Use aria2c for faster downloads (if available)
        """
        self.output_dir = Path(output_dir)
        self.use_aria2 = use_aria2 and shutil.which('aria2c')
        
        # Create directory structure
        self.fastq_dir = self.output_dir / 'fastq'
        self.vcf_dir = self.output_dir / 'vcf'
        self.bam_dir = self.output_dir / 'bam'
        self.reference_dir = self.output_dir / 'reference'
        self.metadata_dir = self.output_dir / 'metadata'
        
        for dir_path in [self.fastq_dir, self.vcf_dir, self.bam_dir, 
                        self.reference_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
        if self.use_aria2:
            logger.info("Using aria2c for faster downloads")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required tools are installed."""
        tools = {
            'wget': shutil.which('wget') is not None,
            'curl': shutil.which('curl') is not None,
            'aria2c': shutil.which('aria2c') is not None,
            'prefetch': shutil.which('prefetch') is not None,  # SRA Toolkit
            'fastq-dump': shutil.which('fastq-dump') is not None,
            'fasterq-dump': shutil.which('fasterq-dump') is not None,
            'samtools': shutil.which('samtools') is not None,
        }
        
        logger.info("Dependency check:")
        for tool, available in tools.items():
            status = "✓" if available else "✗"
            logger.info(f"  {status} {tool}")
        
        return tools
    
    def download_file(self, url: str, output_path: Path, 
                     description: str = "") -> bool:
        """
        Download a file using the best available method.
        
        Args:
            url: URL to download
            output_path: Where to save the file
            description: Human-readable description
            
        Returns:
            True if successful
        """
        if output_path.exists():
            logger.info(f"✓ Already exists: {output_path.name}")
            return True
        
        logger.info(f"Downloading: {description or output_path.name}")
        logger.info(f"  URL: {url}")
        logger.info(f"  Destination: {output_path}")
        
        try:
            if self.use_aria2:
                # aria2c for faster, resumable downloads
                cmd = [
                    'aria2c',
                    '--continue=true',
                    '--max-connection-per-server=16',
                    '--min-split-size=1M',
                    '--split=16',
                    '--dir', str(output_path.parent),
                    '--out', output_path.name,
                    url
                ]
                subprocess.run(cmd, check=True)
            elif shutil.which('wget'):
                # wget with resume capability
                cmd = ['wget', '-c', '-O', str(output_path), url]
                subprocess.run(cmd, check=True)
            else:
                # Python fallback
                logger.info("  Using Python urllib (slower, no resume)")
                urllib.request.urlretrieve(url, output_path)
            
            logger.info(f"✓ Downloaded: {output_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to download {url}: {e}")
            if output_path.exists():
                output_path.unlink()  # Clean up partial download
            return False
    
    def download_sra_dataset(self, accession: str, output_dir: Path) -> bool:
        """
        Download dataset from SRA using SRA Toolkit.
        
        Args:
            accession: SRA accession (e.g., SRR000001, SRX000001)
            output_dir: Directory for output FASTQ files
            
        Returns:
            True if successful
        """
        if not shutil.which('fasterq-dump'):
            logger.error("fasterq-dump not found. Install SRA Toolkit:")
            logger.error("  conda install -c bioconda sra-tools")
            return False
        
        logger.info(f"Downloading SRA dataset: {accession}")
        
        try:
            # Use fasterq-dump (faster than fastq-dump)
            cmd = [
                'fasterq-dump',
                accession,
                '--outdir', str(output_dir),
                '--split-files',  # Split paired-end reads
                '--progress',
                '--threads', '4'
            ]
            
            subprocess.run(cmd, check=True)
            logger.info(f"✓ Downloaded SRA dataset: {accession}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to download {accession}: {e}")
            return False
    
    def download_reference_genome(self, genome_id: str = 'hg38_chr22') -> Path:
        """
        Download reference genome.
        
        Args:
            genome_id: Reference genome identifier
            
        Returns:
            Path to downloaded genome
        """
        if genome_id not in self.REFERENCE_GENOMES:
            raise ValueError(f"Unknown reference genome: {genome_id}")
        
        ref_info = self.REFERENCE_GENOMES[genome_id]
        output_path = self.reference_dir / f"{genome_id}.fa.gz"
        
        success = self.download_file(
            ref_info['url'],
            output_path,
            ref_info['description']
        )
        
        if success:
            # Decompress if needed
            decompressed = output_path.with_suffix('')
            if not decompressed.exists():
                logger.info(f"Decompressing {output_path.name}...")
                subprocess.run(['gunzip', '-k', str(output_path)], check=True)
                logger.info(f"✓ Decompressed: {decompressed.name}")
            
            return decompressed
        
        return None
    
    def download_vcf_dataset(self, dataset_id: str) -> Path:
        """
        Download VCF variant dataset.
        
        Args:
            dataset_id: VCF dataset identifier
            
        Returns:
            Path to downloaded VCF
        """
        if dataset_id not in self.VCF_DATASETS:
            raise ValueError(f"Unknown VCF dataset: {dataset_id}")
        
        vcf_info = self.VCF_DATASETS[dataset_id]
        output_path = self.vcf_dir / f"{dataset_id}.vcf.gz"
        
        success = self.download_file(
            vcf_info['url'],
            output_path,
            vcf_info['description']
        )
        
        if success:
            # Download index if available
            index_url = vcf_info['url'] + '.tbi'
            index_path = output_path.with_suffix('.vcf.gz.tbi')
            self.download_file(index_url, index_path, "VCF index")
            
            return output_path
        
        return None
    
    def download_curated_dataset(self, dataset_id: str) -> Optional[Path]:
        """
        Download a curated genomic dataset.
        
        Args:
            dataset_id: Dataset identifier from CURATED_DATASETS
            
        Returns:
            Path to downloaded file
        """
        if dataset_id not in self.CURATED_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_id}")
        
        dataset = self.CURATED_DATASETS[dataset_id]
        
        # Determine output directory based on data type
        if dataset.data_type == 'fastq':
            output_dir = self.fastq_dir
        elif dataset.data_type == 'vcf':
            output_dir = self.vcf_dir
        elif dataset.data_type == 'bam':
            output_dir = self.bam_dir
        else:
            output_dir = self.output_dir
        
        # Preserve original filename
        filename = dataset.url.split('/')[-1]
        output_path = output_dir / filename
        
        success = self.download_file(
            dataset.url,
            output_path,
            dataset.description
        )
        
        if success:
            # Save metadata
            metadata = {
                'name': dataset.name,
                'source': dataset.source,
                'data_type': dataset.data_type,
                'description': dataset.description,
                'size_mb': dataset.size_mb,
                'coverage': dataset.coverage,
                'platform': dataset.platform,
                'url': dataset.url,
                'local_path': str(output_path)
            }
            
            metadata_file = self.metadata_dir / f"{dataset.name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return output_path
        
        return None
    
    def download_quick_test_set(self):
        """
        Download a curated set of small, high-quality datasets for testing.
        Includes:
        - Reference genome (chr22)
        - 1-2 FASTQ samples with diverse variants
        - VCF with known variants
        """
        logger.info("=" * 70)
        logger.info("Downloading Quick Test Dataset")
        logger.info("=" * 70)
        
        downloads = []
        
        # 1. Reference genome
        logger.info("\n1. Reference Genome (chr22)")
        ref = self.download_reference_genome('hg38_chr22')
        if ref:
            downloads.append(('reference', ref))
        
        # 2. Small FASTQ dataset (or BAM we can convert)
        logger.info("\n2. Sample Data (1000 Genomes, chr22)")
        fastq = self.download_curated_dataset('1kg_hg00096_exome')
        if fastq:
            downloads.append(('sample_data', fastq))
        
        # 3. VCF with variants
        logger.info("\n3. Variant Calls (1000 Genomes Phase 3, chr22)")
        vcf = self.download_vcf_dataset('1kg_phase3_chr22')
        if vcf:
            downloads.append(('variants', vcf))
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("Download Summary")
        logger.info("=" * 70)
        
        for name, path in downloads:
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ {name}: {path.name} ({size_mb:.1f} MB)")
        
        logger.info(f"\nAll files saved to: {self.output_dir}")
        logger.info("\nNext steps:")
        logger.info("1. If you downloaded BAM files, convert to FASTQ:")
        logger.info("   samtools fastq input.bam > output.fastq")
        logger.info("2. Run GenomeVault pipeline:")
        logger.info("   python benchmarks/run_alignment_optimized_pipeline.py")
        
        return downloads
    
    def create_download_manifest(self) -> Path:
        """
        Create a manifest of available datasets for easy reference.
        
        Returns:
            Path to manifest file
        """
        manifest = {
            'curated_datasets': {},
            'reference_genomes': self.REFERENCE_GENOMES,
            'vcf_datasets': self.VCF_DATASETS
        }
        
        for dataset_id, dataset in self.CURATED_DATASETS.items():
            manifest['curated_datasets'][dataset_id] = {
                'name': dataset.name,
                'source': dataset.source,
                'type': dataset.data_type,
                'description': dataset.description,
                'size_mb': dataset.size_mb,
                'coverage': dataset.coverage,
                'platform': dataset.platform
            }
        
        manifest_file = self.output_dir / 'available_datasets.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Created dataset manifest: {manifest_file}")
        return manifest_file


def setup_sra_toolkit():
    """Instructions for setting up SRA Toolkit."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    SRA Toolkit Setup Instructions                     ║
╚══════════════════════════════════════════════════════════════════════╝

To download data from SRA (Sequence Read Archive), install SRA Toolkit:

Option 1: Using conda (recommended)
    conda install -c bioconda sra-tools

Option 2: Using brew (macOS)
    brew install sra-tools

Option 3: Manual installation
    https://github.com/ncbi/sra-tools/wiki/02.-Installing-SRA-Toolkit

After installation, configure SRA Toolkit:
    vdb-config --interactive

For faster downloads, also install:
    conda install -c bioconda aria2c
    
""")


def main():
    parser = argparse.ArgumentParser(
        description='Download genomic data for GenomeVault testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test dataset (recommended for first use)
  python scripts/download_genomic_data.py --preset quick-test
  
  # Specific curated dataset
  python scripts/download_genomic_data.py --dataset giab_na12878_illumina
  
  # Reference genome only
  python scripts/download_genomic_data.py --reference hg38_chr22
  
  # VCF variants only
  python scripts/download_genomic_data.py --vcf 1kg_phase3_chr22
  
  # SRA accession
  python scripts/download_genomic_data.py --sra-accession SRR000001
  
  # List available datasets
  python scripts/download_genomic_data.py --list-datasets
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/downloaded'),
        help='Output directory for downloads (default: data/downloaded)'
    )
    
    parser.add_argument(
        '--preset',
        choices=['quick-test', 'full-test', 'giab-gold-standard'],
        help='Download a preset collection of datasets'
    )
    
    parser.add_argument(
        '--dataset',
        help='Download specific curated dataset by ID'
    )
    
    parser.add_argument(
        '--reference',
        help='Download reference genome by ID (e.g., hg38_chr22)'
    )
    
    parser.add_argument(
        '--vcf',
        help='Download VCF dataset by ID'
    )
    
    parser.add_argument(
        '--sra-accession',
        help='Download from SRA by accession (e.g., SRR000001)'
    )
    
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List all available datasets and exit'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies and exit'
    )
    
    parser.add_argument(
        '--no-aria2',
        action='store_true',
        help='Disable aria2c even if available'
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = GenomicDataDownloader(
        args.output_dir,
        use_aria2=not args.no_aria2
    )
    
    # Handle different modes
    if args.check_deps:
        deps = downloader.check_dependencies()
        
        missing_critical = []
        if not (deps['wget'] or deps['curl']):
            missing_critical.append('wget or curl')
        
        if missing_critical:
            logger.error(f"Missing critical dependencies: {', '.join(missing_critical)}")
            sys.exit(1)
        
        if not deps['prefetch'] or not deps['fasterq-dump']:
            logger.warning("SRA Toolkit not found. SRA downloads will not work.")
            logger.info("Run: python scripts/download_genomic_data.py --help-sra")
        
        logger.info("\n✓ All critical dependencies satisfied")
        sys.exit(0)
    
    if args.list_datasets:
        manifest = downloader.create_download_manifest()
        
        print("\n" + "=" * 70)
        print("Available Curated Datasets")
        print("=" * 70)
        for dataset_id, dataset in downloader.CURATED_DATASETS.items():
            print(f"\n{dataset_id}:")
            print(f"  Name: {dataset.name}")
            print(f"  Source: {dataset.source}")
            print(f"  Type: {dataset.data_type}")
            print(f"  Size: ~{dataset.size_mb} MB")
            print(f"  Coverage: {dataset.coverage}")
            print(f"  Description: {dataset.description}")
        
        print(f"\nFull manifest saved to: {manifest}")
        sys.exit(0)
    
    # Download based on arguments
    if args.preset == 'quick-test':
        downloader.download_quick_test_set()
    
    elif args.dataset:
        downloader.download_curated_dataset(args.dataset)
    
    elif args.reference:
        downloader.download_reference_genome(args.reference)
    
    elif args.vcf:
        downloader.download_vcf_dataset(args.vcf)
    
    elif args.sra_accession:
        downloader.download_sra_dataset(args.sra_accession, downloader.fastq_dir)
    
    else:
        parser.print_help()
        print("\n💡 Tip: Start with: python scripts/download_genomic_data.py --preset quick-test")


if __name__ == '__main__':
    main()
