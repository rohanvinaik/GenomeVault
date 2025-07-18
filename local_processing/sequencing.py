"""
Reference-based alignment and variant calling for GenomeVault.
Implements secure local processing with differential storage.
"""
import subprocess
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np

from utils.config import config
from utils.logging import logger, audit_logger, performance_logger


@dataclass
class QualityMetrics:
    """Quality metrics for sequencing data."""
    total_reads: int
    mapped_reads: int
    mapping_rate: float
    mean_coverage: float
    coverage_uniformity: float
    q30_bases_percent: float
    duplication_rate: float
    contamination_estimate: float


@dataclass
class Variant:
    """Genetic variant representation."""
    chromosome: str
    position: int
    reference_allele: str
    alternate_allele: str
    quality: float
    genotype: str
    allele_frequency: Optional[float] = None
    annotations: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'chr': self.chromosome,
            'pos': self.position,
            'ref': self.reference_allele,
            'alt': self.alternate_allele,
            'qual': self.quality,
            'gt': self.genotype,
            'af': self.allele_frequency,
            'ann': self.annotations
        }
    
    def calculate_hash(self) -> str:
        """Calculate privacy-preserving hash of variant."""
        variant_str = f"{self.chromosome}:{self.position}:{self.reference_allele}:{self.alternate_allele}"
        return hashlib.sha256(variant_str.encode()).hexdigest()


class SequencingProcessor:
    """
    Process genomic sequencing data with privacy-preserving transformations.
    Implements reference-based differential storage.
    """
    
    def __init__(self, reference_genome: Optional[Path] = None):
        """
        Initialize sequencing processor.
        
        Args:
            reference_genome: Path to reference genome (e.g., GRCh38)
        """
        self.reference_genome = reference_genome or self._get_default_reference()
        self.temp_dir = config.processing.temp_dir / 'sequencing'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate tools
        self._validate_tools()
        
        logger.info("SequencingProcessor initialized", extra={'privacy_safe': True})
    
    def _get_default_reference(self) -> Path:
        """Get default reference genome path."""
        # In production, this would download/access the reference
        return Path("/data/references/GRCh38/GRCh38.fa")
    
    def _validate_tools(self):
        """Validate required tools are available."""
        tools = {
            'bwa': config.processing.bwa_path,
            'samtools': 'samtools',
            'bcftools': 'bcftools'
        }
        
        for tool_name, tool_path in tools.items():
            try:
                result = subprocess.run(
                    [tool_path, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0:
                    logger.warning(f"{tool_name} validation failed", extra={'privacy_safe': True})
            except Exception as e:
                logger.error(f"Tool validation error for {tool_name}: {e}", extra={'privacy_safe': True})
    
    @performance_logger.log_operation("align_fastq")
    def align_fastq(self, fastq_path: Path, sample_id: str, 
                   paired_end: Optional[Path] = None) -> Tuple[Path, QualityMetrics]:
        """
        Align FASTQ reads to reference genome using BWA-MEM2.
        
        Args:
            fastq_path: Path to FASTQ file (read 1 if paired)
            sample_id: Sample identifier
            paired_end: Path to read 2 FASTQ for paired-end data
            
        Returns:
            Tuple of (BAM file path, quality metrics)
        """
        # Create temporary workspace
        work_dir = self.temp_dir / sample_id
        work_dir.mkdir(exist_ok=True)
        
        try:
            # Output paths
            sam_path = work_dir / f"{sample_id}.sam"
            bam_path = work_dir / f"{sample_id}.sorted.bam"
            
            # Build BWA command
            bwa_cmd = [
                config.processing.bwa_path, 'mem',
                '-t', str(config.processing.max_threads),
                '-M',  # Mark shorter split hits as secondary
                '-R', f"@RG\\tID:{sample_id}\\tSM:{sample_id}\\tPL:ILLUMINA",
                str(self.reference_genome),
                str(fastq_path)
            ]
            
            if paired_end:
                bwa_cmd.append(str(paired_end))
            
            # Run alignment
            logger.info(f"Starting alignment for {sample_id}", extra={'privacy_safe': True})
            
            with open(sam_path, 'w') as sam_file:
                align_process = subprocess.Popen(
                    bwa_cmd,
                    stdout=sam_file,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                _, stderr = align_process.communicate()
                
                if align_process.returncode != 0:
                    raise RuntimeError(f"BWA alignment failed: {stderr}")
            
            # Convert SAM to sorted BAM
            sort_cmd = [
                'samtools', 'sort',
                '-@', str(config.processing.max_threads),
                '-m', '4G',
                '-o', str(bam_path),
                str(sam_path)
            ]
            
            subprocess.run(sort_cmd, check=True, capture_output=True)
            
            # Index BAM
            subprocess.run(['samtools', 'index', str(bam_path)], check=True)
            
            # Calculate quality metrics
            metrics = self._calculate_quality_metrics(bam_path)
            
            # Clean up intermediate files
            sam_path.unlink()
            
            # Audit log
            audit_logger.log_event(
                event_type="data_processing",
                actor=sample_id,
                action="align_reads",
                resource=str(fastq_path),
                metadata={'metrics': metrics.__dict__}
            )
            
            return bam_path, metrics
            
        except Exception as e:
            logger.error(f"Alignment failed for {sample_id}: {e}")
            shutil.rmtree(work_dir, ignore_errors=True)
            raise
    
    def _calculate_quality_metrics(self, bam_path: Path) -> QualityMetrics:
        """Calculate quality metrics from aligned BAM file."""
        # Get basic stats
        stats_cmd = ['samtools', 'flagstat', str(bam_path)]
        stats_result = subprocess.run(stats_cmd, capture_output=True, text=True, check=True)
        
        # Parse stats
        total_reads = 0
        mapped_reads = 0
        
        for line in stats_result.stdout.splitlines():
            if 'in total' in line:
                total_reads = int(line.split()[0])
            elif 'mapped (' in line and 'primary' not in line:
                mapped_reads = int(line.split()[0])
        
        mapping_rate = mapped_reads / total_reads if total_reads > 0 else 0
        
        # Get coverage stats
        coverage_cmd = ['samtools', 'depth', str(bam_path)]
        coverage_result = subprocess.run(coverage_cmd, capture_output=True, text=True, check=True)
        
        coverages = []
        for line in coverage_result.stdout.splitlines()[:100000]:  # Sample first 100k positions
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                coverages.append(int(parts[2]))
        
        mean_coverage = np.mean(coverages) if coverages else 0
        coverage_uniformity = 1 - (np.std(coverages) / mean_coverage if mean_coverage > 0 else 0)
        
        return QualityMetrics(
            total_reads=total_reads,
            mapped_reads=mapped_reads,
            mapping_rate=mapping_rate,
            mean_coverage=mean_coverage,
            coverage_uniformity=min(coverage_uniformity, 1.0),
            q30_bases_percent=0.85,  # Placeholder - would calculate from FASTQ
            duplication_rate=0.05,   # Placeholder - would calculate with Picard
            contamination_estimate=0.001  # Placeholder - would calculate with VerifyBamID
        )
    
    @performance_logger.log_operation("call_variants")
    def call_variants(self, bam_path: Path, sample_id: str,
                     use_deep_variant: bool = True) -> List[Variant]:
        """
        Call variants from aligned BAM file.
        
        Args:
            bam_path: Path to sorted BAM file
            sample_id: Sample identifier
            use_deep_variant: Use DeepVariant if available
            
        Returns:
            List of called variants
        """
        work_dir = self.temp_dir / sample_id
        vcf_path = work_dir / f"{sample_id}.vcf"
        
        try:
            if use_deep_variant and self._is_deepvariant_available():
                variants = self._call_variants_deepvariant(bam_path, vcf_path)
            else:
                variants = self._call_variants_bcftools(bam_path, vcf_path)
            
            # Filter variants
            filtered_variants = self._filter_variants(variants)
            
            # Audit log
            audit_logger.log_event(
                event_type="data_processing",
                actor=sample_id,
                action="call_variants",
                resource=str(bam_path),
                metadata={'variant_count': len(filtered_variants)}
            )
            
            return filtered_variants
            
        except Exception as e:
            logger.error(f"Variant calling failed for {sample_id}: {e}")
            raise
    
    def _call_variants_bcftools(self, bam_path: Path, vcf_path: Path) -> List[Variant]:
        """Call variants using bcftools."""
        # Pile up reads
        mpileup_cmd = [
            'bcftools', 'mpileup',
            '-f', str(self.reference_genome),
            '--max-depth', '1000',
            '-q', '20',  # Min mapping quality
            '-Q', '20',  # Min base quality
            str(bam_path)
        ]
        
        # Call variants
        call_cmd = [
            'bcftools', 'call',
            '-mv',  # Multiallelic caller
            '-Ov',  # Output VCF
            '-o', str(vcf_path)
        ]
        
        # Run pipeline
        mpileup_proc = subprocess.Popen(mpileup_cmd, stdout=subprocess.PIPE)
        call_proc = subprocess.Popen(call_cmd, stdin=mpileup_proc.stdout)
        mpileup_proc.stdout.close()
        call_proc.communicate()
        
        if call_proc.returncode != 0:
            raise RuntimeError("Variant calling failed")
        
        # Parse VCF
        return self._parse_vcf(vcf_path)
    
    def _is_deepvariant_available(self) -> bool:
        """Check if DeepVariant is available."""
        # In production, would check for DeepVariant installation
        return False
    
    def _call_variants_deepvariant(self, bam_path: Path, vcf_path: Path) -> List[Variant]:
        """Call variants using DeepVariant."""
        # Placeholder for DeepVariant integration
        raise NotImplementedError("DeepVariant integration pending")
    
    def _parse_vcf(self, vcf_path: Path) -> List[Variant]:
        """Parse variants from VCF file."""
        variants = []
        
        with open(vcf_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 10:
                    continue
                
                # Extract variant info
                chrom = parts[0]
                pos = int(parts[1])
                ref = parts[3]
                alt = parts[4]
                qual = float(parts[5]) if parts[5] != '.' else 0
                
                # Parse genotype
                format_fields = parts[8].split(':')
                sample_fields = parts[9].split(':')
                
                gt_idx = format_fields.index('GT') if 'GT' in format_fields else 0
                genotype = sample_fields[gt_idx] if gt_idx < len(sample_fields) else '0/0'
                
                # Create variant
                variant = Variant(
                    chromosome=chrom,
                    position=pos,
                    reference_allele=ref,
                    alternate_allele=alt,
                    quality=qual,
                    genotype=genotype
                )
                
                variants.append(variant)
        
        return variants
    
    def _filter_variants(self, variants: List[Variant], 
                        min_quality: float = 20.0) -> List[Variant]:
        """Filter variants based on quality criteria."""
        filtered = []
        
        for variant in variants:
            # Quality filter
            if variant.quality < min_quality:
                continue
            
            # Filter out homozygous reference
            if variant.genotype in ['0/0', '0|0']:
                continue
            
            filtered.append(variant)
        
        logger.info(f"Filtered {len(variants)} to {len(filtered)} variants", 
                   extra={'privacy_safe': True})
        
        return filtered
    
    def compute_differential_storage(self, variants: List[Variant],
                                   reference_version: str = "GRCh38") -> Dict:
        """
        Compute differential storage representation of variants.
        
        Args:
            variants: List of called variants
            reference_version: Reference genome version
            
        Returns:
            Differential storage representation
        """
        # Group variants by chromosome
        chrom_variants = {}
        for variant in variants:
            if variant.chromosome not in chrom_variants:
                chrom_variants[variant.chromosome] = []
            chrom_variants[variant.chromosome].append(variant)
        
        # Create differential representation
        diff_storage = {
            'reference': reference_version,
            'variant_count': len(variants),
            'chromosomes': {}
        }
        
        for chrom, chrom_vars in chrom_variants.items():
            # Sort by position
            chrom_vars.sort(key=lambda v: v.position)
            
            # Encode variants
            encoded_variants = []
            for var in chrom_vars:
                encoded = {
                    'p': var.position,  # position
                    'r': var.reference_allele,
                    'a': var.alternate_allele,
                    'g': var.genotype,
                    'q': round(var.quality, 1)
                }
                
                # Add annotations if present
                if var.annotations:
                    encoded['ann'] = var.annotations
                
                encoded_variants.append(encoded)
            
            diff_storage['chromosomes'][chrom] = {
                'variants': encoded_variants,
                'hash': self._compute_chromosome_hash(encoded_variants)
            }
        
        # Compute overall hash
        diff_storage['storage_hash'] = self._compute_storage_hash(diff_storage)
        
        return diff_storage
    
    def _compute_chromosome_hash(self, variants: List[Dict]) -> str:
        """Compute hash of chromosome variants."""
        variant_str = json.dumps(variants, sort_keys=True)
        return hashlib.sha256(variant_str.encode()).hexdigest()[:16]
    
    def _compute_storage_hash(self, storage: Dict) -> str:
        """Compute hash of entire differential storage."""
        # Remove hash fields for computation
        storage_copy = storage.copy()
        storage_copy.pop('storage_hash', None)
        
        storage_str = json.dumps(storage_copy, sort_keys=True)
        return hashlib.sha256(storage_str.encode()).hexdigest()
    
    def process_sample(self, fastq_path: Path, sample_id: str,
                      paired_end: Optional[Path] = None,
                      cleanup: bool = True) -> Dict:
        """
        Complete processing pipeline for a sample.
        
        Args:
            fastq_path: Path to FASTQ file
            sample_id: Sample identifier
            paired_end: Path to paired-end FASTQ
            cleanup: Whether to cleanup intermediate files
            
        Returns:
            Processing results including differential storage
        """
        logger.info(f"Starting sample processing for {sample_id}", 
                   extra={'privacy_safe': True})
        
        try:
            # Align reads
            bam_path, metrics = self.align_fastq(fastq_path, sample_id, paired_end)
            
            # Call variants
            variants = self.call_variants(bam_path, sample_id)
            
            # Compute differential storage
            diff_storage = self.compute_differential_storage(variants)
            
            # Prepare results
            results = {
                'sample_id': sample_id,
                'metrics': metrics.__dict__,
                'variant_count': len(variants),
                'differential_storage': diff_storage,
                'processing_complete': True
            }
            
            # Cleanup if requested
            if cleanup:
                work_dir = self.temp_dir / sample_id
                shutil.rmtree(work_dir, ignore_errors=True)
            
            logger.info(f"Sample processing complete for {sample_id}", 
                       extra={'privacy_safe': True})
            
            return results
            
        except Exception as e:
            logger.error(f"Sample processing failed for {sample_id}: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = SequencingProcessor()
    
    # Process sample (would use real FASTQ in production)
    sample_path = Path("/data/samples/sample001_R1.fastq.gz")
    
    if sample_path.exists():
        results = processor.process_sample(
            fastq_path=sample_path,
            sample_id="sample001",
            cleanup=False
        )
        
        print(f"Processing complete: {results['variant_count']} variants found")
        print(f"Storage hash: {results['differential_storage']['storage_hash']}")
