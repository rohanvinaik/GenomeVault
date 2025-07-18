"""
GenomeVault Epigenetics Processing

Handles epigenetic data processing including methylation analysis,
chromatin accessibility (ATAC-seq), and histone modifications.
"""

import os
import subprocess
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from scipy import stats, signal
import tempfile
from sklearn.mixture import GaussianMixture

from ..utils import get_logger, get_config, secure_hash
from ..utils.logging import log_operation

logger = get_logger(__name__)
config = get_config()


@dataclass
class MethylationSite:
    """Single CpG methylation site"""
    chromosome: str
    position: int
    strand: str
    methylated_reads: int
    unmethylated_reads: int
    beta_value: float  # Methylation level (0-1)
    context: str = "CG"  # CG, CHG, CHH
    gene_annotation: Optional[str] = None
    regulatory_annotation: Optional[str] = None
    
    @property
    def coverage(self) -> int:
        """Total coverage at this site"""
        return self.methylated_reads + self.unmethylated_reads
    
    @property
    def m_value(self) -> float:
        """M-value transformation of beta"""
        # M = log2(beta / (1 - beta))
        # Add small offset to avoid log(0)
        offset = 0.001
        beta_adj = max(offset, min(self.beta_value, 1 - offset))
        return np.log2(beta_adj / (1 - beta_adj))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'chr': self.chromosome,
            'pos': self.position,
            'strand': self.strand,
            'methylated': self.methylated_reads,
            'unmethylated': self.unmethylated_reads,
            'beta': self.beta_value,
            'coverage': self.coverage,
            'm_value': self.m_value,
            'context': self.context,
            'gene': self.gene_annotation,
            'regulatory': self.regulatory_annotation
        }


@dataclass
class ChromatinPeak:
    """Chromatin accessibility peak"""
    chromosome: str
    start: int
    end: int
    peak_score: float
    summit: int
    fold_enrichment: float
    p_value: float
    q_value: float
    nearest_gene: Optional[str] = None
    distance_to_tss: Optional[int] = None
    peak_annotation: Optional[str] = None
    
    @property
    def length(self) -> int:
        """Peak length"""
        return self.end - self.start
    
    @property
    def peak_id(self) -> str:
        """Unique peak identifier"""
        return f"{self.chromosome}:{self.start}-{self.end}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'chr': self.chromosome,
            'start': self.start,
            'end': self.end,
            'length': self.length,
            'score': self.peak_score,
            'summit': self.summit,
            'fold_enrichment': self.fold_enrichment,
            'p_value': self.p_value,
            'q_value': self.q_value,
            'nearest_gene': self.nearest_gene,
            'distance_to_tss': self.distance_to_tss,
            'annotation': self.peak_annotation
        }


@dataclass
class EpigeneticProfile:
    """Complete epigenetic profile"""
    sample_id: str
    profile_type: str  # methylation, chromatin, histone
    methylation_sites: Optional[List[MethylationSite]] = None
    chromatin_peaks: Optional[List[ChromatinPeak]] = None
    global_metrics: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    normalization_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_methylation_matrix(self) -> Optional[pd.DataFrame]:
        """Get methylation data as DataFrame"""
        if not self.methylation_sites:
            return None
        
        data = [site.to_dict() for site in self.methylation_sites]
        return pd.DataFrame(data)
    
    def get_chromatin_matrix(self) -> Optional[pd.DataFrame]:
        """Get chromatin data as DataFrame"""
        if not self.chromatin_peaks:
            return None
        
        data = [peak.to_dict() for peak in self.chromatin_peaks]
        return pd.DataFrame(data)


class MethylationProcessor:
    """Process whole-genome bisulfite sequencing (WGBS) data"""
    
    def __init__(self,
                 reference_path: Optional[Path] = None,
                 temp_dir: Optional[Path] = None,
                 max_threads: Optional[int] = None):
        """Initialize methylation processor"""
        self.reference_path = reference_path or self._get_default_reference()
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "genomevault_methyl"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_threads = max_threads or config.processing.max_cores
        self.min_coverage = 5  # Minimum coverage for calling methylation
        self.min_quality = 20
        
        self._validate_tools()
    
    def _get_default_reference(self) -> Path:
        """Get default reference genome"""
        ref_dir = config.storage.data_dir / "references"
        return ref_dir / "GRCh38.fa"
    
    def _validate_tools(self):
        """Validate required tools"""
        required_tools = ['bismark', 'samtools', 'bowtie2']
        self.available_tools = {}
        
        for tool in required_tools:
            try:
                subprocess.run([tool, '--version'], 
                             capture_output=True, check=True)
                self.available_tools[tool] = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.available_tools[tool] = False
                logger.warning(f"{tool} is not available")
    
    @log_operation("process_methylation")
    def process(self, 
                input_path: Union[Path, List[Path]], 
                sample_id: str,
                paired_end: bool = True) -> EpigeneticProfile:
        """Process WGBS data"""
        logger.info(f"Processing methylation data for sample {sample_id}")
        
        # Handle input paths
        if isinstance(input_path, Path):
            input_paths = [input_path]
        else:
            input_paths = input_path
        
        # Create working directory
        work_dir = self.temp_dir / f"methyl_{sample_id}_{os.getpid()}"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Run bismark alignment
            aligned_bam = self._run_bismark_alignment(input_paths, work_dir, paired_end)
            
            # Extract methylation calls
            methylation_sites = self._extract_methylation(aligned_bam, work_dir)
            
            # Apply beta-mixture normalization
            normalized_sites = self._normalize_methylation(methylation_sites)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_methylation_metrics(normalized_sites)
            
            # Calculate global methylation patterns
            global_metrics = self._calculate_global_methylation(normalized_sites)
            
            # Create profile
            profile = EpigeneticProfile(
                sample_id=sample_id,
                profile_type='methylation',
                methylation_sites=normalized_sites,
                global_metrics=global_metrics,
                quality_metrics=quality_metrics,
                normalization_params={'method': 'beta_mixture'},
                metadata={
                    'paired_end': paired_end,
                    'min_coverage': self.min_coverage,
                    'total_sites': len(normalized_sites)
                }
            )
            
            logger.info(f"Methylation processing complete. {len(normalized_sites)} CpG sites analyzed")
            return profile
            
        finally:
            # Cleanup
            if work_dir.exists():
                import shutil
                shutil.rmtree(work_dir)
    
    def _run_bismark_alignment(self, 
                              input_paths: List[Path],
                              work_dir: Path,
                              paired_end: bool) -> Path:
        """Run Bismark alignment for bisulfite sequencing"""
        # Prepare genome if needed
        genome_dir = self.reference_path.parent / "bismark_genome"
        if not genome_dir.exists():
            logger.info("Preparing bisulfite genome...")
            genome_dir.mkdir()
            subprocess.run([
                'bismark_genome_preparation',
                '--bowtie2',
                str(genome_dir)
            ], check=True)
        
        # Run alignment
        output_prefix = work_dir / "aligned"
        
        bismark_cmd = [
            'bismark',
            '--bowtie2',
            '-o', str(work_dir),
            '--temp_dir', str(work_dir),
            '--parallel', str(max(1, self.max_threads // 4)),
            str(genome_dir)
        ]
        
        if paired_end and len(input_paths) == 2:
            bismark_cmd.extend(['-1', str(input_paths[0]), '-2', str(input_paths[1])])
        else:
            bismark_cmd.append(str(input_paths[0]))
        
        subprocess.run(bismark_cmd, check=True)
        
        # Find output BAM
        bam_files = list(work_dir.glob("*.bam"))
        if not bam_files:
            raise RuntimeError("Bismark alignment failed - no BAM file produced")
        
        return bam_files[0]
    
    def _extract_methylation(self, bam_path: Path, work_dir: Path) -> List[MethylationSite]:
        """Extract methylation calls from aligned BAM"""
        # Run methylation extractor
        subprocess.run([
            'bismark_methylation_extractor',
            '--bedGraph',
            '--counts',
            '--buffer_size', '10G',
            '-o', str(work_dir),
            str(bam_path)
        ], check=True)
        
        # Parse bedGraph file
        bedgraph_files = list(work_dir.glob("*.bedGraph.gz"))
        if not bedgraph_files:
            bedgraph_files = list(work_dir.glob("*.bedGraph"))
        
        if not bedgraph_files:
            raise RuntimeError("No methylation bedGraph file found")
        
        methylation_sites = []
        
        # Parse methylation calls
        open_func = gzip.open if str(bedgraph_files[0]).endswith('.gz') else open
        with open_func(bedgraph_files[0], 'rt') as f:
            for line in f:
                if line.startswith('track') or line.startswith('#'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                
                try:
                    chrom = parts[0]
                    start = int(parts[1])
                    end = int(parts[2])
                    
                    # Parse methylation info
                    # Format: percentage methylated<tab>methylated<tab>unmethylated
                    info_parts = parts[3].split()
                    if len(info_parts) >= 3:
                        methylated = int(info_parts[1])
                        unmethylated = int(info_parts[2])
                    else:
                        continue
                    
                    # Calculate beta value
                    total = methylated + unmethylated
                    if total >= self.min_coverage:
                        beta = methylated / total
                        
                        site = MethylationSite(
                            chromosome=chrom,
                            position=start,
                            strand='+',  # Will be updated if strand-specific
                            methylated_reads=methylated,
                            unmethylated_reads=unmethylated,
                            beta_value=beta
                        )
                        methylation_sites.append(site)
                
                except (ValueError, IndexError) as e:
                    logger.debug(f"Skipping malformed line: {line.strip()}")
                    continue
        
        return methylation_sites
    
    def _normalize_methylation(self, sites: List[MethylationSite]) -> List[MethylationSite]:
        """Apply beta-mixture quantile normalization"""
        if not sites:
            return sites
        
        # Extract beta values
        beta_values = np.array([site.beta_value for site in sites])
        
        # Fit beta mixture model
        # Typically 3 components: unmethylated, partially methylated, fully methylated
        gmm = GaussianMixture(n_components=3, random_state=42)
        
        # Transform to M-values for better gaussian fit
        m_values = np.array([site.m_value for site in sites])
        m_values_reshaped = m_values.reshape(-1, 1)
        
        try:
            gmm.fit(m_values_reshaped)
            
            # Normalize within each component
            labels = gmm.predict(m_values_reshaped)
            
            normalized_betas = np.zeros_like(beta_values)
            for component in range(3):
                mask = labels == component
                if np.sum(mask) > 0:
                    # Quantile normalize within component
                    component_betas = beta_values[mask]
                    ranks = stats.rankdata(component_betas)
                    quantiles = ranks / (len(ranks) + 1)
                    
                    # Map to standard beta distribution
                    normalized_betas[mask] = stats.beta.ppf(quantiles, a=2, b=2)
            
            # Update sites with normalized values
            for i, site in enumerate(sites):
                site.beta_value = normalized_betas[i]
        
        except Exception as e:
            logger.warning(f"Beta-mixture normalization failed: {e}. Using original values.")
        
        return sites
    
    def _calculate_methylation_metrics(self, sites: List[MethylationSite]) -> Dict[str, float]:
        """Calculate quality metrics for methylation data"""
        if not sites:
            return {}
        
        coverages = [site.coverage for site in sites]
        beta_values = [site.beta_value for site in sites]
        
        metrics = {
            'total_cpg_sites': len(sites),
            'mean_coverage': np.mean(coverages),
            'median_coverage': np.median(coverages),
            'sites_above_5x': sum(1 for c in coverages if c >= 5),
            'sites_above_10x': sum(1 for c in coverages if c >= 10),
            'mean_methylation': np.mean(beta_values),
            'median_methylation': np.median(beta_values),
            'methylation_std': np.std(beta_values)
        }
        
        # Methylation level distribution
        unmethylated = sum(1 for b in beta_values if b < 0.2)
        partial = sum(1 for b in beta_values if 0.2 <= b < 0.8)
        methylated = sum(1 for b in beta_values if b >= 0.8)
        
        metrics['unmethylated_sites'] = unmethylated
        metrics['partially_methylated_sites'] = partial
        metrics['methylated_sites'] = methylated
        
        return metrics
    
    def _calculate_global_methylation(self, sites: List[MethylationSite]) -> Dict[str, float]:
        """Calculate global methylation patterns"""
        if not sites:
            return {}
        
        # Group by context
        context_methylation = {}
        for context in ['CG', 'CHG', 'CHH']:
            context_sites = [s for s in sites if s.context == context]
            if context_sites:
                context_methylation[f'{context}_mean'] = np.mean(
                    [s.beta_value for s in context_sites]
                )
                context_methylation[f'{context}_count'] = len(context_sites)
        
        # Calculate chromosome-level methylation
        chrom_methylation = {}
        for site in sites:
            if site.chromosome not in chrom_methylation:
                chrom_methylation[site.chromosome] = []
            chrom_methylation[site.chromosome].append(site.beta_value)
        
        # Average per chromosome
        for chrom, values in chrom_methylation.items():
            context_methylation[f'chr_{chrom}_mean'] = np.mean(values)
        
        return context_methylation


class ChromatinAccessibilityProcessor:
    """Process ATAC-seq data for chromatin accessibility"""
    
    def __init__(self,
                 reference_path: Optional[Path] = None,
                 temp_dir: Optional[Path] = None,
                 max_threads: Optional[int] = None):
        """Initialize chromatin processor"""
        self.reference_path = reference_path or self._get_default_reference()
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "genomevault_atac"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_threads = max_threads or config.processing.max_cores
        self.min_quality = 30
        
        self._validate_tools()
    
    def _get_default_reference(self) -> Path:
        """Get default reference genome"""
        ref_dir = config.storage.data_dir / "references"
        return ref_dir / "GRCh38.fa"
    
    def _validate_tools(self):
        """Validate required tools"""
        required_tools = ['bowtie2', 'samtools', 'macs2', 'bedtools']
        self.available_tools = {}
        
        for tool in required_tools:
            try:
                if tool == 'macs2':
                    subprocess.run(['macs2', '--version'], 
                                 capture_output=True, check=True)
                else:
                    subprocess.run([tool, '--version'], 
                                 capture_output=True, check=True)
                self.available_tools[tool] = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.available_tools[tool] = False
                logger.warning(f"{tool} is not available")
    
    @log_operation("process_chromatin_accessibility")
    def process(self, 
                input_path: Union[Path, List[Path]], 
                sample_id: str,
                paired_end: bool = True) -> EpigeneticProfile:
        """Process ATAC-seq data"""
        logger.info(f"Processing ATAC-seq data for sample {sample_id}")
        
        # Handle input paths
        if isinstance(input_path, Path):
            input_paths = [input_path]
        else:
            input_paths = input_path
        
        # Create working directory
        work_dir = self.temp_dir / f"atac_{sample_id}_{os.getpid()}"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Align reads
            aligned_bam = self._align_atac_reads(input_paths, work_dir, paired_end)
            
            # Remove duplicates and mitochondrial reads
            filtered_bam = self._filter_atac_reads(aligned_bam, work_dir)
            
            # Call peaks
            peaks = self._call_peaks(filtered_bam, work_dir)
            
            # Annotate peaks
            annotated_peaks = self._annotate_peaks(peaks)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_atac_metrics(filtered_bam, annotated_peaks)
            
            # Create profile
            profile = EpigeneticProfile(
                sample_id=sample_id,
                profile_type='chromatin',
                chromatin_peaks=annotated_peaks,
                quality_metrics=quality_metrics,
                metadata={
                    'paired_end': paired_end,
                    'total_peaks': len(annotated_peaks)
                }
            )
            
            logger.info(f"ATAC-seq processing complete. {len(annotated_peaks)} peaks called")
            return profile
            
        finally:
            # Cleanup
            if work_dir.exists():
                import shutil
                shutil.rmtree(work_dir)
    
    def _align_atac_reads(self, 
                         input_paths: List[Path],
                         work_dir: Path,
                         paired_end: bool) -> Path:
        """Align ATAC-seq reads using bowtie2"""
        # Check for bowtie2 index
        index_prefix = self.reference_path.with_suffix('')
        if not (index_prefix.parent / f"{index_prefix.name}.1.bt2").exists():
            logger.info("Building bowtie2 index...")
            subprocess.run([
                'bowtie2-build',
                str(self.reference_path),
                str(index_prefix)
            ], check=True)
        
        # Run alignment
        output_sam = work_dir / "aligned.sam"
        
        bowtie2_cmd = [
            'bowtie2',
            '-x', str(index_prefix),
            '-p', str(self.max_threads),
            '--very-sensitive',
            '-S', str(output_sam)
        ]
        
        if paired_end and len(input_paths) == 2:
            bowtie2_cmd.extend(['-1', str(input_paths[0]), '-2', str(input_paths[1])])
        else:
            bowtie2_cmd.extend(['-U', str(input_paths[0])])
        
        subprocess.run(bowtie2_cmd, check=True)
        
        # Convert to sorted BAM
        output_bam = work_dir / "aligned.bam"
        subprocess.run([
            'samtools', 'sort',
            '-@', str(self.max_threads),
            '-o', str(output_bam),
            str(output_sam)
        ], check=True)
        
        # Index BAM
        subprocess.run(['samtools', 'index', str(output_bam)], check=True)
        
        # Remove SAM
        output_sam.unlink()
        
        return output_bam
    
    def _filter_atac_reads(self, bam_path: Path, work_dir: Path) -> Path:
        """Filter ATAC-seq reads (remove duplicates, mitochondrial)"""
        # Remove duplicates
        dedup_bam = work_dir / "dedup.bam"
        subprocess.run([
            'samtools', 'markdup',
            '-r',  # Remove duplicates
            '-@', str(self.max_threads),
            str(bam_path),
            str(dedup_bam)
        ], check=True)
        
        # Filter out mitochondrial reads and low quality
        filtered_bam = work_dir / "filtered.bam"
        subprocess.run([
            'samtools', 'view',
            '-b',
            '-q', str(self.min_quality),
            '-@', str(self.max_threads),
            str(dedup_bam),
            'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
            'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
            'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22',
            'chrX', 'chrY',
            '-o', str(filtered_bam)
        ], check=True)
        
        # Index filtered BAM
        subprocess.run(['samtools', 'index', str(filtered_bam)], check=True)
        
        return filtered_bam
    
    def _call_peaks(self, bam_path: Path, work_dir: Path) -> List[ChromatinPeak]:
        """Call peaks using MACS2"""
        # Run MACS2
        output_prefix = work_dir / "peaks"
        
        macs2_cmd = [
            'macs2', 'callpeak',
            '-t', str(bam_path),
            '-f', 'BAMPE',  # Paired-end BAM
            '-g', 'hs',  # Human genome
            '-n', str(output_prefix),
            '--outdir', str(work_dir),
            '-q', '0.05',
            '--nomodel',
            '--shift', '-75',
            '--extsize', '150'
        ]
        
        subprocess.run(macs2_cmd, check=True)
        
        # Parse narrowPeak file
        peaks = []
        peak_file = work_dir / "peaks_peaks.narrowPeak"
        
        if not peak_file.exists():
            logger.warning("No peaks called by MACS2")
            return peaks
        
        with open(peak_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 10:
                    peak = ChromatinPeak(
                        chromosome=parts[0],
                        start=int(parts[1]),
                        end=int(parts[2]),
                        peak_score=float(parts[4]),
                        summit=int(parts[1]) + int(parts[9]),
                        fold_enrichment=float(parts[6]),
                        p_value=float(parts[7]),
                        q_value=float(parts[8])
                    )
                    peaks.append(peak)
        
        return peaks
    
    def _annotate_peaks(self, peaks: List[ChromatinPeak]) -> List[ChromatinPeak]:
        """Annotate peaks with nearest genes and regulatory elements"""
        # This is a simplified annotation
        # In production, would use tools like HOMER or ChIPseeker
        
        for peak in peaks:
            # Simple distance-based annotation
            if peak.start < 2000:  # Near chromosome start
                peak.peak_annotation = "promoter"
                peak.distance_to_tss = peak.start
            elif peak.end - peak.start > 1000:
                peak.peak_annotation = "enhancer"
            else:
                peak.peak_annotation = "regulatory"
        
        return peaks
    
    def _calculate_atac_metrics(self, 
                               bam_path: Path, 
                               peaks: List[ChromatinPeak]) -> Dict[str, float]:
        """Calculate ATAC-seq quality metrics"""
        # Get alignment statistics
        stats_output = subprocess.run([
            'samtools', 'flagstat', str(bam_path)
        ], capture_output=True, text=True)
        
        # Parse statistics
        total_reads = 0
        mapped_reads = 0
        for line in stats_output.stdout.split('\n'):
            if 'total' in line and 'secondary' not in line:
                total_reads = int(line.split()[0])
            elif 'mapped (' in line and 'secondary' not in line:
                mapped_reads = int(line.split()[0])
        
        # Peak statistics
        peak_lengths = [p.length for p in peaks]
        
        metrics = {
            'total_reads': total_reads,
            'mapped_reads': mapped_reads,
            'mapping_rate': mapped_reads / total_reads * 100 if total_reads > 0 else 0,
            'total_peaks': len(peaks),
            'mean_peak_length': np.mean(peak_lengths) if peak_lengths else 0,
            'median_peak_length': np.median(peak_lengths) if peak_lengths else 0,
            'total_peak_space': sum(peak_lengths)
        }
        
        # Peak score distribution
        if peaks:
            scores = [p.peak_score for p in peaks]
            metrics['mean_peak_score'] = np.mean(scores)
            metrics['median_peak_score'] = np.median(scores)
        
        return metrics


# Convenience function to create appropriate processor
def create_epigenetic_processor(data_type: str, **kwargs):
    """Create appropriate epigenetic processor based on data type"""
    if data_type.lower() in ['methylation', 'wgbs', 'rrbs']:
        return MethylationProcessor(**kwargs)
    elif data_type.lower() in ['chromatin', 'atac', 'atac-seq']:
        return ChromatinAccessibilityProcessor(**kwargs)
    else:
        raise ValueError(f"Unknown epigenetic data type: {data_type}")
