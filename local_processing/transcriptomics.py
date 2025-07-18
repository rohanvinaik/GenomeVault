"""
GenomeVault Transcriptomics Processing

Handles RNA-seq data processing including alignment, quantification,
and differential expression analysis.
"""

import os
import subprocess
import gzip
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from scipy import stats
import tempfile

from ..utils import get_logger, get_config, secure_hash
from ..utils.logging import log_operation

logger = get_logger(__name__)
config = get_config()


@dataclass
class TranscriptExpression:
    """Expression data for a single transcript/gene"""
    gene_id: str
    gene_name: str
    transcript_id: Optional[str] = None
    raw_count: float = 0.0
    tpm: float = 0.0  # Transcripts Per Million
    fpkm: float = 0.0  # Fragments Per Kilobase per Million
    normalized_count: float = 0.0
    length: int = 0
    biotype: str = "protein_coding"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'gene_id': self.gene_id,
            'gene_name': self.gene_name,
            'transcript_id': self.transcript_id,
            'raw_count': self.raw_count,
            'tpm': self.tpm,
            'fpkm': self.fpkm,
            'normalized_count': self.normalized_count,
            'length': self.length,
            'biotype': self.biotype
        }


@dataclass
class ExpressionProfile:
    """Complete expression profile for a sample"""
    sample_id: str
    expressions: List[TranscriptExpression]
    total_reads: int
    mapped_reads: int
    library_size: int
    normalization_factors: Dict[str, float]
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_expression_matrix(self) -> pd.DataFrame:
        """Get expression data as pandas DataFrame"""
        data = []
        for expr in self.expressions:
            data.append({
                'gene_id': expr.gene_id,
                'gene_name': expr.gene_name,
                'raw_count': expr.raw_count,
                'tpm': expr.tpm,
                'fpkm': expr.fpkm,
                'normalized_count': expr.normalized_count
            })
        return pd.DataFrame(data)
    
    def filter_by_expression(self, min_tpm: float = 1.0) -> List[TranscriptExpression]:
        """Filter genes by minimum expression level"""
        return [expr for expr in self.expressions if expr.tpm >= min_tpm]


@dataclass
class BatchEffectResult:
    """Results from batch effect correction"""
    corrected_expressions: Dict[str, float]
    batch_components: np.ndarray
    variance_explained: float
    metadata: Dict[str, Any]


class TranscriptomicsProcessor:
    """Main processor for transcriptomics data"""
    
    SUPPORTED_FORMATS = {'.fastq', '.fq', '.fastq.gz', '.fq.gz', '.bam'}
    
    def __init__(self,
                 reference_path: Optional[Path] = None,
                 annotation_path: Optional[Path] = None,
                 temp_dir: Optional[Path] = None,
                 max_threads: Optional[int] = None):
        """
        Initialize transcriptomics processor
        
        Args:
            reference_path: Path to reference genome
            annotation_path: Path to gene annotation (GTF/GFF)
            temp_dir: Temporary directory for processing
            max_threads: Maximum threads to use
        """
        self.reference_path = reference_path or self._get_default_reference()
        self.annotation_path = annotation_path or self._get_default_annotation()
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "genomevault_rna"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_threads = max_threads or config.processing.max_cores
        self.min_mapping_quality = 10
        
        # Validate tools
        self._validate_tools()
        
        # Load gene annotations
        self.gene_info = self._load_gene_annotations()
    
    def _get_default_reference(self) -> Path:
        """Get default reference genome path"""
        ref_dir = config.storage.data_dir / "references"
        return ref_dir / "GRCh38.fa"
    
    def _get_default_annotation(self) -> Path:
        """Get default annotation path"""
        ref_dir = config.storage.data_dir / "references"
        return ref_dir / "gencode.v42.annotation.gtf"
    
    def _validate_tools(self):
        """Validate required tools are installed"""
        required_tools = {
            'STAR': ('STAR', '--version'),
            'kallisto': ('kallisto', 'version'),
            'samtools': ('samtools', '--version'),
            'featureCounts': ('featureCounts', '-v')
        }
        
        self.available_tools = {}
        for tool_name, (command, arg) in required_tools.items():
            try:
                subprocess.run([command, arg], 
                             capture_output=True, check=True)
                self.available_tools[tool_name] = True
                logger.info(f"{tool_name} is available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.available_tools[tool_name] = False
                logger.warning(f"{tool_name} is not available")
    
    def _load_gene_annotations(self) -> Dict[str, Dict[str, Any]]:
        """Load gene annotations from GTF file"""
        gene_info = {}
        
        if not self.annotation_path.exists():
            logger.warning("Gene annotation file not found")
            return gene_info
        
        # Simple GTF parser
        with open(self.annotation_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                
                if parts[2] == 'gene':
                    # Parse attributes
                    attrs = {}
                    for attr in parts[8].split(';'):
                        if ' ' in attr:
                            key, value = attr.strip().split(' ', 1)
                            attrs[key] = value.strip('"')
                    
                    gene_id = attrs.get('gene_id', '')
                    if gene_id:
                        gene_info[gene_id] = {
                            'gene_name': attrs.get('gene_name', gene_id),
                            'biotype': attrs.get('gene_type', 'unknown'),
                            'chromosome': parts[0],
                            'start': int(parts[3]),
                            'end': int(parts[4]),
                            'strand': parts[6],
                            'length': int(parts[4]) - int(parts[3]) + 1
                        }
        
        logger.info(f"Loaded {len(gene_info)} gene annotations")
        return gene_info
    
    @log_operation("process_transcriptomics")
    def process(self, 
                input_path: Union[Path, List[Path]], 
                sample_id: str,
                paired_end: bool = True) -> ExpressionProfile:
        """
        Process RNA-seq data to generate expression profile
        
        Args:
            input_path: Path(s) to input files
            sample_id: Sample identifier
            paired_end: Whether data is paired-end
            
        Returns:
            ExpressionProfile with normalized expression values
        """
        logger.info(f"Processing RNA-seq data for sample {sample_id}")
        
        # Handle input paths
        if isinstance(input_path, Path):
            input_paths = [input_path]
        else:
            input_paths = input_path
        
        # Validate inputs
        for path in input_paths:
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")
        
        # Create working directory
        work_dir = self.temp_dir / f"rna_{sample_id}_{os.getpid()}"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Choose processing method based on available tools
            if self.available_tools.get('kallisto'):
                logger.info("Using kallisto for quantification")
                expression_data = self._process_with_kallisto(
                    input_paths, work_dir, paired_end
                )
            elif self.available_tools.get('STAR'):
                logger.info("Using STAR for alignment")
                expression_data = self._process_with_star(
                    input_paths, work_dir, paired_end
                )
            else:
                raise RuntimeError("No suitable RNA-seq tools available")
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(expression_data)
            
            # Normalize expression values
            normalized_data = self._normalize_expression(expression_data)
            
            # Create expression profile
            profile = ExpressionProfile(
                sample_id=sample_id,
                expressions=normalized_data['expressions'],
                total_reads=normalized_data['total_reads'],
                mapped_reads=normalized_data['mapped_reads'],
                library_size=normalized_data['library_size'],
                normalization_factors=normalized_data['factors'],
                quality_metrics=quality_metrics,
                metadata={
                    'paired_end': paired_end,
                    'processing_method': normalized_data.get('method', 'unknown')
                }
            )
            
            logger.info(f"Expression profiling complete. {len(profile.expressions)} genes quantified")
            return profile
            
        finally:
            # Cleanup
            if work_dir.exists():
                import shutil
                shutil.rmtree(work_dir)
    
    def _process_with_kallisto(self, 
                              input_paths: List[Path], 
                              work_dir: Path,
                              paired_end: bool) -> Dict[str, Any]:
        """Process using kallisto pseudoalignment"""
        # Check for kallisto index
        index_path = self.reference_path.parent / "kallisto_index.idx"
        
        if not index_path.exists():
            logger.info("Building kallisto index...")
            transcriptome = self.reference_path.parent / "transcriptome.fa"
            if not transcriptome.exists():
                raise FileNotFoundError("Transcriptome file not found for kallisto")
            
            subprocess.run([
                'kallisto', 'index',
                '-i', str(index_path),
                str(transcriptome)
            ], check=True)
        
        # Run kallisto quantification
        output_dir = work_dir / "kallisto_output"
        output_dir.mkdir()
        
        kallisto_cmd = [
            'kallisto', 'quant',
            '-i', str(index_path),
            '-o', str(output_dir),
            '-t', str(self.max_threads),
            '--plaintext'
        ]
        
        if paired_end and len(input_paths) == 2:
            kallisto_cmd.extend([str(input_paths[0]), str(input_paths[1])])
        else:
            kallisto_cmd.append('--single')
            kallisto_cmd.extend(['-l', '200', '-s', '20'])  # Fragment length for single-end
            kallisto_cmd.append(str(input_paths[0]))
        
        subprocess.run(kallisto_cmd, check=True)
        
        # Parse kallisto output
        abundance_file = output_dir / "abundance.tsv"
        expression_data = self._parse_kallisto_output(abundance_file)
        expression_data['method'] = 'kallisto'
        
        return expression_data
    
    def _process_with_star(self,
                          input_paths: List[Path],
                          work_dir: Path,
                          paired_end: bool) -> Dict[str, Any]:
        """Process using STAR aligner"""
        # Check for STAR index
        genome_dir = self.reference_path.parent / "STAR_index"
        
        if not genome_dir.exists():
            logger.info("Building STAR index...")
            genome_dir.mkdir()
            
            subprocess.run([
                'STAR',
                '--runMode', 'genomeGenerate',
                '--genomeDir', str(genome_dir),
                '--genomeFastaFiles', str(self.reference_path),
                '--sjdbGTFfile', str(self.annotation_path),
                '--runThreadN', str(self.max_threads)
            ], check=True)
        
        # Run STAR alignment
        output_prefix = work_dir / "star_"
        
        star_cmd = [
            'STAR',
            '--genomeDir', str(genome_dir),
            '--runThreadN', str(self.max_threads),
            '--outFileNamePrefix', str(output_prefix),
            '--outSAMtype', 'BAM', 'SortedByCoordinate',
            '--quantMode', 'GeneCounts'
        ]
        
        if paired_end and len(input_paths) == 2:
            star_cmd.extend([
                '--readFilesIn', 
                str(input_paths[0]), 
                str(input_paths[1])
            ])
        else:
            star_cmd.extend(['--readFilesIn', str(input_paths[0])])
        
        # Handle compressed files
        if str(input_paths[0]).endswith('.gz'):
            star_cmd.extend(['--readFilesCommand', 'zcat'])
        
        subprocess.run(star_cmd, check=True)
        
        # Parse STAR gene counts
        counts_file = work_dir / "star_ReadsPerGene.out.tab"
        expression_data = self._parse_star_counts(counts_file)
        expression_data['method'] = 'STAR'
        
        return expression_data
    
    def _parse_kallisto_output(self, abundance_file: Path) -> Dict[str, Any]:
        """Parse kallisto abundance output"""
        df = pd.read_csv(abundance_file, sep='\t')
        
        expressions = []
        total_counts = 0
        
        for _, row in df.iterrows():
            # Map transcript to gene if possible
            transcript_id = row['target_id']
            gene_id = transcript_id.split('.')[0]  # Simple mapping
            
            gene_info = self.gene_info.get(gene_id, {})
            
            expr = TranscriptExpression(
                gene_id=gene_id,
                gene_name=gene_info.get('gene_name', gene_id),
                transcript_id=transcript_id,
                raw_count=row['est_counts'],
                tpm=row['tpm'],
                length=int(row['length']),
                biotype=gene_info.get('biotype', 'unknown')
            )
            expressions.append(expr)
            total_counts += row['est_counts']
        
        return {
            'expressions': expressions,
            'total_reads': int(total_counts),
            'mapped_reads': int(total_counts * 0.9),  # Estimate
            'library_size': int(total_counts)
        }
    
    def _parse_star_counts(self, counts_file: Path) -> Dict[str, Any]:
        """Parse STAR gene counts output"""
        # Read counts file
        counts_data = []
        total_reads = 0
        
        with open(counts_file, 'r') as f:
            for line in f:
                if line.startswith('N_'):
                    # Parse summary stats
                    parts = line.strip().split('\t')
                    if parts[0] == 'N_unmapped':
                        total_reads += int(parts[1])
                    elif parts[0] == 'N_multimapping':
                        total_reads += int(parts[1])
                    elif parts[0] == 'N_noFeature':
                        total_reads += int(parts[1])
                else:
                    # Parse gene counts
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        gene_id = parts[0]
                        count = int(parts[1])  # Unstranded count
                        counts_data.append((gene_id, count))
                        total_reads += count
        
        # Create expressions
        expressions = []
        for gene_id, count in counts_data:
            gene_info = self.gene_info.get(gene_id, {})
            
            expr = TranscriptExpression(
                gene_id=gene_id,
                gene_name=gene_info.get('gene_name', gene_id),
                raw_count=count,
                length=gene_info.get('length', 1000),
                biotype=gene_info.get('biotype', 'unknown')
            )
            expressions.append(expr)
        
        return {
            'expressions': expressions,
            'total_reads': total_reads,
            'mapped_reads': sum(c for _, c in counts_data),
            'library_size': total_reads
        }
    
    def _normalize_expression(self, expression_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize expression values (TPM, FPKM, etc.)"""
        expressions = expression_data['expressions']
        library_size = expression_data['library_size']
        
        # Calculate normalization factors
        total_length_normalized = sum(
            expr.raw_count / (expr.length / 1000)
            for expr in expressions if expr.length > 0
        )
        
        # Normalize each gene
        for expr in expressions:
            if expr.length > 0:
                # TPM calculation
                rpk = expr.raw_count / (expr.length / 1000)
                expr.tpm = (rpk / total_length_normalized) * 1e6 if total_length_normalized > 0 else 0
                
                # FPKM calculation
                expr.fpkm = (expr.raw_count * 1e9) / (expr.length * library_size) if library_size > 0 else 0
                
                # Size factor normalization (similar to DESeq2)
                expr.normalized_count = expr.raw_count  # Placeholder
        
        # Calculate size factors
        size_factors = self._calculate_size_factors([expr.raw_count for expr in expressions])
        
        # Apply size factor normalization
        for i, expr in enumerate(expressions):
            if i < len(size_factors):
                expr.normalized_count = expr.raw_count / size_factors[i] if size_factors[i] > 0 else 0
        
        return {
            'expressions': expressions,
            'total_reads': expression_data['total_reads'],
            'mapped_reads': expression_data['mapped_reads'],
            'library_size': library_size,
            'factors': {
                'size_factor': np.median(size_factors) if size_factors else 1.0,
                'total_length_normalized': total_length_normalized
            },
            'method': expression_data.get('method', 'unknown')
        }
    
    def _calculate_size_factors(self, counts: List[float]) -> np.ndarray:
        """Calculate size factors for normalization (similar to DESeq2)"""
        if not counts or all(c == 0 for c in counts):
            return np.ones(len(counts))
        
        # Convert to numpy array
        counts_array = np.array(counts)
        
        # Filter out zeros
        non_zero_counts = counts_array[counts_array > 0]
        
        if len(non_zero_counts) == 0:
            return np.ones(len(counts))
        
        # Calculate geometric mean
        log_counts = np.log(non_zero_counts)
        geo_mean = np.exp(np.mean(log_counts))
        
        # Calculate size factors
        size_factors = counts_array / geo_mean
        size_factors[size_factors == 0] = 1.0
        
        return size_factors
    
    def _calculate_quality_metrics(self, expression_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics for expression data"""
        expressions = expression_data['expressions']
        
        # Count expressed genes
        expressed_genes = [e for e in expressions if e.raw_count > 0]
        highly_expressed = [e for e in expressions if e.tpm > 10]
        
        # Calculate metrics
        metrics = {
            'total_genes': len(expressions),
            'expressed_genes': len(expressed_genes),
            'highly_expressed_genes': len(highly_expressed),
            'percent_expressed': len(expressed_genes) / len(expressions) * 100 if expressions else 0,
            'mapping_rate': expression_data['mapped_reads'] / expression_data['total_reads'] * 100 
                           if expression_data['total_reads'] > 0 else 0
        }
        
        # Gene biotype distribution
        biotype_counts = {}
        for expr in expressions:
            biotype = expr.biotype
            if expr.raw_count > 0:
                biotype_counts[biotype] = biotype_counts.get(biotype, 0) + 1
        
        metrics['biotype_distribution'] = biotype_counts
        
        # Expression distribution stats
        tpm_values = [e.tpm for e in expressions if e.tpm > 0]
        if tpm_values:
            metrics['median_tpm'] = np.median(tpm_values)
            metrics['mean_tpm'] = np.mean(tpm_values)
            metrics['cv_tpm'] = np.std(tpm_values) / np.mean(tpm_values) if np.mean(tpm_values) > 0 else 0
        
        return metrics
    
    def batch_effect_correction(self, 
                               profiles: List[ExpressionProfile],
                               batch_labels: List[str],
                               method: str = "combat") -> List[ExpressionProfile]:
        """
        Correct batch effects across multiple samples
        
        Args:
            profiles: List of expression profiles
            batch_labels: Batch labels for each profile
            method: Correction method (combat, ruv, etc.)
            
        Returns:
            Batch-corrected expression profiles
        """
        logger.info(f"Correcting batch effects using {method} method")
        
        # Create expression matrix
        all_genes = set()
        for profile in profiles:
            all_genes.update(expr.gene_id for expr in profile.expressions)
        
        gene_list = sorted(all_genes)
        expression_matrix = np.zeros((len(gene_list), len(profiles)))
        
        for j, profile in enumerate(profiles):
            gene_to_tpm = {expr.gene_id: expr.tpm for expr in profile.expressions}
            for i, gene_id in enumerate(gene_list):
                expression_matrix[i, j] = gene_to_tpm.get(gene_id, 0)
        
        # Apply batch correction
        if method == "combat":
            corrected_matrix = self._combat_correction(
                expression_matrix, batch_labels
            )
        elif method == "quantile":
            corrected_matrix = self._quantile_normalization(expression_matrix)
        else:
            logger.warning(f"Unknown batch correction method: {method}")
            corrected_matrix = expression_matrix
        
        # Update profiles with corrected values
        corrected_profiles = []
        for j, profile in enumerate(profiles):
            corrected_expressions = []
            
            for expr in profile.expressions:
                if expr.gene_id in gene_list:
                    i = gene_list.index(expr.gene_id)
                    corrected_expr = TranscriptExpression(
                        gene_id=expr.gene_id,
                        gene_name=expr.gene_name,
                        transcript_id=expr.transcript_id,
                        raw_count=expr.raw_count,
                        tpm=corrected_matrix[i, j],
                        fpkm=expr.fpkm,
                        normalized_count=expr.normalized_count,
                        length=expr.length,
                        biotype=expr.biotype
                    )
                    corrected_expressions.append(corrected_expr)
            
            corrected_profile = ExpressionProfile(
                sample_id=profile.sample_id,
                expressions=corrected_expressions,
                total_reads=profile.total_reads,
                mapped_reads=profile.mapped_reads,
                library_size=profile.library_size,
                normalization_factors=profile.normalization_factors,
                quality_metrics=profile.quality_metrics,
                metadata={**profile.metadata, 'batch_corrected': True}
            )
            corrected_profiles.append(corrected_profile)
        
        return corrected_profiles
    
    def _combat_correction(self, 
                          expression_matrix: np.ndarray,
                          batch_labels: List[str]) -> np.ndarray:
        """Simplified ComBat batch correction"""
        # This is a simplified version - full ComBat requires more complex calculations
        
        # Convert to log scale
        log_expr = np.log2(expression_matrix + 1)
        
        # Calculate batch means
        unique_batches = list(set(batch_labels))
        batch_means = {}
        
        for batch in unique_batches:
            batch_indices = [i for i, b in enumerate(batch_labels) if b == batch]
            batch_means[batch] = np.mean(log_expr[:, batch_indices], axis=1)
        
        # Calculate overall mean
        overall_mean = np.mean(log_expr, axis=1)
        
        # Adjust for batch effects
        corrected = np.zeros_like(log_expr)
        for j, batch in enumerate(batch_labels):
            batch_effect = batch_means[batch] - overall_mean
            corrected[:, j] = log_expr[:, j] - batch_effect
        
        # Convert back from log scale
        return np.power(2, corrected) - 1
    
    def _quantile_normalization(self, expression_matrix: np.ndarray) -> np.ndarray:
        """Quantile normalization"""
        # Rank genes in each sample
        ranked = np.zeros_like(expression_matrix)
        for j in range(expression_matrix.shape[1]):
            order = np.argsort(expression_matrix[:, j])
            ranked[order, j] = np.arange(len(order))
        
        # Calculate row means of sorted data
        sorted_matrix = np.sort(expression_matrix, axis=0)
        row_means = np.mean(sorted_matrix, axis=1)
        
        # Assign mean values based on ranks
        normalized = np.zeros_like(expression_matrix)
        for j in range(expression_matrix.shape[1]):
            ranks = ranked[:, j].astype(int)
            normalized[:, j] = row_means[ranks]
        
        return normalized
