"""
Epigenetics Processing Module

Handles epigenetic data including:
- DNA methylation (WGBS, RRBS, arrays)
- Chromatin accessibility (ATAC-seq)
- Histone modifications (ChIP-seq)
- Positional encoding for genomic context
"""

import gzip
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from genomevault.core.config import get_config
from genomevault.core.exceptions import ProcessingError, ValidationError
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)
config = get_config()


class EpigeneticDataType(Enum):
    """Types of epigenetic data"""

    METHYLATION = "methylation"
    CHROMATIN_ACCESSIBILITY = "chromatin_accessibility"
    HISTONE_MARKS = "histone_marks"
    THREE_D_INTERACTIONS = "3d_interactions"


class MethylationContext(Enum):
    """DNA methylation contexts"""

    CG = "CG"  # CpG context
    CHG = "CHG"  # CHG context (H = A, C, or T)
    CHH = "CHH"  # CHH context
    ALL = "ALL"  # All contexts


@dataclass
class MethylationSite:
    """Individual methylation site data"""

    chromosome: str
    position: int
    context: MethylationContext
    methylation_level: float  # 0-1
    coverage: int
    strand: str = "+"
    gene_id: Optional[str] = None
    region_type: Optional[str] = None  # promoter, gene_body, intergenic

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "chromosome": self.chromosome,
            "position": self.position,
            "context": self.context.value,
            "methylation_level": self.methylation_level,
            "coverage": self.coverage,
            "strand": self.strand,
            "gene_id": self.gene_id,
            "region_type": self.region_type,
        }


@dataclass
class ChromatinPeak:
    """Chromatin accessibility peak"""

    chromosome: str
    start: int
    end: int
    score: float
    summit: int
    fold_enrichment: float
    p_value: float
    q_value: float
    nearest_gene: Optional[str] = None
    distance_to_tss: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "chromosome": self.chromosome,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "summit": self.summit,
            "fold_enrichment": self.fold_enrichment,
            "p_value": self.p_value,
            "q_value": self.q_value,
            "nearest_gene": self.nearest_gene,
            "distance_to_tss": self.distance_to_tss,
        }


@dataclass
class EpigeneticProfile:
    """Complete epigenetic profile for a sample"""

    sample_id: str
    data_type: EpigeneticDataType
    methylation_sites: Optional[List[MethylationSite]] = None
    chromatin_peaks: Optional[List[ChromatinPeak]] = None
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

    def get_methylation_by_region(
        self, chromosome: str, start: int, end: int
    ) -> List[MethylationSite]:
        """Get methylation sites in a specific region"""
        if not self.methylation_sites:
            return []

        return [
            site
            for site in self.methylation_sites
            if site.chromosome == chromosome and start <= site.position <= end
        ]

    def get_peaks_by_gene(self, gene_id: str) -> List[ChromatinPeak]:
        """Get chromatin peaks near a specific gene"""
        if not self.chromatin_peaks:
            return []

        return [peak for peak in self.chromatin_peaks if peak.nearest_gene == gene_id]

    def calculate_regional_methylation(
        self, chromosome: str, start: int, end: int
    ) -> Optional[float]:
        """Calculate average methylation in a region"""
        sites = self.get_methylation_by_region(chromosome, start, end)
        if not sites:
            return None

        weighted_sum = sum(site.methylation_level * site.coverage for site in sites)
        total_coverage = sum(site.coverage for site in sites)

        return weighted_sum / total_coverage if total_coverage > 0 else None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        if self.methylation_sites:
            data = [site.to_dict() for site in self.methylation_sites]
        elif self.chromatin_peaks:
            data = [peak.to_dict() for peak in self.chromatin_peaks]
        else:
            data = []

        return pd.DataFrame(data)


class MethylationProcessor:
    """Process DNA methylation data"""

    def __init__(
        self,
        reference_genome: Optional[Path] = None,
        annotation_file: Optional[Path] = None,
        min_coverage: int = 5,
        max_threads: int = 4,
    ):
        """
        Initialize methylation processor

        Args:
            reference_genome: Path to reference genome
            annotation_file: Path to gene annotation
            min_coverage: Minimum coverage for methylation calls
            max_threads: Maximum threads for processing
        """
        self.reference_genome = reference_genome
        self.annotation_file = annotation_file
        self.min_coverage = min_coverage
        self.max_threads = max_threads
        self.gene_annotations = self._load_annotations()

        logger.info("Initialized MethylationProcessor")

    def _load_annotations(self) -> Dict[str, Dict[str, Any]]:
        """Load gene annotations for region assignment"""
        if not self.annotation_file or not self.annotation_file.exists():
            logger.warning("No annotation file provided")
            return {}

        # In production, would parse GTF/GFF file
        # Mock annotations for now
        return {
            "ENSG00000141510": {
                "name": "TP53",
                "chr": "chr17",
                "start": 7565097,
                "end": 7590856,
                "promoter_start": 7564097,
                "promoter_end": 7566097,
            },
            "ENSG00000012048": {
                "name": "BRCA1",
                "chr": "chr17",
                "start": 41196312,
                "end": 41277500,
                "promoter_start": 41195312,
                "promoter_end": 41197312,
            },
        }

    def process(
        self,
        input_path: Path,
        sample_id: str,
        data_format: str = "bismark",
        context: MethylationContext = MethylationContext.CG,
    ) -> EpigeneticProfile:
        """
        Process methylation data

        Args:
            input_path: Path to methylation data file
            sample_id: Sample identifier
            data_format: Input format ('bismark', 'bedgraph', 'biscuit')
            context: Methylation context to analyze

        Returns:
            EpigeneticProfile with methylation data
        """
        logger.info(f"Processing methylation data for {sample_id}")

        try:
            # Load methylation data based on format
            if data_format == "bismark":
                methylation_data = self._load_bismark_output(input_path)
            elif data_format == "bedgraph":
                methylation_data = self._load_bedgraph(input_path)
            else:
                raise ValidationError(f"Unsupported format: {data_format}")

            # Filter by context and coverage
            filtered_data = self._filter_methylation_data(methylation_data, context)

            # Annotate with genomic regions
            annotated_sites = self._annotate_methylation_sites(filtered_data)

            # Calculate quality metrics
            quality_metrics = self._calculate_methylation_metrics(annotated_sites)

            # Perform quantile normalization
            normalized_sites = self._normalize_methylation(annotated_sites)

            # Create profile
            profile = EpigeneticProfile(
                sample_id=sample_id,
                data_type=EpigeneticDataType.METHYLATION,
                methylation_sites=normalized_sites,
                quality_metrics=quality_metrics,
                processing_metadata={
                    "processor_version": "1.0.0",
                    "processed_at": datetime.now().isoformat(),
                    "data_format": data_format,
                    "context": context.value,
                    "min_coverage": self.min_coverage,
                },
            )

            logger.info(f"Successfully processed {len(normalized_sites)} methylation sites")
            return profile

        except Exception as e:
            logger.error(f"Error processing methylation data: {str(e)}")
            raise ProcessingError(f"Failed to process methylation data: {str(e)}")

    def _load_bismark_output(self, file_path: Path) -> pd.DataFrame:
        """Load Bismark methylation extractor output"""
        logger.info(f"Loading Bismark output from {file_path}")

        # Mock data for demonstration
        # In production, would parse actual Bismark output
        np.random.seed(42)

        n_sites = 10000
        chromosomes = ["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"]

        data = {
            "chromosome": np.random.choice(chromosomes, n_sites),
            "position": np.random.randint(1, 250000000, n_sites),
            "strand": np.random.choice(["+", "-"], n_sites),
            "methylated": np.random.binomial(20, 0.7, n_sites),
            "unmethylated": np.random.binomial(20, 0.3, n_sites),
            "context": np.random.choice(["CG", "CHG", "CHH"], n_sites, p=[0.8, 0.15, 0.05]),
        }

        df = pd.DataFrame(data)
        df["coverage"] = df["methylated"] + df["unmethylated"]
        df["methylation_level"] = df["methylated"] / df["coverage"]

        return df

    def _load_bedgraph(self, file_path: Path) -> pd.DataFrame:
        """Load BedGraph format methylation data"""
        logger.info(f"Loading BedGraph from {file_path}")

        # In production, would parse actual BedGraph file
        # For now, generate mock data
        return self._load_bismark_output(file_path)  # Reuse mock data

    def _filter_methylation_data(
        self, data: pd.DataFrame, context: MethylationContext
    ) -> pd.DataFrame:
        """Filter methylation data by context and coverage"""
        filtered = data[data["coverage"] >= self.min_coverage].copy()

        if context != MethylationContext.ALL:
            filtered = filtered[filtered["context"] == context.value]

        logger.info(f"Filtered to {len(filtered)} sites with coverage >= {self.min_coverage}")
        return filtered

    def _annotate_methylation_sites(self, data: pd.DataFrame) -> List[MethylationSite]:
        """Annotate methylation sites with genomic regions"""
        sites = []

        for _, row in data.iterrows():
            # Find nearest gene and region type
            gene_id, region_type = self._find_genomic_region(row["chromosome"], row["position"])

            site = MethylationSite(
                chromosome=row["chromosome"],
                position=int(row["position"]),
                context=MethylationContext(row["context"]),
                methylation_level=float(row["methylation_level"]),
                coverage=int(row["coverage"]),
                strand=row["strand"],
                gene_id=gene_id,
                region_type=region_type,
            )
            sites.append(site)

        return sites

    def _find_genomic_region(
        self, chromosome: str, position: int
    ) -> Tuple[Optional[str], Optional[str]]:
        """Find gene and region type for a genomic position"""
        for gene_id, info in self.gene_annotations.items():
            if info["chr"] != chromosome:
                continue

            # Check if in promoter
            if info["promoter_start"] <= position <= info["promoter_end"]:
                return gene_id, "promoter"

            # Check if in gene body
            if info["start"] <= position <= info["end"]:
                return gene_id, "gene_body"

        return None, "intergenic"

    def _calculate_methylation_metrics(self, sites: List[MethylationSite]) -> Dict[str, Any]:
        """Calculate quality control metrics for methylation data"""
        if not sites:
            return {}

        methylation_levels = [s.methylation_level for s in sites]
        coverages = [s.coverage for s in sites]

        # Calculate global methylation statistics
        metrics = {
            "total_sites": len(sites),
            "mean_methylation": float(np.mean(methylation_levels)),
            "median_methylation": float(np.median(methylation_levels)),
            "std_methylation": float(np.std(methylation_levels)),
            "mean_coverage": float(np.mean(coverages)),
            "median_coverage": float(np.median(coverages)),
            "sites_by_context": {},
            "sites_by_region": {},
            "hypomethylated_sites": sum(1 for m in methylation_levels if m < 0.2),
            "hypermethylated_sites": sum(1 for m in methylation_levels if m > 0.8),
            "intermediate_sites": sum(1 for m in methylation_levels if 0.2 <= m <= 0.8),
        }

        # Count by context
        context_counts = defaultdict(int)
        for site in sites:
            context_counts[site.context.value] += 1
        metrics["sites_by_context"] = dict(context_counts)

        # Count by region
        region_counts = defaultdict(int)
        for site in sites:
            if site.region_type:
                region_counts[site.region_type] += 1
        metrics["sites_by_region"] = dict(region_counts)

        return metrics

    def _normalize_methylation(self, sites: List[MethylationSite]) -> List[MethylationSite]:
        """Perform beta-mixture quantile normalization"""
        if not sites:
            return sites

        # Extract methylation levels
        methylation_values = np.array([s.methylation_level for s in sites])

        # Apply beta-mixture model (simplified)
        # In production, would use proper beta-mixture modeling
        # For now, apply quantile normalization

        # Rank the values
        sorted_indices = np.argsort(methylation_values)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(methylation_values))

        # Calculate quantiles
        quantiles = (ranks + 0.5) / len(ranks)

        # Map to beta distribution quantiles
        from scipy.stats import beta

        a, b = 2, 5  # Beta distribution parameters
        normalized_values = beta.ppf(quantiles, a, b)

        # Create normalized sites
        normalized_sites = []
        for i, site in enumerate(sites):
            normalized_site = MethylationSite(
                chromosome=site.chromosome,
                position=site.position,
                context=site.context,
                methylation_level=float(normalized_values[i]),
                coverage=site.coverage,
                strand=site.strand,
                gene_id=site.gene_id,
                region_type=site.region_type,
            )
            normalized_sites.append(normalized_site)

        logger.info("Applied beta-mixture quantile normalization")
        return normalized_sites

    def differential_methylation(
        self,
        group1_profiles: List[EpigeneticProfile],
        group2_profiles: List[EpigeneticProfile],
        min_difference: float = 0.2,
        fdr_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """
        Identify differentially methylated regions

        Args:
            group1_profiles: Control group profiles
            group2_profiles: Treatment group profiles
            min_difference: Minimum methylation difference
            fdr_threshold: FDR threshold for significance

        Returns:
            DataFrame with differential methylation results
        """
        logger.info("Performing differential methylation analysis")

        # Collect all sites
        all_sites = set()
        for profile in group1_profiles + group2_profiles:
            if profile.methylation_sites:
                for site in profile.methylation_sites:
                    all_sites.add((site.chromosome, site.position))

        results = []

        for chr_pos in all_sites:
            chr, pos = chr_pos

            # Get methylation values for each group
            group1_values = []
            group2_values = []

            for profile in group1_profiles:
                sites = [
                    s
                    for s in profile.methylation_sites
                    if s.chromosome == chr and s.position == pos
                ]
                if sites:
                    group1_values.append(sites[0].methylation_level)

            for profile in group2_profiles:
                sites = [
                    s
                    for s in profile.methylation_sites
                    if s.chromosome == chr and s.position == pos
                ]
                if sites:
                    group2_values.append(sites[0].methylation_level)

            if len(group1_values) >= 2 and len(group2_values) >= 2:
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(group1_values, group2_values)

                mean1 = np.mean(group1_values)
                mean2 = np.mean(group2_values)
                diff = mean2 - mean1

                results.append(
                    {
                        "chromosome": chr,
                        "position": pos,
                        "mean_group1": mean1,
                        "mean_group2": mean2,
                        "methylation_difference": diff,
                        "p_value": p_value,
                        "t_statistic": t_stat,
                    }
                )

        if not results:
            logger.warning("No differential methylation sites found")
            return pd.DataFrame()

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Multiple testing correction
        from statsmodels.stats.multitest import multipletests

        _, fdr_values, _, _ = multipletests(results_df["p_value"], method="fdr_bh")
        results_df["fdr"] = fdr_values

        # Filter by significance and difference
        results_df["significant"] = (results_df["fdr"] < fdr_threshold) & (
            np.abs(results_df["methylation_difference"]) >= min_difference
        )

        # Sort by p-value
        results_df.sort_values("p_value", inplace=True)

        logger.info(f"Found {results_df['significant'].sum()} differentially methylated sites")

        return results_df


class ChromatinAccessibilityProcessor:
    """Process chromatin accessibility data (ATAC-seq)"""

    def __init__(
        self,
        reference_genome: Optional[Path] = None,
        annotation_file: Optional[Path] = None,
        peak_caller: str = "macs2",
        max_threads: int = 4,
    ):
        """
        Initialize chromatin accessibility processor

        Args:
            reference_genome: Path to reference genome
            annotation_file: Path to gene annotation
            peak_caller: Peak calling software to use
            max_threads: Maximum threads for processing
        """
        self.reference_genome = reference_genome
        self.annotation_file = annotation_file
        self.peak_caller = peak_caller
        self.max_threads = max_threads
        self.gene_annotations = self._load_annotations()

        logger.info("Initialized ChromatinAccessibilityProcessor")

    def _load_annotations(self) -> Dict[str, Dict[str, Any]]:
        """Load gene annotations"""
        # Reuse methylation processor's mock annotations
        return MethylationProcessor()._load_annotations()

    def process(
        self,
        input_path: Union[Path, List[Path]],
        sample_id: str,
        paired_end: bool = True,
        peak_format: str = "narrowPeak",
    ) -> EpigeneticProfile:
        """
        Process ATAC-seq data

        Args:
            input_path: Path to FASTQ or peak files
            sample_id: Sample identifier
            paired_end: Whether sequencing is paired-end
            peak_format: Peak file format

        Returns:
            EpigeneticProfile with chromatin accessibility data
        """
        logger.info(f"Processing ATAC-seq data for {sample_id}")

        try:
            # Detect input type
            if isinstance(input_path, Path) and input_path.suffix in [
                ".narrowPeak",
                ".broadPeak",
                ".bed",
            ]:
                # Load pre-called peaks
                peaks = self._load_peak_file(input_path, peak_format)
            else:
                # Process from FASTQ (mock implementation)
                peaks = self._process_fastq_to_peaks(input_path, paired_end)

            # Annotate peaks with nearest genes
            annotated_peaks = self._annotate_peaks(peaks)

            # Calculate quality metrics
            quality_metrics = self._calculate_peak_metrics(annotated_peaks)

            # Create profile
            profile = EpigeneticProfile(
                sample_id=sample_id,
                data_type=EpigeneticDataType.CHROMATIN_ACCESSIBILITY,
                chromatin_peaks=annotated_peaks,
                quality_metrics=quality_metrics,
                processing_metadata={
                    "processor_version": "1.0.0",
                    "processed_at": datetime.now().isoformat(),
                    "peak_caller": self.peak_caller,
                    "peak_format": peak_format,
                    "paired_end": paired_end,
                },
            )

            logger.info(f"Successfully processed {len(annotated_peaks)} chromatin peaks")
            return profile

        except Exception as e:
            logger.error(f"Error processing ATAC-seq data: {str(e)}")
            raise ProcessingError(f"Failed to process ATAC-seq data: {str(e)}")

    def _load_peak_file(self, file_path: Path, format: str) -> pd.DataFrame:
        """Load peak file"""
        logger.info(f"Loading peaks from {file_path}")

        if format == "narrowPeak":
            # Standard ENCODE narrowPeak format
            columns = [
                "chr",
                "start",
                "end",
                "name",
                "score",
                "strand",
                "signal_value",
                "p_value",
                "q_value",
                "peak",
            ]
            df = pd.read_csv(file_path, sep="\t", names=columns)
            df["summit"] = df["start"] + df["peak"]
            df["fold_enrichment"] = df["signal_value"]
        else:
            # Generate mock peaks for demonstration
            df = self._generate_mock_peaks()

        return df

    def _process_fastq_to_peaks(
        self, input_paths: Union[Path, List[Path]], paired_end: bool
    ) -> pd.DataFrame:
        """Process FASTQ files to peaks (mock implementation)"""
        logger.info("Processing FASTQ to peaks (mock implementation)")
        return self._generate_mock_peaks()

    def _generate_mock_peaks(self) -> pd.DataFrame:
        """Generate mock peak data for demonstration"""
        np.random.seed(42)

        n_peaks = 5000
        chromosomes = ["chr" + str(i) for i in range(1, 23)] + ["chrX"]

        data = []
        for i in range(n_peaks):
            chr = np.random.choice(chromosomes)
            start = np.random.randint(1000000, 200000000)
            length = np.random.randint(200, 2000)

            data.append(
                {
                    "chr": chr,
                    "start": start,
                    "end": start + length,
                    "score": np.random.uniform(50, 1000),
                    "summit": start + length // 2,
                    "fold_enrichment": np.random.uniform(2, 50),
                    "p_value": 10 ** -np.random.uniform(3, 20),
                    "q_value": 10 ** -np.random.uniform(2, 15),
                }
            )

        return pd.DataFrame(data)

    def _annotate_peaks(self, peaks_df: pd.DataFrame) -> List[ChromatinPeak]:
        """Annotate peaks with nearest genes"""
        peaks = []

        for _, row in peaks_df.iterrows():
            # Find nearest gene
            nearest_gene, distance = self._find_nearest_gene(row["chr"], row["summit"])

            peak = ChromatinPeak(
                chromosome=row["chr"],
                start=int(row["start"]),
                end=int(row["end"]),
                score=float(row["score"]),
                summit=int(row["summit"]),
                fold_enrichment=float(row["fold_enrichment"]),
                p_value=float(row["p_value"]),
                q_value=float(row["q_value"]),
                nearest_gene=nearest_gene,
                distance_to_tss=distance,
            )
            peaks.append(peak)

        return peaks

    def _find_nearest_gene(
        self, chromosome: str, position: int
    ) -> Tuple[Optional[str], Optional[int]]:
        """Find nearest gene and distance to TSS"""
        min_distance = float("inf")
        nearest_gene = None

        for gene_id, info in self.gene_annotations.items():
            if info["chr"] != chromosome:
                continue

            # Calculate distance to TSS (assuming + strand)
            tss = info["start"]
            distance = abs(position - tss)

            if distance < min_distance:
                min_distance = distance
                nearest_gene = gene_id

        return nearest_gene, int(min_distance) if nearest_gene else None

    def _calculate_peak_metrics(self, peaks: List[ChromatinPeak]) -> Dict[str, Any]:
        """Calculate quality metrics for peaks"""
        if not peaks:
            return {}

        scores = [p.score for p in peaks]
        enrichments = [p.fold_enrichment for p in peaks]
        distances = [p.distance_to_tss for p in peaks if p.distance_to_tss is not None]

        metrics = {
            "total_peaks": len(peaks),
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "mean_fold_enrichment": float(np.mean(enrichments)),
            "median_fold_enrichment": float(np.median(enrichments)),
            "peaks_near_tss": sum(1 for d in distances if d < 3000),
            "distal_peaks": sum(1 for d in distances if d > 50000),
            "mean_peak_width": float(np.mean([p.end - p.start for p in peaks])),
            "significant_peaks": sum(1 for p in peaks if p.q_value < 0.05),
            "highly_enriched_peaks": sum(1 for p in peaks if p.fold_enrichment > 10),
        }

        # Peak distribution by chromosome
        chr_counts = defaultdict(int)
        for peak in peaks:
            chr_counts[peak.chromosome] += 1
        metrics["peaks_by_chromosome"] = dict(chr_counts)

        return metrics

    def find_differential_peaks(
        self,
        group1_profiles: List[EpigeneticProfile],
        group2_profiles: List[EpigeneticProfile],
        min_fold_change: float = 2.0,
        fdr_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """
        Find differential chromatin accessibility

        Args:
            group1_profiles: Control group profiles
            group2_profiles: Treatment group profiles
            min_fold_change: Minimum fold change
            fdr_threshold: FDR threshold

        Returns:
            DataFrame with differential peaks
        """
        logger.info("Finding differential chromatin accessibility")

        # This is a simplified implementation
        # In production, would use DiffBind or similar tools

        # Collect all peak regions
        all_regions = set()
        for profile in group1_profiles + group2_profiles:
            if profile.chromatin_peaks:
                for peak in profile.chromatin_peaks:
                    all_regions.add((peak.chromosome, peak.start, peak.end))

        results = []

        # For each region, compare enrichment between groups
        for region in all_regions:
            chr, start, end = region

            # Get enrichment values for each group
            group1_enrichments = []
            group2_enrichments = []

            for profile in group1_profiles:
                peaks = [
                    p
                    for p in profile.chromatin_peaks
                    if p.chromosome == chr and p.start <= start <= p.end
                ]
                if peaks:
                    group1_enrichments.append(peaks[0].fold_enrichment)

            for profile in group2_profiles:
                peaks = [
                    p
                    for p in profile.chromatin_peaks
                    if p.chromosome == chr and p.start <= start <= p.end
                ]
                if peaks:
                    group2_enrichments.append(peaks[0].fold_enrichment)

            if group1_enrichments and group2_enrichments:
                # Calculate statistics
                mean1 = np.mean(group1_enrichments)
                mean2 = np.mean(group2_enrichments)
                fold_change = mean2 / mean1 if mean1 > 0 else 0

                # Simplified p-value calculation
                if len(group1_enrichments) >= 2 and len(group2_enrichments) >= 2:
                    _, p_value = stats.ttest_ind(
                        np.log2(group1_enrichments + 1), np.log2(group2_enrichments + 1)
                    )
                else:
                    p_value = 1.0

                results.append(
                    {
                        "chromosome": chr,
                        "start": start,
                        "end": end,
                        "mean_enrichment_group1": mean1,
                        "mean_enrichment_group2": mean2,
                        "fold_change": fold_change,
                        "log2_fold_change": np.log2(fold_change) if fold_change > 0 else 0,
                        "p_value": p_value,
                    }
                )

        if not results:
            logger.warning("No differential peaks found")
            return pd.DataFrame()

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Multiple testing correction
        from statsmodels.stats.multitest import multipletests

        _, fdr_values, _, _ = multipletests(results_df["p_value"], method="fdr_bh")
        results_df["fdr"] = fdr_values

        # Filter by significance and fold change
        results_df["significant"] = (results_df["fdr"] < fdr_threshold) & (
            np.abs(results_df["log2_fold_change"]) >= np.log2(min_fold_change)
        )

        # Sort by p-value
        results_df.sort_values("p_value", inplace=True)

        logger.info(f"Found {results_df['significant'].sum()} differential peaks")

        return results_df


def create_epigenetic_processor(
    data_type: EpigeneticDataType, **kwargs
) -> Union[MethylationProcessor, ChromatinAccessibilityProcessor]:
    """
    Factory function to create appropriate epigenetic processor

    Args:
        data_type: Type of epigenetic data
        **kwargs: Processor-specific arguments

    Returns:
        Appropriate processor instance
    """
    if data_type == EpigeneticDataType.METHYLATION:
        return MethylationProcessor(**kwargs)
    elif data_type == EpigeneticDataType.CHROMATIN_ACCESSIBILITY:
        return ChromatinAccessibilityProcessor(**kwargs)
    else:
        raise ValidationError(f"Unsupported epigenetic data type: {data_type}")
