"""
GenomeVault Hypervector Encoding

Implements hierarchical hyperdimensional computing (HDC) for privacy-preserving
representation of multi-omics data with similarity preservation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from scipy.spatial import distance
from scipy.stats import norm
import pickle

from ..utils import get_logger, get_config, secure_hash
from ..utils.logging import log_operation
from ..local_processing import (
    GenomicProfile, 
    ExpressionProfile,
    EpigeneticProfile,
    ProteomicsProfile,
    PhenotypeProfile
)

logger = get_logger(__name__)
config = get_config()


class VectorType(Enum):
    """Types of hypervectors"""
    BASE = "base"           # Individual features
    MID = "mid"             # Gene/pathway level
    HIGH = "high"           # System-wide patterns
    COMPOSITE = "composite" # Multi-modal binding


class ProjectionType(Enum):
    """Types of projection matrices"""
    RANDOM_GAUSSIAN = "random_gaussian"
    SPARSE_RANDOM = "sparse_random"
    LEARNED = "learned"
    ORTHOGONAL = "orthogonal"


@dataclass
class Hypervector:
    """Hyperdimensional vector representation"""
    vector_id: str
    vector_type: VectorType
    dimensions: int
    data: np.ndarray
    sparsity: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate hypervector after initialization"""
        if len(self.data) != self.dimensions:
            raise ValueError(f"Data length {len(self.data)} doesn't match dimensions {self.dimensions}")
        
        # Calculate sparsity if not provided
        if self.sparsity == 0:
            self.sparsity = 1.0 - (np.count_nonzero(self.data) / self.dimensions)
    
    @property
    def norm(self) -> float:
        """L2 norm of the vector"""
        return np.linalg.norm(self.data)
    
    def normalize(self) -> 'Hypervector':
        """Return normalized version of vector"""
        norm = self.norm
        if norm > 0:
            normalized_data = self.data / norm
        else:
            normalized_data = self.data
        
        return Hypervector(
            vector_id=f"{self.vector_id}_normalized",
            vector_type=self.vector_type,
            dimensions=self.dimensions,
            data=normalized_data,
            sparsity=self.sparsity,
            metadata={**self.metadata, 'normalized': True}
        )
    
    def to_binary(self) -> bytes:
        """Convert to compact binary representation"""
        # Header: type(1) + dimensions(4) + sparsity(4) + metadata_len(4)
        header = bytearray()
        header.append(self.vector_type.value.encode()[0])  # First char as type indicator
        header.extend(self.dimensions.to_bytes(4, 'little'))
        header.extend(int(self.sparsity * 10000).to_bytes(4, 'little'))  # Store as int
        
        # Metadata
        metadata_json = json.dumps(self.metadata).encode()
        header.extend(len(metadata_json).to_bytes(4, 'little'))
        
        # Data - use sparse representation if beneficial
        if self.sparsity > 0.8:  # Highly sparse
            # Store indices and values of non-zero elements
            non_zero_indices = np.nonzero(self.data)[0]
            non_zero_values = self.data[non_zero_indices]
            
            data_bytes = bytearray()
            data_bytes.extend(len(non_zero_indices).to_bytes(4, 'little'))
            data_bytes.extend(non_zero_indices.astype(np.uint32).tobytes())
            data_bytes.extend(non_zero_values.astype(np.float32).tobytes())
        else:
            # Store full vector
            data_bytes = bytearray()
            data_bytes.extend(b'\x00' * 4)  # 0 indicates full storage
            data_bytes.extend(self.data.astype(np.float32).tobytes())
        
        return bytes(header) + metadata_json + bytes(data_bytes)
    
    @classmethod
    def from_binary(cls, binary_data: bytes) -> 'Hypervector':
        """Reconstruct from binary representation"""
        # Parse header
        vector_type_char = chr(binary_data[0])
        type_map = {'b': VectorType.BASE, 'm': VectorType.MID, 'h': VectorType.HIGH, 'c': VectorType.COMPOSITE}
        vector_type = type_map.get(vector_type_char, VectorType.BASE)
        
        dimensions = int.from_bytes(binary_data[1:5], 'little')
        sparsity = int.from_bytes(binary_data[5:9], 'little') / 10000.0
        metadata_len = int.from_bytes(binary_data[9:13], 'little')
        
        # Parse metadata
        metadata_start = 13
        metadata_end = metadata_start + metadata_len
        metadata = json.loads(binary_data[metadata_start:metadata_end].decode())
        
        # Parse data
        data_start = metadata_end
        num_non_zero = int.from_bytes(binary_data[data_start:data_start+4], 'little')
        
        if num_non_zero == 0:  # Full storage
            data = np.frombuffer(binary_data[data_start+4:], dtype=np.float32)
        else:  # Sparse storage
            indices = np.frombuffer(
                binary_data[data_start+4:data_start+4+num_non_zero*4], 
                dtype=np.uint32
            )
            values = np.frombuffer(
                binary_data[data_start+4+num_non_zero*4:], 
                dtype=np.float32
            )
            data = np.zeros(dimensions, dtype=np.float32)
            data[indices] = values
        
        return cls(
            vector_id=metadata.get('vector_id', 'unknown'),
            vector_type=vector_type,
            dimensions=dimensions,
            data=data,
            sparsity=sparsity,
            metadata=metadata
        )


@dataclass
class ProjectionMatrix:
    """Projection matrix for hypervector transformation"""
    matrix_id: str
    input_dim: int
    output_dim: int
    projection_type: ProjectionType
    matrix: np.ndarray
    normalization_factor: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def project(self, input_vector: np.ndarray) -> np.ndarray:
        """Project input vector to hyperdimensional space"""
        if len(input_vector) != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {len(input_vector)}")
        
        # Perform projection
        projected = self.matrix @ input_vector
        
        # Apply normalization
        if self.normalization_factor != 1.0:
            projected = projected * self.normalization_factor
        
        return projected
    
    def save(self, path: str):
        """Save projection matrix to file"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'ProjectionMatrix':
        """Load projection matrix from file"""
        with open(path, 'rb') as f:
            return pickle.load(f)


class HypervectorEncoder:
    """Main encoder for transforming multi-omics data to hypervectors"""
    
    # Default dimensions for different vector types
    DEFAULT_DIMENSIONS = {
        VectorType.BASE: 10000,
        VectorType.MID: 15000,
        VectorType.HIGH: 20000,
        VectorType.COMPOSITE: 10000
    }
    
    # Domain-specific projection keys
    DOMAIN_PROJECTIONS = {
        'oncology': 'cancer_optimized',
        'rare_disease': 'rare_variant_sensitive',
        'population': 'ancestry_preserving',
        'pharmacogenomics': 'drug_response_focused'
    }
    
    def __init__(self, 
                 base_dim: Optional[int] = None,
                 projection_type: ProjectionType = ProjectionType.SPARSE_RANDOM,
                 seed: int = 42):
        """
        Initialize hypervector encoder
        
        Args:
            base_dim: Base hypervector dimensions
            projection_type: Type of projection to use
            seed: Random seed for reproducibility
        """
        self.base_dim = base_dim or self.DEFAULT_DIMENSIONS[VectorType.BASE]
        self.projection_type = projection_type
        self.seed = seed
        
        # Set random seed
        np.random.seed(seed)
        
        # Cache for projection matrices
        self._projection_cache: Dict[str, ProjectionMatrix] = {}
        
        # Initialize base projections
        self._initialize_projections()
        
        logger.info(f"Initialized HypervectorEncoder with {self.base_dim} base dimensions")
    
    def _initialize_projections(self):
        """Initialize default projection matrices"""
        # Genomic variant projection
        self._projection_cache['genomic_variant'] = self._create_projection_matrix(
            input_dim=100,  # Features per variant
            output_dim=self.base_dim,
            matrix_id='genomic_variant_base'
        )
        
        # Gene expression projection
        self._projection_cache['gene_expression'] = self._create_projection_matrix(
            input_dim=1000,  # Expression features
            output_dim=self.base_dim,
            matrix_id='expression_base'
        )
        
        # Methylation projection
        self._projection_cache['methylation'] = self._create_projection_matrix(
            input_dim=500,  # Methylation features
            output_dim=self.base_dim,
            matrix_id='methylation_base'
        )
        
        # Protein projection
        self._projection_cache['protein'] = self._create_projection_matrix(
            input_dim=200,  # Protein features
            output_dim=self.base_dim,
            matrix_id='protein_base'
        )
        
        # Phenotype projection
        self._projection_cache['phenotype'] = self._create_projection_matrix(
            input_dim=50,  # Clinical features
            output_dim=self.base_dim,
            matrix_id='phenotype_base'
        )
    
    def _create_projection_matrix(self, 
                                 input_dim: int, 
                                 output_dim: int,
                                 matrix_id: str) -> ProjectionMatrix:
        """Create projection matrix based on type"""
        if self.projection_type == ProjectionType.RANDOM_GAUSSIAN:
            # Random Gaussian projection
            matrix = np.random.randn(output_dim, input_dim) / np.sqrt(input_dim)
            
        elif self.projection_type == ProjectionType.SPARSE_RANDOM:
            # Sparse random projection (faster, memory efficient)
            # Each element is 0 with prob 2/3, ±1 with prob 1/6 each
            s = np.sqrt(3)  # Scaling factor
            matrix = np.random.choice(
                [0, s, -s], 
                size=(output_dim, input_dim),
                p=[2/3, 1/6, 1/6]
            ) / np.sqrt(input_dim)
            
        elif self.projection_type == ProjectionType.ORTHOGONAL:
            # Orthogonal random projection
            # Generate random matrix and orthogonalize
            matrix = np.random.randn(output_dim, input_dim)
            q, r = np.linalg.qr(matrix.T)
            matrix = q.T[:output_dim]
            
        else:
            raise ValueError(f"Unknown projection type: {self.projection_type}")
        
        return ProjectionMatrix(
            matrix_id=matrix_id,
            input_dim=input_dim,
            output_dim=output_dim,
            projection_type=self.projection_type,
            matrix=matrix,
            normalization_factor=1.0,
            metadata={'seed': self.seed}
        )
    
    @log_operation("encode_genomic_profile")
    def encode_genomic(self, profile: GenomicProfile) -> Dict[str, Hypervector]:
        """
        Encode genomic profile into hypervectors
        
        Args:
            profile: GenomicProfile from sequencing processor
            
        Returns:
            Dictionary of hypervectors at different abstraction levels
        """
        logger.info(f"Encoding genomic profile {profile.sample_id} with {len(profile.variants)} variants")
        
        hypervectors = {}
        
        # 1. Base-level: Individual variant encoding
        variant_vectors = []
        for variant in profile.variants:
            features = self._extract_variant_features(variant)
            projection = self._projection_cache['genomic_variant']
            variant_vector = projection.project(features)
            variant_vectors.append(variant_vector)
        
        # Aggregate variant vectors
        if variant_vectors:
            base_vector = np.mean(variant_vectors, axis=0)
        else:
            base_vector = np.zeros(self.base_dim)
        
        hypervectors['genomic_base'] = Hypervector(
            vector_id=f"{profile.sample_id}_genomic_base",
            vector_type=VectorType.BASE,
            dimensions=self.base_dim,
            data=base_vector,
            sparsity=0.0,
            metadata={
                'variant_count': len(profile.variants),
                'reference_genome': profile.reference_genome
            }
        )
        
        # 2. Mid-level: Gene-level aggregation
        gene_vectors = self._aggregate_by_genes(profile.variants)
        mid_dim = self.DEFAULT_DIMENSIONS[VectorType.MID]
        mid_projection = self._create_projection_matrix(
            input_dim=self.base_dim,
            output_dim=mid_dim,
            matrix_id='genomic_mid'
        )
        
        mid_vector = mid_projection.project(base_vector)
        
        hypervectors['genomic_mid'] = Hypervector(
            vector_id=f"{profile.sample_id}_genomic_mid",
            vector_type=VectorType.MID,
            dimensions=mid_dim,
            data=mid_vector,
            sparsity=0.0,
            metadata={'gene_count': len(gene_vectors)}
        )
        
        # 3. High-level: System-wide patterns
        high_dim = self.DEFAULT_DIMENSIONS[VectorType.HIGH]
        high_features = self._extract_system_features(profile)
        high_projection = self._create_projection_matrix(
            input_dim=len(high_features),
            output_dim=high_dim,
            matrix_id='genomic_high'
        )
        
        high_vector = high_projection.project(high_features)
        
        hypervectors['genomic_high'] = Hypervector(
            vector_id=f"{profile.sample_id}_genomic_high",
            vector_type=VectorType.HIGH,
            dimensions=high_dim,
            data=high_vector,
            sparsity=0.0,
            metadata={'quality_metrics': profile.quality_metrics.__dict__}
        )
        
        logger.info(f"Generated {len(hypervectors)} hypervectors for genomic profile")
        return hypervectors
    
    def _extract_variant_features(self, variant) -> np.ndarray:
        """Extract numerical features from a variant"""
        features = []
        
        # Chromosome encoding (one-hot for chr1-22, X, Y, M)
        chr_features = np.zeros(25)
        chr_map = {f'chr{i}': i-1 for i in range(1, 23)}
        chr_map.update({'chrX': 22, 'chrY': 23, 'chrM': 24})
        if variant.chromosome in chr_map:
            chr_features[chr_map[variant.chromosome]] = 1
        features.extend(chr_features)
        
        # Position encoding (normalized by chromosome length)
        # Approximate chromosome lengths
        chr_lengths = {
            'chr1': 249e6, 'chr2': 242e6, 'chr3': 198e6, 'chr4': 190e6,
            'chr5': 182e6, 'chr6': 171e6, 'chr7': 159e6, 'chr8': 145e6,
            'chr9': 138e6, 'chr10': 133e6, 'chr11': 135e6, 'chr12': 133e6,
            'chr13': 114e6, 'chr14': 107e6, 'chr15': 102e6, 'chr16': 90e6,
            'chr17': 83e6, 'chr18': 80e6, 'chr19': 59e6, 'chr20': 64e6,
            'chr21': 47e6, 'chr22': 51e6, 'chrX': 155e6, 'chrY': 59e6, 'chrM': 16569
        }
        
        chr_len = chr_lengths.get(variant.chromosome, 250e6)
        normalized_pos = variant.position / chr_len
        features.append(normalized_pos)
        
        # Variant type encoding
        variant_type = 'SNP'
        if len(variant.reference) > len(variant.alternate):
            variant_type = 'DEL'
        elif len(variant.reference) < len(variant.alternate):
            variant_type = 'INS'
        elif len(variant.reference) > 1:
            variant_type = 'COMPLEX'
        
        type_features = np.zeros(4)
        type_map = {'SNP': 0, 'INS': 1, 'DEL': 2, 'COMPLEX': 3}
        type_features[type_map[variant_type]] = 1
        features.extend(type_features)
        
        # Quality and depth
        features.append(variant.quality / 100.0)  # Normalized quality
        features.append(min(variant.depth / 100.0, 1.0))  # Normalized depth
        
        # Genotype encoding
        gt_features = np.zeros(3)
        if variant.genotype == '0/0':
            gt_features[0] = 1
        elif variant.genotype == '0/1' or variant.genotype == '1/0':
            gt_features[1] = 1
        elif variant.genotype == '1/1':
            gt_features[2] = 1
        features.extend(gt_features)
        
        # Allele frequency
        features.append(variant.allele_frequency)
        
        # Pad to fixed size
        features = np.array(features)
        if len(features) < 100:
            features = np.pad(features, (0, 100 - len(features)))
        
        return features[:100]
    
    def _aggregate_by_genes(self, variants) -> Dict[str, np.ndarray]:
        """Aggregate variants by genes (placeholder)"""
        # In production, would map variants to genes using annotation
        gene_vectors = {}
        
        # Simple simulation: group by chromosome regions
        for variant in variants:
            # Simulate gene assignment
            gene_region = f"{variant.chromosome}_region_{variant.position // 1000000}"
            if gene_region not in gene_vectors:
                gene_vectors[gene_region] = []
            
            features = self._extract_variant_features(variant)
            gene_vectors[gene_region].append(features)
        
        # Average vectors per gene
        for gene, vectors in gene_vectors.items():
            gene_vectors[gene] = np.mean(vectors, axis=0)
        
        return gene_vectors
    
    def _extract_system_features(self, profile: GenomicProfile) -> np.ndarray:
        """Extract system-wide features from genomic profile"""
        features = []
        
        # Variant statistics
        total_variants = len(profile.variants)
        features.append(np.log1p(total_variants))  # Log-scaled variant count
        
        # Variant type distribution
        snp_count = sum(1 for v in profile.variants 
                       if len(v.reference) == 1 and len(v.alternate) == 1)
        indel_count = total_variants - snp_count
        features.extend([
            snp_count / (total_variants + 1),
            indel_count / (total_variants + 1)
        ])
        
        # Quality metrics
        if profile.quality_metrics:
            features.extend([
                profile.quality_metrics.coverage_mean / 100.0,
                profile.quality_metrics.coverage_std / 100.0,
                profile.quality_metrics.coverage_uniformity,
                profile.quality_metrics.q30_bases / (profile.quality_metrics.total_bases + 1),
                profile.quality_metrics.gc_content
            ])
        else:
            features.extend([0.0] * 5)
        
        # Chromosome distribution
        chr_counts = np.zeros(25)
        for variant in profile.variants:
            chr_map = {f'chr{i}': i-1 for i in range(1, 23)}
            chr_map.update({'chrX': 22, 'chrY': 23, 'chrM': 24})
            if variant.chromosome in chr_map:
                chr_counts[chr_map[variant.chromosome]] += 1
        
        chr_distribution = chr_counts / (total_variants + 1)
        features.extend(chr_distribution)
        
        return np.array(features)
    
    @log_operation("encode_expression_profile")
    def encode_expression(self, profile: ExpressionProfile) -> Dict[str, Hypervector]:
        """Encode gene expression profile into hypervectors"""
        logger.info(f"Encoding expression profile {profile.sample_id}")
        
        hypervectors = {}
        
        # Extract expression features
        expression_features = []
        gene_names = []
        
        for expr in profile.expressions:
            if expr.tpm > 0:  # Only include expressed genes
                expression_features.append([
                    np.log1p(expr.tpm),  # Log-transformed TPM
                    np.log1p(expr.raw_count),
                    expr.normalized_count / 1000.0,
                    1.0 if expr.biotype == 'protein_coding' else 0.0
                ])
                gene_names.append(expr.gene_name)
        
        if expression_features:
            # Flatten features
            features_array = np.array(expression_features).flatten()
            
            # Pad or truncate to fixed size
            if len(features_array) < 1000:
                features_array = np.pad(features_array, (0, 1000 - len(features_array)))
            else:
                features_array = features_array[:1000]
            
            # Project to hypervector space
            projection = self._projection_cache['gene_expression']
            base_vector = projection.project(features_array)
        else:
            base_vector = np.zeros(self.base_dim)
        
        hypervectors['expression_base'] = Hypervector(
            vector_id=f"{profile.sample_id}_expression_base",
            vector_type=VectorType.BASE,
            dimensions=self.base_dim,
            data=base_vector,
            sparsity=0.0,
            metadata={
                'total_genes': len(profile.expressions),
                'expressed_genes': len(expression_features)
            }
        )
        
        return hypervectors
    
    @log_operation("encode_epigenetic_profile")
    def encode_epigenetic(self, profile: EpigeneticProfile) -> Dict[str, Hypervector]:
        """Encode epigenetic profile into hypervectors"""
        logger.info(f"Encoding epigenetic profile {profile.sample_id}")
        
        hypervectors = {}
        
        if profile.methylation_sites:
            # Methylation encoding
            methylation_features = self._extract_methylation_features(profile.methylation_sites)
            projection = self._projection_cache['methylation']
            methylation_vector = projection.project(methylation_features)
            
            hypervectors['methylation_base'] = Hypervector(
                vector_id=f"{profile.sample_id}_methylation_base",
                vector_type=VectorType.BASE,
                dimensions=self.base_dim,
                data=methylation_vector,
                sparsity=0.0,
                metadata={
                    'total_sites': len(profile.methylation_sites),
                    'mean_methylation': profile.global_metrics.get('CG_mean', 0.0)
                }
            )
        
        if profile.chromatin_peaks:
            # Chromatin accessibility encoding
            chromatin_features = self._extract_chromatin_features(profile.chromatin_peaks)
            # Reuse methylation projection for now
            projection = self._projection_cache['methylation']
            chromatin_vector = projection.project(chromatin_features)
            
            hypervectors['chromatin_base'] = Hypervector(
                vector_id=f"{profile.sample_id}_chromatin_base",
                vector_type=VectorType.BASE,
                dimensions=self.base_dim,
                data=chromatin_vector,
                sparsity=0.0,
                metadata={
                    'total_peaks': len(profile.chromatin_peaks)
                }
            )
        
        return hypervectors
    
    def _extract_methylation_features(self, methylation_sites) -> np.ndarray:
        """Extract features from methylation sites"""
        # Aggregate methylation by genomic regions
        region_size = 1000000  # 1Mb regions
        region_methylation = {}
        
        for site in methylation_sites:
            region_key = f"{site.chromosome}_{site.position // region_size}"
            if region_key not in region_methylation:
                region_methylation[region_key] = []
            region_methylation[region_key].append(site.beta_value)
        
        # Calculate region statistics
        features = []
        for region, values in sorted(region_methylation.items())[:500]:  # Top 500 regions
            if values:
                features.extend([
                    np.mean(values),  # Mean methylation
                    np.std(values),   # Methylation variability
                    len(values) / 1000.0  # Site density
                ])
        
        # Pad to fixed size
        features = np.array(features)
        if len(features) < 500:
            features = np.pad(features, (0, 500 - len(features)))
        else:
            features = features[:500]
        
        return features
    
    def _extract_chromatin_features(self, chromatin_peaks) -> np.ndarray:
        """Extract features from chromatin peaks"""
        features = []
        
        # Peak statistics
        peak_scores = [p.peak_score for p in chromatin_peaks]
        fold_enrichments = [p.fold_enrichment for p in chromatin_peaks]
        
        features.extend([
            len(chromatin_peaks) / 10000.0,  # Normalized peak count
            np.mean(peak_scores) if peak_scores else 0.0,
            np.std(peak_scores) if peak_scores else 0.0,
            np.mean(fold_enrichments) if fold_enrichments else 0.0,
            np.std(fold_enrichments) if fold_enrichments else 0.0
        ])
        
        # Peak length distribution
        peak_lengths = [p.length for p in chromatin_peaks]
        if peak_lengths:
            features.extend(np.percentile(peak_lengths, [10, 25, 50, 75, 90]) / 10000.0)
        else:
            features.extend([0.0] * 5)
        
        # Pad to fixed size
        features = np.array(features)
        if len(features) < 500:
            features = np.pad(features, (0, 500 - len(features)))
        
        return features[:500]
    
    @log_operation("encode_proteomics_profile")
    def encode_proteomics(self, profile: ProteomicsProfile) -> Dict[str, Hypervector]:
        """Encode proteomics profile into hypervectors"""
        logger.info(f"Encoding proteomics profile {profile.sample_id}")
        
        hypervectors = {}
        
        # Extract protein features
        protein_features = []
        
        for protein in profile.proteins[:200]:  # Top 200 proteins
            protein_features.extend([
                protein.abundance,  # Already log-transformed
                protein.sequence_coverage / 100.0,
                protein.peptide_count / 50.0,
                len(protein.modifications) / 10.0
            ])
        
        # Pad to fixed size
        protein_features = np.array(protein_features)
        if len(protein_features) < 200:
            protein_features = np.pad(protein_features, (0, 200 - len(protein_features)))
        else:
            protein_features = protein_features[:200]
        
        # Project to hypervector
        projection = self._projection_cache['protein']
        protein_vector = projection.project(protein_features)
        
        hypervectors['protein_base'] = Hypervector(
            vector_id=f"{profile.sample_id}_protein_base",
            vector_type=VectorType.BASE,
            dimensions=self.base_dim,
            data=protein_vector,
            sparsity=0.0,
            metadata={
                'total_proteins': profile.total_proteins,
                'quantification_method': profile.quantification_method
            }
        )
        
        return hypervectors
    
    @log_operation("encode_phenotype_profile")
    def encode_phenotype(self, profile: PhenotypeProfile) -> Dict[str, Hypervector]:
        """Encode phenotype profile into hypervectors"""
        logger.info(f"Encoding phenotype profile {profile.sample_id}")
        
        hypervectors = {}
        
        # Extract phenotype features
        features = []
        
        # Demographics
        if 'age' in profile.demographics:
            features.append(profile.demographics['age'] / 100.0)
        else:
            features.append(0.0)
        
        if 'gender' in profile.demographics:
            features.append(1.0 if profile.demographics['gender'] == 'male' else 0.0)
        else:
            features.append(0.5)
        
        # Risk factors
        risk_factors = profile.calculate_risk_factors()
        for risk_type in ['age', 'bmi', 'smoking', 'family_history']:
            features.append(risk_factors.get(risk_type, 0.0))
        
        # Diagnosis encoding (presence/absence of major conditions)
        condition_categories = ['diabetes', 'hypertension', 'heart_disease', 'cancer']
        active_conditions = {d.metadata.get('category') if d.metadata else None 
                           for d in profile.get_active_conditions()}
        
        for category in condition_categories:
            features.append(1.0 if category in active_conditions else 0.0)
        
        # Medication count (normalized)
        features.append(len(profile.get_active_medications()) / 20.0)
        
        # Lab values (simplified - in production would be more comprehensive)
        lab_values = {}
        for measurement in profile.measurements:
            if measurement.measurement_type not in lab_values:
                lab_values[measurement.measurement_type] = []
            if isinstance(measurement.value, (int, float)):
                lab_values[measurement.measurement_type].append(measurement.value)
        
        # Common lab tests
        common_labs = ['glucose', 'hemoglobin_a1c', 'cholesterol_total', 'creatinine']
        for lab in common_labs:
            if lab in lab_values and lab_values[lab]:
                # Use most recent value, normalized
                value = lab_values[lab][-1]
                normalized_value = self._normalize_lab_value(lab, value)
                features.append(normalized_value)
            else:
                features.append(0.5)  # Default to middle of range
        
        # Pad to fixed size
        features = np.array(features)
        if len(features) < 50:
            features = np.pad(features, (0, 50 - len(features)))
        else:
            features = features[:50]
        
        # Project to hypervector
        projection = self._projection_cache['phenotype']
        phenotype_vector = projection.project(features)
        
        hypervectors['phenotype_base'] = Hypervector(
            vector_id=f"{profile.sample_id}_phenotype_base",
            vector_type=VectorType.BASE,
            dimensions=self.base_dim,
            data=phenotype_vector,
            sparsity=0.0,
            metadata={
                'measurement_count': len(profile.measurements),
                'diagnosis_count': len(profile.diagnoses),
                'medication_count': len(profile.medications)
            }
        )
        
        return hypervectors
    
    def _normalize_lab_value(self, lab_type: str, value: float) -> float:
        """Normalize lab values to 0-1 range"""
        # Reference ranges (simplified)
        ranges = {
            'glucose': (70, 140),
            'hemoglobin_a1c': (4, 7),
            'cholesterol_total': (100, 300),
            'creatinine': (0.5, 1.5)
        }
        
        if lab_type in ranges:
            min_val, max_val = ranges[lab_type]
            normalized = (value - min_val) / (max_val - min_val)
            return max(0.0, min(1.0, normalized))
        
        return 0.5  # Default
