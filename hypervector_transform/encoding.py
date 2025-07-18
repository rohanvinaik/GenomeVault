"""
Hyperdimensional vector encoding of multi-omics features.
Implements hierarchical encoding with privacy-preserving transformations.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import hashlib
import pickle
from pathlib import Path
import numba
from scipy.sparse import csr_matrix
import json

from utils.config import config, CompressionTier
from utils.logging import logger, performance_logger


@dataclass
class HypervectorMetadata:
    """Metadata for hypervector encoding."""
    dimensions: int
    sparsity: float
    projection_version: str
    domain_type: str
    compression_tier: CompressionTier
    feature_count: int
    
    def to_dict(self) -> Dict:
        return {
            'dimensions': self.dimensions,
            'sparsity': self.sparsity,
            'projection_version': self.projection_version,
            'domain_type': self.domain_type,
            'compression_tier': self.compression_tier.value,
            'feature_count': self.feature_count
        }


class HypervectorEncoder:
    """
    Encode multi-omics features into high-dimensional vectors.
    Implements hierarchical encoding with similarity preservation.
    """
    
    def __init__(self, 
                 base_dim: int = None,
                 compression_tier: CompressionTier = None,
                 use_sparse: bool = True):
        """
        Initialize hypervector encoder.
        
        Args:
            base_dim: Base dimension for hypervectors
            compression_tier: Compression tier to use
            use_sparse: Use sparse representations for efficiency
        """
        self.base_dim = base_dim or config.hypervector.base_dimensions
        self.mid_dim = config.hypervector.mid_dimensions
        self.high_dim = config.hypervector.high_dimensions
        self.compression_tier = compression_tier or config.hypervector.compression_tier
        self.use_sparse = use_sparse
        
        # Initialize projection matrices
        self._initialize_projections()
        
        # Feature mappings for different tiers
        self._initialize_feature_mappings()
        
        logger.info(f"HypervectorEncoder initialized with {self.base_dim} dimensions", 
                   extra={'privacy_safe': True})
    
    def _initialize_projections(self):
        """Initialize random projection matrices."""
        # Use deterministic seed for reproducibility
        np.random.seed(42)
        
        # Domain-specific projection matrices
        self.projections = {
            'genomic': self._create_projection_matrix(1000000, self.base_dim),
            'transcriptomic': self._create_projection_matrix(50000, self.base_dim),
            'epigenetic': self._create_projection_matrix(100000, self.base_dim),
            'proteomic': self._create_projection_matrix(20000, self.base_dim),
            'clinical': self._create_projection_matrix(10000, self.base_dim)
        }
        
        # Hierarchical projections
        self.hierarchy_projections = {
            'base_to_mid': self._create_projection_matrix(self.base_dim, self.mid_dim),
            'mid_to_high': self._create_projection_matrix(self.mid_dim, self.high_dim)
        }
    
    def _create_projection_matrix(self, input_dim: int, output_dim: int) -> np.ndarray:
        """
        Create a random projection matrix with good properties.
        Uses sparse random projections for efficiency.
        """
        if self.use_sparse:
            # Sparse random projection (Achlioptas, 2003)
            # Elements are {-1, 0, 1} with probabilities {1/6, 2/3, 1/6}
            sparsity = 2/3
            data = []
            rows = []
            cols = []
            
            for i in range(output_dim):
                # Number of non-zero elements per row
                nnz = int(input_dim * (1 - sparsity))
                # Random positions
                positions = np.random.choice(input_dim, nnz, replace=False)
                # Random values (-1 or 1)
                values = np.random.choice([-1, 1], nnz) * np.sqrt(3)
                
                for j, pos in enumerate(positions):
                    data.append(values[j])
                    rows.append(i)
                    cols.append(pos)
            
            projection = csr_matrix((data, (rows, cols)), shape=(output_dim, input_dim))
            return projection / np.sqrt(input_dim)
        else:
            # Dense random projection
            projection = np.random.randn(output_dim, input_dim)
            return projection / np.sqrt(input_dim)
    
    def _initialize_feature_mappings(self):
        """Initialize feature mappings for different compression tiers."""
        self.feature_mappings = {
            CompressionTier.MINI: self._load_mini_features(),
            CompressionTier.CLINICAL: self._load_clinical_features(),
            CompressionTier.FULL_HDC: None  # All features for full tier
        }
    
    def _load_mini_features(self) -> Dict[str, List[str]]:
        """Load ~5,000 most-studied SNPs for mini tier."""
        # In production, would load from curated database
        return {
            'genomic': [
                # Common disease-associated SNPs
                'rs1815739',  # ACTN3 (athletic performance)
                'rs9939609',  # FTO (obesity)
                'rs1801133',  # MTHFR (folate metabolism)
                'rs6265',     # BDNF (neuroplasticity)
                'rs4680',     # COMT (dopamine metabolism)
                # ... ~5000 total
            ]
        }
    
    def _load_clinical_features(self) -> Dict[str, List[str]]:
        """Load ACMG + PharmGKB variants for clinical tier."""
        # In production, would load from clinical databases
        return {
            'genomic': [
                # ACMG secondary findings genes
                'BRCA1', 'BRCA2', 'MLH1', 'MSH2', 'MSH6',
                'APC', 'MUTYH', 'VHL', 'MEN1', 'RET',
                # PharmGKB variants
                'CYP2C19*2', 'CYP2C19*3', 'CYP2C19*17',
                'CYP2D6*4', 'CYP2D6*10',
                'VKORC1', 'TPMT*2', 'TPMT*3A',
                # ... ~120,000 total
            ]
        }
    
    @performance_logger.log_operation("encode_features")
    def encode_features(self, features: Dict[str, Union[np.ndarray, List]], 
                       domain: str = 'genomic') -> np.ndarray:
        """
        Encode features into hyperdimensional vector.
        
        Args:
            features: Dictionary of feature arrays
            domain: Domain type for specialized encoding
            
        Returns:
            Encoded hypervector
        """
        # Select features based on compression tier
        if self.compression_tier != CompressionTier.FULL_HDC:
            features = self._filter_features(features, domain)
        
        # Convert features to vector
        feature_vector = self._features_to_vector(features, domain)
        
        # Apply projection
        if domain in self.projections:
            projection = self.projections[domain]
            if self.use_sparse:
                hypervector = projection.dot(feature_vector)
            else:
                hypervector = np.dot(projection, feature_vector)
        else:
            raise ValueError(f"Unknown domain: {domain}")
        
        # Apply non-linearity
        hypervector = self._apply_nonlinearity(hypervector)
        
        # Normalize
        hypervector = self._normalize(hypervector)
        
        return hypervector
    
    def _filter_features(self, features: Dict, domain: str) -> Dict:
        """Filter features based on compression tier."""
        if self.feature_mappings[self.compression_tier] is None:
            return features
        
        allowed_features = self.feature_mappings[self.compression_tier].get(domain, [])
        filtered = {}
        
        for key, values in features.items():
            if key in allowed_features:
                filtered[key] = values
        
        return filtered
    
    def _features_to_vector(self, features: Dict, domain: str) -> np.ndarray:
        """Convert feature dictionary to numerical vector."""
        if domain == 'genomic':
            return self._genomic_features_to_vector(features)
        elif domain == 'transcriptomic':
            return self._transcriptomic_features_to_vector(features)
        elif domain == 'epigenetic':
            return self._epigenetic_features_to_vector(features)
        elif domain == 'proteomic':
            return self._proteomic_features_to_vector(features)
        elif domain == 'clinical':
            return self._clinical_features_to_vector(features)
        else:
            raise ValueError(f"Unknown domain: {domain}")
    
    def _genomic_features_to_vector(self, features: Dict) -> np.ndarray:
        """Convert genomic features to vector."""
        # Initialize vector based on compression tier
        if self.compression_tier == CompressionTier.MINI:
            vector_size = 5000
        elif self.compression_tier == CompressionTier.CLINICAL:
            vector_size = 120000
        else:
            vector_size = 1000000
        
        vector = np.zeros(vector_size)
        
        # Encode variants
        if 'variants' in features:
            for i, variant in enumerate(features['variants']):
                if i >= vector_size:
                    break
                # Encode variant presence and zygosity
                if variant.get('genotype') == '0/1' or variant.get('genotype') == '0|1':
                    vector[i] = 1  # Heterozygous
                elif variant.get('genotype') == '1/1' or variant.get('genotype') == '1|1':
                    vector[i] = 2  # Homozygous
        
        return vector
    
    def _transcriptomic_features_to_vector(self, features: Dict) -> np.ndarray:
        """Convert transcriptomic features to vector."""
        vector_size = 50000
        vector = np.zeros(vector_size)
        
        # Encode expression values
        if 'expression' in features:
            expr_values = features['expression']
            # Log-transform and normalize
            expr_values = np.log2(expr_values + 1)
            expr_values = (expr_values - np.mean(expr_values)) / (np.std(expr_values) + 1e-8)
            
            min_len = min(len(expr_values), vector_size)
            vector[:min_len] = expr_values[:min_len]
        
        return vector
    
    def _epigenetic_features_to_vector(self, features: Dict) -> np.ndarray:
        """Convert epigenetic features to vector."""
        vector_size = 100000
        vector = np.zeros(vector_size)
        
        # Encode methylation values
        if 'methylation' in features:
            meth_values = features['methylation']
            # Beta values are already 0-1
            min_len = min(len(meth_values), vector_size)
            vector[:min_len] = meth_values[:min_len]
        
        return vector
    
    def _proteomic_features_to_vector(self, features: Dict) -> np.ndarray:
        """Convert proteomic features to vector."""
        vector_size = 20000
        vector = np.zeros(vector_size)
        
        # Encode protein abundances
        if 'proteins' in features:
            protein_values = features['proteins']
            # Log-transform
            protein_values = np.log2(protein_values + 1)
            
            min_len = min(len(protein_values), vector_size)
            vector[:min_len] = protein_values[:min_len]
        
        return vector
    
    def _clinical_features_to_vector(self, features: Dict) -> np.ndarray:
        """Convert clinical features to vector."""
        vector_size = 10000
        vector = np.zeros(vector_size)
        
        # Encode various clinical features
        idx = 0
        
        # Continuous values
        if 'labs' in features:
            for lab_value in features['labs'].values():
                if idx >= vector_size:
                    break
                vector[idx] = float(lab_value)
                idx += 1
        
        # Categorical values (one-hot encode)
        if 'diagnoses' in features:
            for diagnosis in features['diagnoses']:
                if idx >= vector_size:
                    break
                vector[idx] = 1
                idx += 1
        
        return vector
    
    def _apply_nonlinearity(self, vector: np.ndarray) -> np.ndarray:
        """Apply non-linear transformation."""
        # Use tanh for bounded output
        return np.tanh(vector)
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize hypervector."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    @numba.jit(nopython=True)
    def _fast_cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Fast cosine similarity computation."""
        dot_product = np.dot(v1, v2)
        norm1 = np.sqrt(np.dot(v1, v1))
        norm2 = np.sqrt(np.dot(v2, v2))
        
        if norm1 * norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def compute_similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Compute similarity between two hypervectors."""
        return self._fast_cosine_similarity(hv1, hv2)
    
    def create_hierarchical_encoding(self, base_vector: np.ndarray,
                                   level: str = 'mid') -> np.ndarray:
        """
        Create hierarchical encoding from base vector.
        
        Args:
            base_vector: Base-level hypervector
            level: Target level ('mid' or 'high')
            
        Returns:
            Higher-level encoding
        """
        if level == 'mid':
            projection = self.hierarchy_projections['base_to_mid']
            if self.use_sparse:
                result = projection.dot(base_vector)
            else:
                result = np.dot(projection, base_vector)
        elif level == 'high':
            # First to mid, then to high
            mid_vector = self.create_hierarchical_encoding(base_vector, 'mid')
            projection = self.hierarchy_projections['mid_to_high']
            if self.use_sparse:
                result = projection.dot(mid_vector)
            else:
                result = np.dot(projection, mid_vector)
        else:
            raise ValueError(f"Unknown level: {level}")
        
        return self._normalize(self._apply_nonlinearity(result))
    
    def compress(self, hypervector: np.ndarray) -> bytes:
        """
        Compress hypervector for storage.
        
        Args:
            hypervector: Hypervector to compress
            
        Returns:
            Compressed binary representation
        """
        # Quantize to int8 for compression
        quantized = (hypervector * 127).astype(np.int8)
        
        # Apply sparsification if beneficial
        sparsity = np.sum(np.abs(quantized) < 10) / len(quantized)
        
        if sparsity > config.hypervector.sparsity_threshold:
            # Store as sparse
            indices = np.where(np.abs(quantized) >= 10)[0]
            values = quantized[indices]
            
            # Pack into bytes
            data = {
                'sparse': True,
                'dim': len(hypervector),
                'indices': indices.tolist(),
                'values': values.tolist(),
                'sparsity': sparsity
            }
        else:
            # Store as dense
            data = {
                'sparse': False,
                'dim': len(hypervector),
                'values': quantized.tobytes(),
                'sparsity': sparsity
            }
        
        return pickle.dumps(data, protocol=5)
    
    def decompress(self, compressed: bytes) -> np.ndarray:
        """
        Decompress hypervector from storage.
        
        Args:
            compressed: Compressed binary representation
            
        Returns:
            Decompressed hypervector
        """
        data = pickle.loads(compressed)
        
        if data['sparse']:
            # Reconstruct from sparse
            vector = np.zeros(data['dim'], dtype=np.float32)
            indices = np.array(data['indices'])
            values = np.array(data['values'], dtype=np.float32)
            vector[indices] = values / 127.0
        else:
            # Reconstruct from dense
            vector = np.frombuffer(data['values'], dtype=np.int8).astype(np.float32)
            vector = vector / 127.0
        
        return vector
    
    def get_metadata(self, hypervector: np.ndarray, domain: str) -> HypervectorMetadata:
        """Get metadata for hypervector."""
        sparsity = np.sum(np.abs(hypervector) < 0.1) / len(hypervector)
        
        return HypervectorMetadata(
            dimensions=len(hypervector),
            sparsity=sparsity,
            projection_version=config.hypervector.projection_version,
            domain_type=domain,
            compression_tier=self.compression_tier,
            feature_count=np.sum(np.abs(hypervector) > 0.1)
        )
    
    def save_projection(self, domain: str, path: Path):
        """Save projection matrix for reproducibility."""
        projection = self.projections.get(domain)
        if projection is None:
            raise ValueError(f"No projection for domain: {domain}")
        
        if self.use_sparse:
            # Save sparse matrix
            from scipy.sparse import save_npz
            save_npz(path, projection)
        else:
            # Save dense matrix
            np.save(path, projection)
    
    def load_projection(self, domain: str, path: Path):
        """Load projection matrix."""
        if self.use_sparse:
            from scipy.sparse import load_npz
            self.projections[domain] = load_npz(path)
        else:
            self.projections[domain] = np.load(path)


# Binding operations for cross-modal integration
class HypervectorBinder:
    """Implements binding operations for hypervector composition."""
    
    @staticmethod
    @numba.jit(nopython=True)
    def circular_convolution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Circular convolution for position-aware binding.
        
        Args:
            x, y: Input hypervectors
            
        Returns:
            Bound hypervector
        """
        n = len(x)
        result = np.zeros(n)
        
        for i in range(n):
            for j in range(n):
                result[i] += x[j] * y[(i - j) % n]
        
        return result / np.sqrt(n)
    
    @staticmethod
    def element_wise_multiply(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Element-wise multiplication for feature binding."""
        return x * y
    
    @staticmethod
    def permute(x: np.ndarray, shift: int = 1) -> np.ndarray:
        """Permute hypervector for sequence encoding."""
        return np.roll(x, shift)
    
    @staticmethod
    def bundle(vectors: List[np.ndarray]) -> np.ndarray:
        """Bundle multiple hypervectors by averaging."""
        bundled = np.mean(vectors, axis=0)
        # Normalize
        norm = np.linalg.norm(bundled)
        if norm > 0:
            bundled = bundled / norm
        return bundled


# Example usage
if __name__ == "__main__":
    # Initialize encoder
    encoder = HypervectorEncoder(compression_tier=CompressionTier.CLINICAL)
    
    # Example genomic features
    genomic_features = {
        'variants': [
            {'genotype': '0/1', 'position': 100},
            {'genotype': '1/1', 'position': 200},
            {'genotype': '0/1', 'position': 300}
        ]
    }
    
    # Encode
    genomic_hv = encoder.encode_features(genomic_features, domain='genomic')
    print(f"Genomic hypervector shape: {genomic_hv.shape}")
    
    # Example transcriptomic features
    transcriptomic_features = {
        'expression': np.random.lognormal(0, 1, 1000)
    }
    
    # Encode
    trans_hv = encoder.encode_features(transcriptomic_features, domain='transcriptomic')
    print(f"Transcriptomic hypervector shape: {trans_hv.shape}")
    
    # Compute similarity
    similarity = encoder.compute_similarity(genomic_hv, trans_hv)
    print(f"Cross-modal similarity: {similarity:.4f}")
    
    # Create hierarchical encoding
    mid_level = encoder.create_hierarchical_encoding(genomic_hv, 'mid')
    print(f"Mid-level encoding shape: {mid_level.shape}")
    
    # Compress and decompress
    compressed = encoder.compress(genomic_hv)
    decompressed = encoder.decompress(compressed)
    
    # Check reconstruction error
    reconstruction_error = np.mean(np.abs(genomic_hv - decompressed))
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    
    # Cross-modal binding
    binder = HypervectorBinder()
    bound = binder.circular_convolution(genomic_hv, trans_hv)
    print(f"Bound hypervector shape: {bound.shape}")
    
    # Calculate storage sizes
    print(f"\nStorage requirements:")
    print(f"Mini tier: {config.get_compression_size(['genomics'])} KB")
    print(f"Clinical tier: {config.get_compression_size(['genomics', 'transcriptomics'])} KB")
