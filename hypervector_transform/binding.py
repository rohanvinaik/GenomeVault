"""
GenomeVault Hypervector Binding Operations

Implements binding, bundling, and permutation operations for hyperdimensional computing
to create composite representations across modalities.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from scipy.fft import fft, ifft
from functools import reduce

from ..utils import get_logger
from .encoding import Hypervector, VectorType

logger = get_logger(__name__)


class HypervectorOperations:
    """Core operations for hyperdimensional computing"""
    
    @staticmethod
    def circular_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Circular convolution for position-aware binding
        
        Uses FFT for efficient computation:
        a ⊛ b = IFFT(FFT(a) ⊙ FFT(b))
        
        Args:
            a: First hypervector
            b: Second hypervector
            
        Returns:
            Bound hypervector
        """
        if len(a) != len(b):
            raise ValueError(f"Vectors must have same dimension: {len(a)} vs {len(b)}")
        
        # Use FFT for efficient circular convolution
        fft_a = fft(a)
        fft_b = fft(b)
        result = ifft(fft_a * fft_b).real
        
        return result
    
    @staticmethod
    def element_wise_multiplication(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Element-wise multiplication for feature-type binding
        
        Args:
            a: First hypervector
            b: Second hypervector
            
        Returns:
            Bound hypervector
        """
        if len(a) != len(b):
            raise ValueError(f"Vectors must have same dimension: {len(a)} vs {len(b)}")
        
        return a * b
    
    @staticmethod
    def bundling(vectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Bundle multiple hypervectors through weighted addition
        
        Args:
            vectors: List of hypervectors to bundle
            weights: Optional weights for each vector
            
        Returns:
            Bundled hypervector
        """
        if not vectors:
            raise ValueError("Cannot bundle empty list of vectors")
        
        if weights is None:
            # Equal weighting
            result = np.mean(vectors, axis=0)
        else:
            if len(weights) != len(vectors):
                raise ValueError(f"Number of weights ({len(weights)}) must match vectors ({len(vectors)})")
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Weighted sum
            result = np.zeros_like(vectors[0])
            for vec, weight in zip(vectors, weights):
                result += weight * vec
        
        return result
    
    @staticmethod
    def permutation(vector: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        Cyclic permutation for sequence representation
        
        Args:
            vector: Input hypervector
            shift: Number of positions to shift (positive = right shift)
            
        Returns:
            Permuted hypervector
        """
        return np.roll(vector, shift)
    
    @staticmethod
    def inverse_permutation(vector: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        Inverse cyclic permutation
        
        Args:
            vector: Input hypervector
            shift: Number of positions to shift
            
        Returns:
            Inverse permuted hypervector
        """
        return np.roll(vector, -shift)
    
    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray, metric: str = 'cosine') -> float:
        """
        Calculate similarity between hypervectors
        
        Args:
            a: First hypervector
            b: Second hypervector
            metric: Similarity metric ('cosine', 'hamming', 'euclidean')
            
        Returns:
            Similarity score
        """
        if len(a) != len(b):
            raise ValueError(f"Vectors must have same dimension: {len(a)} vs {len(b)}")
        
        if metric == 'cosine':
            # Cosine similarity
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return np.dot(a, b) / (norm_a * norm_b)
            
        elif metric == 'hamming':
            # Hamming distance (for binary vectors)
            # Convert to binary first
            binary_a = (a > 0).astype(int)
            binary_b = (b > 0).astype(int)
            hamming_dist = np.sum(binary_a != binary_b)
            # Convert to similarity (0 to 1)
            return 1.0 - (hamming_dist / len(a))
            
        elif metric == 'euclidean':
            # Euclidean distance converted to similarity
            dist = np.linalg.norm(a - b)
            # Convert to similarity using exponential decay
            return np.exp(-dist / len(a))
            
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    @staticmethod
    def normalize(vector: np.ndarray, method: str = 'l2') -> np.ndarray:
        """
        Normalize hypervector
        
        Args:
            vector: Input hypervector
            method: Normalization method ('l2', 'l1', 'max')
            
        Returns:
            Normalized hypervector
        """
        if method == 'l2':
            norm = np.linalg.norm(vector)
            if norm > 0:
                return vector / norm
            return vector
            
        elif method == 'l1':
            norm = np.sum(np.abs(vector))
            if norm > 0:
                return vector / norm
            return vector
            
        elif method == 'max':
            max_val = np.max(np.abs(vector))
            if max_val > 0:
                return vector / max_val
            return vector
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")


class BindingOperations:
    """High-level binding operations for multi-modal data"""
    
    def __init__(self):
        """Initialize binding operations"""
        self.ops = HypervectorOperations()
        logger.info("Initialized BindingOperations")
    
    def bind_genomic_variant(self, 
                           variant_vector: Hypervector,
                           position_vector: Hypervector) -> Hypervector:
        """
        Bind variant with positional information
        
        Args:
            variant_vector: Variant features hypervector
            position_vector: Positional encoding hypervector
            
        Returns:
            Position-aware variant hypervector
        """
        # Use circular convolution for position binding
        bound_data = self.ops.circular_convolution(
            variant_vector.data, 
            position_vector.data
        )
        
        return Hypervector(
            vector_id=f"{variant_vector.vector_id}_pos_bound",
            vector_type=VectorType.COMPOSITE,
            dimensions=variant_vector.dimensions,
            data=bound_data,
            sparsity=0.0,
            metadata={
                'binding_type': 'position',
                'source_vectors': [variant_vector.vector_id, position_vector.vector_id]
            }
        )
    
    def bind_expression_context(self,
                              expression_vector: Hypervector,
                              tissue_vector: Hypervector,
                              condition_vector: Optional[Hypervector] = None) -> Hypervector:
        """
        Bind expression with tissue and condition context
        
        Args:
            expression_vector: Gene expression hypervector
            tissue_vector: Tissue type encoding
            condition_vector: Optional condition/disease context
            
        Returns:
            Context-aware expression hypervector
        """
        # First bind expression with tissue
        tissue_bound = self.ops.element_wise_multiplication(
            expression_vector.data,
            tissue_vector.data
        )
        
        if condition_vector is not None:
            # Further bind with condition
            final_data = self.ops.element_wise_multiplication(
                tissue_bound,
                condition_vector.data
            )
            source_vectors = [expression_vector.vector_id, tissue_vector.vector_id, condition_vector.vector_id]
        else:
            final_data = tissue_bound
            source_vectors = [expression_vector.vector_id, tissue_vector.vector_id]
        
        return Hypervector(
            vector_id=f"{expression_vector.vector_id}_context_bound",
            vector_type=VectorType.COMPOSITE,
            dimensions=expression_vector.dimensions,
            data=final_data,
            sparsity=0.0,
            metadata={
                'binding_type': 'context',
                'source_vectors': source_vectors
            }
        )
    
    def create_temporal_sequence(self,
                               vectors: List[Hypervector],
                               time_deltas: Optional[List[float]] = None) -> Hypervector:
        """
        Create temporal sequence representation
        
        Args:
            vectors: List of hypervectors in temporal order
            time_deltas: Optional time differences between measurements
            
        Returns:
            Temporal sequence hypervector
        """
        if not vectors:
            raise ValueError("Cannot create sequence from empty list")
        
        # Apply permutation based on position
        permuted_vectors = []
        for i, vec in enumerate(vectors):
            # Shift increases with position in sequence
            shift = i * (vec.dimensions // len(vectors))
            permuted = self.ops.permutation(vec.data, shift)
            
            # Weight by time delta if provided
            if time_deltas and i < len(time_deltas):
                # Exponential decay based on time
                weight = np.exp(-time_deltas[i] / 365.0)  # Decay over years
                permuted = permuted * weight
            
            permuted_vectors.append(permuted)
        
        # Bundle all permuted vectors
        sequence_data = self.ops.bundling(permuted_vectors)
        
        return Hypervector(
            vector_id=f"temporal_sequence_{len(vectors)}",
            vector_type=VectorType.COMPOSITE,
            dimensions=vectors[0].dimensions,
            data=sequence_data,
            sparsity=0.0,
            metadata={
                'binding_type': 'temporal',
                'sequence_length': len(vectors),
                'source_vectors': [v.vector_id for v in vectors]
            }
        )
    
    def bind_multi_omics(self,
                        genomic: Optional[Hypervector] = None,
                        transcriptomic: Optional[Hypervector] = None,
                        epigenetic: Optional[Hypervector] = None,
                        proteomic: Optional[Hypervector] = None,
                        phenotypic: Optional[Hypervector] = None,
                        weights: Optional[Dict[str, float]] = None) -> Hypervector:
        """
        Create integrated multi-omics representation
        
        Args:
            genomic: Genomic hypervector
            transcriptomic: Transcriptomic hypervector
            epigenetic: Epigenetic hypervector
            proteomic: Proteomic hypervector
            phenotypic: Phenotypic hypervector
            weights: Optional weights for each modality
            
        Returns:
            Integrated multi-omics hypervector
        """
        # Collect available modalities
        modalities = {}
        if genomic is not None:
            modalities['genomic'] = genomic
        if transcriptomic is not None:
            modalities['transcriptomic'] = transcriptomic
        if epigenetic is not None:
            modalities['epigenetic'] = epigenetic
        if proteomic is not None:
            modalities['proteomic'] = proteomic
        if phenotypic is not None:
            modalities['phenotypic'] = phenotypic
        
        if not modalities:
            raise ValueError("At least one modality must be provided")
        
        # Default weights if not provided
        if weights is None:
            weights = {name: 1.0 for name in modalities.keys()}
        
        # Normalize weights
        total_weight = sum(weights.get(name, 1.0) for name in modalities.keys())
        normalized_weights = {name: weights.get(name, 1.0) / total_weight 
                            for name in modalities.keys()}
        
        # Create bound representation
        if len(modalities) == 1:
            # Single modality
            name, vector = list(modalities.items())[0]
            return vector
        
        # Multi-modal binding
        bound_vectors = []
        modality_weights = []
        
        for name, vector in modalities.items():
            bound_vectors.append(vector.data)
            modality_weights.append(normalized_weights[name])
        
        # Bundle with weights
        integrated_data = self.ops.bundling(bound_vectors, modality_weights)
        
        return Hypervector(
            vector_id=f"multiomics_{len(modalities)}",
            vector_type=VectorType.COMPOSITE,
            dimensions=list(modalities.values())[0].dimensions,
            data=integrated_data,
            sparsity=0.0,
            metadata={
                'binding_type': 'multi_omics',
                'modalities': list(modalities.keys()),
                'weights': normalized_weights,
                'source_vectors': [v.vector_id for v in modalities.values()]
            }
        )
    
    def unbind(self, 
               bound_vector: Hypervector, 
               binding_vector: Hypervector,
               binding_type: str = 'circular') -> Hypervector:
        """
        Unbind a hypervector (approximate inverse operation)
        
        Args:
            bound_vector: The bound hypervector
            binding_vector: The vector that was used for binding
            binding_type: Type of binding used ('circular' or 'element_wise')
            
        Returns:
            Approximately unbound hypervector
        """
        if binding_type == 'circular':
            # For circular convolution, unbind using correlation
            # This is approximate and may have noise
            unbound_data = self.ops.circular_convolution(
                bound_vector.data,
                binding_vector.data[::-1]  # Reverse for correlation
            )
        elif binding_type == 'element_wise':
            # For element-wise, divide (with protection against division by zero)
            safe_binding = binding_vector.data.copy()
            safe_binding[safe_binding == 0] = 1e-10
            unbound_data = bound_vector.data / safe_binding
        else:
            raise ValueError(f"Unknown binding type: {binding_type}")
        
        return Hypervector(
            vector_id=f"{bound_vector.vector_id}_unbound",
            vector_type=bound_vector.vector_type,
            dimensions=bound_vector.dimensions,
            data=unbound_data,
            sparsity=0.0,
            metadata={
                'operation': 'unbind',
                'binding_type': binding_type,
                'source_vector': bound_vector.vector_id
            }
        )
    
    def create_role_filler(self,
                          role_vector: Hypervector,
                          filler_vector: Hypervector) -> Hypervector:
        """
        Create role-filler binding for structured representations
        
        Args:
            role_vector: Role/slot hypervector
            filler_vector: Filler/value hypervector
            
        Returns:
            Role-filler bound hypervector
        """
        # Use circular convolution for role-filler binding
        bound_data = self.ops.circular_convolution(
            role_vector.data,
            filler_vector.data
        )
        
        return Hypervector(
            vector_id=f"{role_vector.vector_id}_filled_{filler_vector.vector_id}",
            vector_type=VectorType.COMPOSITE,
            dimensions=role_vector.dimensions,
            data=bound_data,
            sparsity=0.0,
            metadata={
                'binding_type': 'role_filler',
                'role': role_vector.vector_id,
                'filler': filler_vector.vector_id
            }
        )
    
    def create_graph_binding(self,
                           node_vectors: Dict[str, Hypervector],
                           edges: List[Tuple[str, str, float]]) -> Hypervector:
        """
        Create graph-based binding for network structures
        
        Args:
            node_vectors: Dictionary of node_id -> hypervector
            edges: List of (source_id, target_id, weight) tuples
            
        Returns:
            Graph-bound hypervector
        """
        if not node_vectors:
            raise ValueError("Cannot create graph from empty nodes")
        
        # Initialize with first node
        graph_vector = np.zeros_like(list(node_vectors.values())[0].data)
        
        # Add weighted edges
        for source_id, target_id, weight in edges:
            if source_id not in node_vectors or target_id not in node_vectors:
                logger.warning(f"Edge {source_id}->{target_id} references unknown node")
                continue
            
            # Bind source and target
            edge_binding = self.ops.circular_convolution(
                node_vectors[source_id].data,
                node_vectors[target_id].data
            )
            
            # Add weighted edge to graph
            graph_vector += weight * edge_binding
        
        # Add node information
        node_bundle = self.ops.bundling([v.data for v in node_vectors.values()])
        graph_vector = self.ops.bundling([graph_vector, node_bundle], [0.7, 0.3])
        
        return Hypervector(
            vector_id=f"graph_{len(node_vectors)}nodes_{len(edges)}edges",
            vector_type=VectorType.COMPOSITE,
            dimensions=list(node_vectors.values())[0].dimensions,
            data=graph_vector,
            sparsity=0.0,
            metadata={
                'binding_type': 'graph',
                'node_count': len(node_vectors),
                'edge_count': len(edges),
                'node_ids': list(node_vectors.keys())
            }
        )


class SimilaritySearch:
    """Efficient similarity search in hypervector space"""
    
    def __init__(self):
        """Initialize similarity search"""
        self.ops = HypervectorOperations()
        self.index = {}  # Simple dictionary index for now
        logger.info("Initialized SimilaritySearch")
    
    def add_vector(self, vector: Hypervector):
        """Add vector to search index"""
        self.index[vector.vector_id] = vector
    
    def find_similar(self, 
                    query_vector: Hypervector,
                    k: int = 10,
                    metric: str = 'cosine',
                    min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        """
        Find k most similar vectors to query
        
        Args:
            query_vector: Query hypervector
            k: Number of results to return
            metric: Similarity metric
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (vector_id, similarity) tuples
        """
        similarities = []
        
        for vec_id, vec in self.index.items():
            if vec_id == query_vector.vector_id:
                continue
            
            sim = self.ops.similarity(query_vector.data, vec.data, metric)
            if sim >= min_similarity:
                similarities.append((vec_id, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def find_similar_batch(self,
                          query_vectors: List[Hypervector],
                          k: int = 10,
                          metric: str = 'cosine') -> Dict[str, List[Tuple[str, float]]]:
        """
        Find similar vectors for multiple queries
        
        Args:
            query_vectors: List of query hypervectors
            k: Number of results per query
            metric: Similarity metric
            
        Returns:
            Dictionary mapping query vector_id to similar vectors
        """
        results = {}
        
        for query in query_vectors:
            results[query.vector_id] = self.find_similar(query, k, metric)
        
        return results
