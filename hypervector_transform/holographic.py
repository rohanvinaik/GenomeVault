"""
Holographic representation for distributed information encoding

This module implements holographic reduced representations (HRR) for
encoding complex structured information in hypervectors.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from core.exceptions import EncodingError
from utils.logging import get_logger

from .binding import BindingType, HypervectorBinder
from .encoding import HypervectorEncoder

logger = get_logger(__name__)


@dataclass
class HolographicStructure:
    """Represents a holographic data structure"""
    root: torch.Tensor
    components: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]


class HolographicEncoder:
    """
    Implements Holographic Reduced Representations (HRR)
    
    HRR allows encoding of complex compositional structures in fixed-size
    vectors while maintaining the ability to query for components.
    """
    
    def __init__(self, dimension: int = 10000):
        """
        Initialize holographic encoder
        
        Args:
            dimension: Dimension of hypervectors
        """
        self.dimension = dimension
        self.encoder = HypervectorEncoder()
        self.binder = HypervectorBinder(dimension)
        self._role_vectors = {}
        
        logger.info(f"Initialized HolographicEncoder with {dimension}D vectors")
    
    def encode_structure(self, structure: Dict[str, Any],
                        recursive: bool = True) -> HolographicStructure:
        """
        Encode a hierarchical structure holographically
        
        Args:
            structure: Nested dictionary structure to encode
            recursive: Whether to recursively encode nested structures
            
        Returns:
            Holographic representation
        """
        components = {}
        bound_pairs = []
        
        for role, filler in structure.items():
            # Get or create role vector
            role_vector = self._get_role_vector(role)
            
            # Encode filler
            if isinstance(filler, dict) and recursive:
                # Recursively encode nested structure
                nested = self.encode_structure(filler, recursive=True)
                filler_vector = nested.root
                components[f"{role}_nested"] = nested
            elif isinstance(filler, (list, tuple)):
                # Encode sequence
                filler_vector = self._encode_sequence(filler)
            elif isinstance(filler, torch.Tensor):
                # Already a hypervector
                filler_vector = filler
            else:
                # Encode atomic value
                filler_vector = self._encode_atomic(filler)
            
            # Store component
            components[role] = filler_vector
            
            # Bind role with filler
            bound = self.binder.bind(
                [role_vector, filler_vector],
                BindingType.CIRCULAR
            )
            bound_pairs.append(bound)
        
        # Bundle all role-filler pairs
        root = self.binder.bundle(bound_pairs, normalize=True)
        
        return HolographicStructure(
            root=root,
            components=components,
            metadata={"roles": list(structure.keys())}
        )
    
    def query(self, hologram: torch.Tensor, role: str,
             cleanup: bool = True) -> torch.Tensor:
        """
        Query a holographic representation for a specific role
        
        Args:
            hologram: Holographic vector to query
            role: Role to query for
            cleanup: Whether to clean up the result
            
        Returns:
            Approximate filler vector for the role
        """
        # Get role vector
        role_vector = self._get_role_vector(role)
        
        # Unbind to get approximate filler
        filler_approx = self.binder.unbind(
            hologram,
            [role_vector],
            BindingType.CIRCULAR
        )
        
        # Clean up if requested
        if cleanup:
            filler_approx = self._cleanup_vector(filler_approx)
        
        return filler_approx
    
    def encode_genomic_variant(self, chromosome: str, position: int,
                              ref: str, alt: str,
                              annotations: Optional[Dict] = None) -> torch.Tensor:
        """
        Encode a genomic variant holographically
        
        Args:
            chromosome: Chromosome name
            position: Genomic position
            ref: Reference allele
            alt: Alternative allele
            annotations: Optional annotations
            
        Returns:
            Holographic representation of variant
        """
        structure = {
            "chr": chromosome,
            "pos": position,
            "ref": ref,
            "alt": alt
        }
        
        if annotations:
            structure["annotations"] = annotations
        
        hologram = self.encode_structure(structure)
        return hologram.root
    
    def encode_gene_expression(self, gene_id: str, expression: float,
                             conditions: Dict[str, Any]) -> torch.Tensor:
        """
        Encode gene expression data holographically
        
        Args:
            gene_id: Gene identifier
            expression: Expression level
            conditions: Experimental conditions
            
        Returns:
            Holographic representation
        """
        structure = {
            "gene": gene_id,
            "expression": expression,
            "conditions": conditions
        }
        
        hologram = self.encode_structure(structure)
        return hologram.root
    
    def encode_protein_interaction(self, protein1: str, protein2: str,
                                  interaction_type: str,
                                  confidence: float) -> torch.Tensor:
        """
        Encode protein-protein interaction
        
        Args:
            protein1: First protein ID
            protein2: Second protein ID
            interaction_type: Type of interaction
            confidence: Confidence score
            
        Returns:
            Holographic representation
        """
        structure = {
            "protein1": protein1,
            "protein2": protein2,
            "type": interaction_type,
            "confidence": confidence
        }
        
        hologram = self.encode_structure(structure)
        return hologram.root
    
    def create_memory_trace(self, items: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Create a memory trace by superimposing multiple items
        
        Args:
            items: List of items to store in memory
            
        Returns:
            Memory trace vector
        """
        traces = []
        
        for i, item in enumerate(items):
            # Add position information
            item_with_pos = {
                "position": i,
                "content": item
            }
            
            # Encode item
            hologram = self.encode_structure(item_with_pos)
            traces.append(hologram.root)
        
        # Superimpose all traces
        memory = self.binder.bundle(traces, normalize=True)
        
        return memory
    
    def _get_role_vector(self, role: str) -> torch.Tensor:
        """Get or create a role vector"""
        if role not in self._role_vectors:
            # Create deterministic role vector based on role name
            seed = int(hashlib.md5(role.encode()).hexdigest()[:8], 16)
            torch.manual_seed(seed)
            
            # Create orthogonal role vector
            role_vector = torch.randn(self.dimension)
            role_vector = role_vector / torch.norm(role_vector)
            
            self._role_vectors[role] = role_vector
        
        return self._role_vectors[role]
    
    def _encode_atomic(self, value: Any) -> torch.Tensor:
        """Encode an atomic value"""
        if isinstance(value, (int, float)):
            # Encode numeric value
            return self._encode_numeric(value)
        elif isinstance(value, str):
            # Encode string
            return self._encode_string(value)
        elif isinstance(value, bool):
            # Encode boolean
            return self._encode_boolean(value)
        else:
            # Default: hash and encode
            return self._encode_hash(value)
    
    def _encode_numeric(self, value: float) -> torch.Tensor:
        """Encode a numeric value"""
        # Use thermometer encoding
        vector = torch.zeros(self.dimension)
        
        # Normalize to [0, 1] assuming reasonable range
        normalized = (value + 1000) / 2000  # Assumes [-1000, 1000] range
        normalized = np.clip(normalized, 0, 1)
        
        # Fill proportional to value
        fill_count = int(normalized * self.dimension)
        vector[:fill_count] = 1.0
        
        # Add some randomness
        noise = torch.randn(self.dimension) * 0.1
        vector = vector + noise
        
        return vector / torch.norm(vector)
    
    def _encode_string(self, value: str) -> torch.Tensor:
        """Encode a string value"""
        # Use character n-grams
        ngrams = []
        n = 3  # Trigrams
        
        for i in range(len(value) - n + 1):
            ngrams.append(value[i:i+n])
        
        # Encode each n-gram and bundle
        ngram_vectors = []
        for ngram in ngrams:
            seed = int(hashlib.md5(ngram.encode()).hexdigest()[:8], 16)
            torch.manual_seed(seed)
            ngram_vector = torch.randn(self.dimension)
            ngram_vectors.append(ngram_vector)
        
        if ngram_vectors:
            return self.binder.bundle(ngram_vectors, normalize=True)
        else:
            return self._encode_hash(value)
    
    def _encode_boolean(self, value: bool) -> torch.Tensor:
        """Encode a boolean value"""
        # Use fixed vectors for true/false
        torch.manual_seed(1 if value else 0)
        vector = torch.randn(self.dimension)
        return vector / torch.norm(vector)
    
    def _encode_hash(self, value: Any) -> torch.Tensor:
        """Encode using hash of value"""
        # Hash the string representation
        hash_val = hashlib.sha256(str(value).encode()).hexdigest()
        seed = int(hash_val[:8], 16)
        
        torch.manual_seed(seed)
        vector = torch.randn(self.dimension)
        return vector / torch.norm(vector)
    
    def _encode_sequence(self, sequence: List[Any]) -> torch.Tensor:
        """Encode a sequence of values"""
        if not sequence:
            return torch.zeros(self.dimension)
        
        # Encode each element with position
        encoded_elements = []
        
        for i, element in enumerate(sequence):
            # Encode element
            if isinstance(element, torch.Tensor):
                elem_vector = element
            else:
                elem_vector = self._encode_atomic(element)
            
            # Apply position-specific permutation
            perm = self._get_permutation(i)
            positioned = elem_vector[perm]
            
            encoded_elements.append(positioned)
        
        # Bundle all elements
        return self.binder.bundle(encoded_elements, normalize=True)
    
    def _get_permutation(self, position: int) -> torch.Tensor:
        """Get position-specific permutation"""
        torch.manual_seed(12345 + position)
        return torch.randperm(self.dimension)
    
    def _cleanup_vector(self, vector: torch.Tensor,
                       threshold: float = 0.1) -> torch.Tensor:
        """Clean up a noisy vector"""
        # Simple thresholding
        cleaned = vector.clone()
        cleaned[torch.abs(cleaned) < threshold] = 0
        
        # Renormalize
        norm = torch.norm(cleaned)
        if norm > 0:
            cleaned = cleaned / norm
        
        return cleaned
    
    def similarity_preserving_hash(self, vector: torch.Tensor,
                                  num_bits: int = 64) -> str:
        """
        Create a similarity-preserving hash of a hypervector
        
        Args:
            vector: Hypervector to hash
            num_bits: Number of bits in hash
            
        Returns:
            Hexadecimal hash string
        """
        # Use random hyperplanes for LSH
        torch.manual_seed(42)  # Fixed seed for consistency
        hyperplanes = torch.randn(num_bits, self.dimension)
        
        # Project and threshold
        projections = torch.matmul(hyperplanes, vector)
        bits = (projections > 0).int()
        
        # Convert to hex string
        hash_int = 0
        for i, bit in enumerate(bits):
            hash_int |= (int(bit) << i)
        
        return hex(hash_int)[2:].zfill(num_bits // 4)


# Convenience functions
def encode_variant(chrom: str, pos: int, ref: str, alt: str) -> torch.Tensor:
    """Convenience function to encode a genomic variant"""
    encoder = HolographicEncoder()
    return encoder.encode_genomic_variant(chrom, pos, ref, alt)


def query_hologram(hologram: torch.Tensor, role: str) -> torch.Tensor:
    """Convenience function to query a holographic representation"""
    encoder = HolographicEncoder(hologram.shape[-1])
    return encoder.query(hologram, role)


# Import hashlib
import hashlib
