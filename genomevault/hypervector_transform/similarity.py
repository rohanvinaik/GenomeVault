"""
HDC Similarity Module for GenomeVault
Provides similarity metrics appropriate for hyperdimensional vectors
"""

import numpy as np
import torch
from typing import Union

TensorLike = Union[np.ndarray, torch.Tensor]


class HDCSimilarity:
    """Similarity metrics for hyperdimensional computing vectors"""
    
    @staticmethod
    def compute(v1: TensorLike, v2: TensorLike, 
                method: str = "weighted") -> float:
        """Compute similarity between two HDC vectors.
        
        Args:
            v1: First hypervector
            v2: Second hypervector
            method: Similarity method ("cosine", "hamming", "jaccard", "weighted")
            
        Returns:
            Similarity score in [0, 1]
        """
        # Convert to numpy if needed
        if isinstance(v1, torch.Tensor):
            v1 = v1.detach().cpu().numpy()
        if isinstance(v2, torch.Tensor):
            v2 = v2.detach().cpu().numpy()
        
        if method == "cosine":
            return HDCSimilarity._cosine_similarity(v1, v2)
        elif method == "hamming":
            return HDCSimilarity._hamming_similarity(v1, v2)
        elif method == "jaccard":
            return HDCSimilarity._jaccard_similarity(v1, v2)
        elif method == "weighted":
            return HDCSimilarity._weighted_similarity(v1, v2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Standard cosine similarity"""
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Map from [-1, 1] to [0, 1]
        cosine = dot / (norm1 * norm2)
        return (cosine + 1.0) / 2.0
    
    @staticmethod
    def _hamming_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Hamming similarity for binary/ternary vectors"""
        # Binarize
        v1_sign = np.sign(v1)
        v2_sign = np.sign(v2)
        
        # Count matches
        matches = np.sum(v1_sign == v2_sign)
        
        return matches / len(v1)
    
    @staticmethod
    def _jaccard_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Jaccard similarity on active components"""
        # Identify active (non-zero) components
        active1 = np.abs(v1) > 1e-10
        active2 = np.abs(v2) > 1e-10
        
        # Intersection and union
        intersection = np.sum(active1 & active2)
        union = np.sum(active1 | active2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    @staticmethod
    def _weighted_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Weighted combination optimized for sparse HDC vectors"""
        # Structural similarity (Jaccard on active components)
        active1 = np.abs(v1) > 1e-10
        active2 = np.abs(v2) > 1e-10
        
        intersection = np.sum(active1 & active2)
        union = np.sum(active1 | active2)
        
        if union == 0:
            return 0.0
        
        jaccard = intersection / union
        
        # Magnitude similarity (cosine on active components)
        active_both = active1 & active2
        
        if np.sum(active_both) > 0:
            v1_active = v1[active_both]
            v2_active = v2[active_both]
            
            dot = np.dot(v1_active, v2_active)
            norm1 = np.linalg.norm(v1_active)
            norm2 = np.linalg.norm(v2_active)
            
            if norm1 > 0 and norm2 > 0:
                cosine = dot / (norm1 * norm2)
                # Map to [0, 1]
                cosine = (cosine + 1.0) / 2.0
            else:
                cosine = 0.0
        else:
            cosine = 0.0
        
        # Weighted combination (tuned for HDC fingerprints)
        # More weight on magnitude for dense vectors
        # More weight on structure for sparse vectors
        sparsity = 1.0 - (intersection / len(v1))
        
        structure_weight = 0.3 + 0.4 * sparsity  # 0.3 to 0.7
        magnitude_weight = 1.0 - structure_weight
        
        similarity = structure_weight * jaccard + magnitude_weight * cosine
        
        return similarity


def compute_fingerprint_similarity(fp1: TensorLike, fp2: TensorLike) -> float:
    """Convenience function for fingerprint similarity.
    
    Args:
        fp1: First fingerprint (hypervector)
        fp2: Second fingerprint (hypervector)
        
    Returns:
        Similarity score in [0, 1]
    """
    return HDCSimilarity.compute(fp1, fp2, method="weighted")