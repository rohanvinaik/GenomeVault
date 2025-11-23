"""Core module for Hyperdimensional Computing."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Default dimension
D = 10_000


@dataclass
class HDCConfig:
    """Configuration for HDC encoding"""
    dimension: int = 10000
    seed: Optional[int] = None
    sparsity: float = 0.1  # Sparsity of projection matrix
    normalize: bool = True
    similarity_threshold: float = 0.85  # For similarity comparisons
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.dimension < 100:
            raise ValueError(f"Dimension must be at least 100, got {self.dimension}")
        if not 0 <= self.sparsity <= 1:
            raise ValueError(f"Sparsity must be between 0 and 1, got {self.sparsity}")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError(f"Similarity threshold must be between 0 and 1, got {self.similarity_threshold}")


class HDCEncoder:
    """Hyperdimensional Computing encoder with configurable parameters"""
    
    def __init__(self, config: Optional[Union[HDCConfig, Dict[str, Any]]] = None):
        """
        Initialize HDC encoder with configuration
        
        Args:
            config: HDCConfig object or dict with configuration parameters
        """
        if config is None:
            self.config = HDCConfig()
        elif isinstance(config, dict):
            self.config = HDCConfig(**config)
        elif isinstance(config, HDCConfig):
            self.config = config
        else:
            raise TypeError(f"Config must be HDCConfig or dict, got {type(config)}")
        
        # Initialize random state
        self.rng = np.random.RandomState(self.config.seed)
        
        logger.debug(f"HDC encoder initialized with dimension={self.config.dimension}, "
                    f"sparsity={self.config.sparsity:.2f}")
    
    def encode(self, X: np.ndarray, omics_type: Optional[str] = None) -> np.ndarray:
        """
        Encode feature matrix into hypervectors.
        
        Args:
            X: Input feature matrix (n_samples x n_features) or 1D array
            omics_type: Optional type of omics data (for specialized encoding)
        
        Returns:
            Hypervector matrix (n_samples x dimension) or single hypervector
        """
        # Handle 1D input
        if X.ndim == 1:
            X = X.reshape(1, -1)
            return_1d = True
        else:
            return_1d = False
            
        if X.ndim != 2 or X.size == 0:
            raise ValueError("X must be a non-empty array")
        
        n_samples, n_features = X.shape
        
        # Create random projection matrix (sparse for efficiency)
        projection = self.rng.randn(n_features, self.config.dimension) * \
                    np.sqrt(1 / (n_features * self.config.sparsity))
        mask = self.rng.random((n_features, self.config.dimension)) < self.config.sparsity
        projection *= mask
        
        # Project and binarize
        V = X @ projection
        result = np.sign(V + 1e-10)  # Avoid zero values
        
        # Return 1D if input was 1D
        if return_1d:
            result = result[0]
        
        return result.astype(np.float32)
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between hypervectors.
        
        Args:
            a: First hypervector
            b: Second hypervector
        
        Returns:
            Similarity score in [0, 1]
        """
        if a.shape != b.shape:
            raise ValueError("Hypervectors must have the same shape")
        
        # Cosine similarity for binary vectors
        dot = np.dot(a.flatten(), b.flatten())
        norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
        sim = dot / norm
        return (sim + 1) / 2  # Map from [-1, 1] to [0, 1]
    
    def bundle(self, vectors: np.ndarray) -> np.ndarray:
        """
        Bundle multiple hypervectors into one.
        
        Args:
            vectors: Matrix of hypervectors to bundle
        
        Returns:
            Bundled hypervector
        """
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        
        bundled = np.sum(vectors, axis=0)
        if self.config.normalize:
            bundled = np.sign(bundled + 1e-10)
        return bundled


def encode(X: np.ndarray, *, 
          seed: Optional[int] = None,
          config: Optional[Union[HDCConfig, Dict[str, Any]]] = None,
          **kwargs) -> np.ndarray:
    """Encode feature matrix into hypervectors.

    Args:
        X: Input feature matrix (n_samples x n_features)
        seed: Random seed for reproducibility
        config: HDCConfig or dict with configuration
        **kwargs: Additional keyword arguments for HDCConfig

    Returns:
        Hypervector matrix (n_samples x D)
    """
    # Build config from parameters
    if config is None:
        config_kwargs = {'seed': seed} if seed is not None else {}
        config_kwargs.update(kwargs)
        config = HDCConfig(**config_kwargs) if config_kwargs else HDCConfig()
    elif isinstance(config, dict):
        if seed is not None:
            config['seed'] = seed
        config.update(kwargs)
        config = HDCConfig(**config)
    elif seed is not None or kwargs:
        # Update existing config
        config_dict = {
            'dimension': config.dimension,
            'seed': seed if seed is not None else config.seed,
            'sparsity': config.sparsity,
            'normalize': config.normalize,
            'similarity_threshold': config.similarity_threshold,
        }
        config_dict.update(kwargs)
        config = HDCConfig(**config_dict)
    
    encoder = HDCEncoder(config)
    return encoder.encode(X)


def bundle(vectors: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Bundle multiple hypervectors into one.

    Args:
        vectors: Matrix of hypervectors to bundle
        normalize: Whether to normalize the output

    Returns:
        Bundled hypervector
    """
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")

    bundled = np.sum(vectors, axis=0)
    if normalize:
        bundled = np.sign(bundled + 1e-10)
    return bundled


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between hypervectors.

    Args:
        a: First hypervector
        b: Second hypervector

    Returns:
        Similarity score in [0, 1]
    """
    if a.shape != b.shape:
        raise ValueError("Hypervectors must have the same shape")

    # Cosine similarity for binary vectors
    dot = np.dot(a.flatten(), b.flatten())
    norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
    sim = dot / norm
    return (sim + 1) / 2  # Map from [-1, 1] to [0, 1]
