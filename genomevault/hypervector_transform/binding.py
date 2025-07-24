"""
Binding operations for hyperdimensional computing

This module implements various binding operations that combine hypervectors
while preserving their mathematical properties and biological relationships.
"""

import hashlib
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from genomevault.core.exceptions import BindingError
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class BindingType(Enum):
    """Types of binding operations"""

    MULTIPLY = "multiply"  # Element-wise multiplication
    CIRCULAR = "circular"  # Circular convolution
    PERMUTATION = "permutation"  # Permutation-based binding
    XOR = "xor"  # XOR for binary vectors
    FOURIER = "fourier"  # Fourier-based binding


class HypervectorBinder:
    """
    Implements binding operations for hyperdimensional computing

    Binding operations combine multiple hypervectors into a single vector
    that represents their relationship while maintaining reversibility.
    """

    def __init__(self, dimension: int = 10000):
        """
        Initialize the binder

        Args:
            dimension: Expected dimension of hypervectors
        """
        self.dimension = dimension
        self._permutation_cache = {}

        logger.info("Initialized HypervectorBinder for {dimension}D vectors")

    def bind(
        self,
        vectors: List[torch.Tensor],
        binding_type: BindingType = BindingType.CIRCULAR,
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Bind multiple hypervectors together

        Args:
            vectors: List of hypervectors to bind
            binding_type: Type of binding operation
            weights: Optional weights for weighted binding

        Returns:
            Bound hypervector
        """
        if not vectors:
            raise BindingError("No vectors provided for binding")

        # Validate dimensions
        for i, v in enumerate(vectors):
            if v.shape[-1] != self.dimension:
                raise BindingError(
                    "Vector {i} has dimension {v.shape[-1]}, expected {self.dimension}"
                )

        # Apply weights if provided
        if weights is not None:
            if len(weights) != len(vectors):
                raise BindingError("Number of weights must match number of vectors")
            vectors = [v * w for v, w in zip(vectors, weights)]

        # Perform binding based on type
        if binding_type == BindingType.MULTIPLY:
            result = self._multiply_bind(vectors)
        elif binding_type == BindingType.CIRCULAR:
            result = self._circular_bind(vectors)
        elif binding_type == BindingType.PERMUTATION:
            result = self._permutation_bind(vectors)
        elif binding_type == BindingType.XOR:
            result = self._xor_bind(vectors)
        elif binding_type == BindingType.FOURIER:
            result = self._fourier_bind(vectors)
        else:
            raise BindingError("Unknown binding type: {binding_type}")

        logger.debug("Bound {len(vectors)} vectors using {binding_type.value}")
        return result

    def unbind(
        self,
        bound_vector: torch.Tensor,
        known_vectors: List[torch.Tensor],
        binding_type: BindingType = BindingType.CIRCULAR,
    ) -> torch.Tensor:
        """
        Unbind a vector given some known components

        Args:
            bound_vector: The bound hypervector
            known_vectors: List of known component vectors
            binding_type: Type of binding used

        Returns:
            The unknown component vector
        """
        if binding_type == BindingType.MULTIPLY:
            return self._multiply_unbind(bound_vector, known_vectors)
        elif binding_type == BindingType.CIRCULAR:
            return self._circular_unbind(bound_vector, known_vectors)
        elif binding_type == BindingType.PERMUTATION:
            return self._permutation_unbind(bound_vector, known_vectors)
        elif binding_type == BindingType.XOR:
            return self._xor_unbind(bound_vector, known_vectors)
        else:
            raise BindingError("Unbinding not implemented for {binding_type}")

    def _multiply_bind(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """Element-wise multiplication binding"""
        result = vectors[0].clone()
        for v in vectors[1:]:
            result = result * v
        return result

    def _multiply_unbind(
        self, bound_vector: torch.Tensor, known_vectors: List[torch.Tensor]
    ) -> torch.Tensor:
        """Unbind using element-wise division"""
        result = bound_vector.clone()
        for v in known_vectors:
            # Avoid division by zero
            result = result / (v + 1e-8)
        return result

    def _circular_bind(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """Circular convolution binding"""
        if len(vectors) == 1:
            return vectors[0]

        # Start with first vector
        result = vectors[0]

        # Bind remaining vectors using circular convolution
        for v in vectors[1:]:
            result = self._circular_convolve(result, v)

        return result

    def _circular_convolve(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform circular convolution of two vectors"""
        # Use FFT for efficient circular convolution
        X = torch.fft.fft(x)
        Y = torch.fft.fft(y)
        Z = X * Y
        z = torch.fft.ifft(Z).real
        return z

    def _circular_unbind(
        self, bound_vector: torch.Tensor, known_vectors: List[torch.Tensor]
    ) -> torch.Tensor:
        """Unbind using circular correlation (inverse of convolution)"""
        result = bound_vector

        for v in known_vectors:
            # Circular correlation is convolution with reversed vector
            v_reversed = torch.cat([v[:1], v[1:].flip(0)])
            result = self._circular_convolve(result, v_reversed)

        return result

    def _permutation_bind(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """Permutation-based binding"""
        result = torch.zeros_like(vectors[0])

        for i, v in enumerate(vectors):
            # Get permutation for this position
            perm = self._get_permutation(i)

            # Apply permutation and add
            result += v[perm]

        return result / len(vectors)

    def _permutation_unbind(
        self, bound_vector: torch.Tensor, known_vectors: List[torch.Tensor]
    ) -> torch.Tensor:
        """Unbind using inverse permutations"""
        # Subtract contributions of known vectors
        result = bound_vector * len(known_vectors + 1)

        for i, v in enumerate(known_vectors):
            perm = self._get_permutation(i)
            inv_perm = self._get_inverse_permutation(i)
            result -= v[perm]

        # Apply inverse permutation for unknown position
        unknown_pos = len(known_vectors)
        inv_perm = self._get_inverse_permutation(unknown_pos)

        return result[inv_perm]

    def _xor_bind(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """XOR binding for binary vectors"""
        # Convert to binary
        binary_vectors = [torch.sign(v) > 0 for v in vectors]

        # XOR all vectors
        result = binary_vectors[0]
        for v in binary_vectors[1:]:
            result = result ^ v

        # Convert back to {-1, +1}
        return result.float() * 2 - 1

    def _xor_unbind(
        self, bound_vector: torch.Tensor, known_vectors: List[torch.Tensor]
    ) -> torch.Tensor:
        """XOR unbinding (XOR is its own inverse)"""
        return self._xor_bind([bound_vector] + known_vectors)

    def _fourier_bind(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """Fourier domain binding"""
        # Transform to frequency domain
        freq_vectors = [torch.fft.fft(v) for v in vectors]

        # Multiply in frequency domain
        result_freq = freq_vectors[0]
        for fv in freq_vectors[1:]:
            result_freq = result_freq * fv

        # Transform back
        return torch.fft.ifft(result_freq).real

    def _get_permutation(self, position: int) -> torch.Tensor:
        """Get deterministic permutation for a position"""
        if position in self._permutation_cache:
            return self._permutation_cache[position]

        # Create deterministic permutation based on position
        torch.manual_seed(42 + position)  # Fixed seed + position
        perm = torch.randperm(self.dimension)

        self._permutation_cache[position] = perm
        return perm

    def _get_inverse_permutation(self, position: int) -> torch.Tensor:
        """Get inverse of a permutation"""
        perm = self._get_permutation(position)
        inv_perm = torch.zeros_like(perm)
        inv_perm[perm] = torch.arange(self.dimension)
        return inv_perm

    def bundle(self, vectors: List[torch.Tensor], normalize: bool = True) -> torch.Tensor:
        """
        Bundle vectors using superposition (addition)

        Args:
            vectors: List of hypervectors to bundle
            normalize: Whether to normalize the result

        Returns:
            Bundled hypervector
        """
        if not vectors:
            raise BindingError("No vectors provided for bundling")

        # Simple addition
        result = torch.stack(vectors).sum(dim=0)

        # Normalize if requested
        if normalize:
            result = result / torch.norm(result)

        return result

    def protect(self, vector: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Protect a hypervector using a key vector

        Args:
            vector: Hypervector to protect
            key: Key hypervector

        Returns:
            Protected hypervector
        """
        return self.bind([vector, key], BindingType.MULTIPLY)

    def unprotect(self, protected: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Unprotect a hypervector using the key

        Args:
            protected: Protected hypervector
            key: Key hypervector

        Returns:
            Original hypervector
        """
        return self.unbind(protected, [key], BindingType.MULTIPLY)


class PositionalBinder(HypervectorBinder):
    """
    Specialized binder for position-aware binding

    Used for encoding sequential information like genomic positions
    """

    def __init__(self, dimension: int = 10000, max_positions: int = 1000000):
        """
        Initialize positional binder

        Args:
            dimension: Hypervector dimension
            max_positions: Maximum number of positions to support
        """
        super().__init__(dimension)
        self.max_positions = max_positions
        self._position_vectors = {}

    def bind_with_position(self, vector: torch.Tensor, position: int) -> torch.Tensor:
        """
        Bind a vector with its position information

        Args:
            vector: Hypervector to bind
            position: Position index

        Returns:
            Position-bound hypervector
        """
        pos_vector = self._get_position_vector(position)
        return self.bind([vector, pos_vector], BindingType.CIRCULAR)

    def bind_sequence(self, vectors: List[torch.Tensor], start_position: int = 0) -> torch.Tensor:
        """
        Bind a sequence of vectors with their positions

        Args:
            vectors: List of hypervectors in sequence order
            start_position: Starting position index

        Returns:
            Sequence-bound hypervector
        """
        bound_vectors = []

        for i, v in enumerate(vectors):
            pos = start_position + i
            bound = self.bind_with_position(v, pos)
            bound_vectors.append(bound)

        # Bundle all position-bound vectors
        return self.bundle(bound_vectors)

    def _get_position_vector(self, position: int) -> torch.Tensor:
        """Get or create position encoding vector"""
        if position in self._position_vectors:
            return self._position_vectors[position]

        # Create position vector using sinusoidal encoding
        pos_vector = self._sinusoidal_position_encoding(position)

        # Cache for reuse
        self._position_vectors[position] = pos_vector

        return pos_vector

    def _sinusoidal_position_encoding(self, position: int) -> torch.Tensor:
        """Create sinusoidal position encoding"""
        encoding = torch.zeros(self.dimension)

        for i in range(0, self.dimension, 2):
            # Different frequencies for each dimension
            freq = 1.0 / (10000 ** (i / self.dimension))

            encoding[i] = np.sin(position * freq)
            if i + 1 < self.dimension:
                encoding[i + 1] = np.cos(position * freq)

        return encoding


class CrossModalBinder(HypervectorBinder):
    """
    Specialized binder for cross-modal binding

    Used for combining different types of omics data
    """

    def __init__(self, dimension: int = 10000):
        """Initialize cross-modal binder"""
        super().__init__(dimension)
        self.modality_signatures = {}

    def bind_modalities(
        self,
        modality_vectors: Dict[str, torch.Tensor],
        preserve_individual: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Bind multiple modalities together

        Args:
            modality_vectors: Dict mapping modality names to hypervectors
            preserve_individual: Whether to preserve individual modality access

        Returns:
            Dict containing bound vectors
        """
        results = {}

        # Create modality signatures if needed
        for modality in modality_vectors:
            if modality not in self.modality_signatures:
                self._create_modality_signature(modality)

        # Bind each modality with its signature
        signed_vectors = {}
        for modality, vector in modality_vectors.items():
            signature = self.modality_signatures[modality]
            signed = self.bind([vector, signature], BindingType.MULTIPLY)
            signed_vectors[modality] = signed

            if preserve_individual:
                results["{modality}_signed"] = signed

        # Create combined representation
        combined = self.bundle(list(signed_vectors.values()))
        results["combined"] = combined

        # Create pairwise combinations
        modalities = list(modality_vectors.keys())
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                m1, m2 = modalities[i], modalities[j]
                pair_name = "{m1}_{m2}"
                results[pair_name] = self.bind(
                    [signed_vectors[m1], signed_vectors[m2]], BindingType.CIRCULAR
                )

        return results

    def _create_modality_signature(self, modality: str):
        """Create unique signature for a modality"""
        # Use hash of modality name as seed
        seed = int(hashlib.md5(modality.encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)

        # Create random hypervector as signature
        signature = torch.randn(self.dimension)
        signature = signature / torch.norm(signature)

        self.modality_signatures[modality] = signature


# Convenience functions
def circular_bind(vectors: List[torch.Tensor]) -> torch.Tensor:
    """Convenience function for circular binding"""
    if not vectors:
        raise ValueError("No vectors provided")

    binder = HypervectorBinder(vectors[0].shape[-1])
    return binder.bind(vectors, BindingType.CIRCULAR)


def protect_vector(vector: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Convenience function for vector protection"""
    binder = HypervectorBinder(vector.shape[-1])
    return binder.protect(vector, key)
