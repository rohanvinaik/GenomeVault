"""
Bit-packed hypervector implementation for memory-efficient genomic encoding
"""

import numba
import numpy as np
import torch
from numba import cuda

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    from genomevault.observability.logging import configure_logging

    logger = configure_logging()
    logger.exception("Unhandled exception")
    cp = None
    CUPY_AVAILABLE = False
    raise

from genomevault.core.constants import HYPERVECTOR_DIMENSIONS
from genomevault.core.exceptions import HypervectorError


class PackedHV:
    """Bit-packed hypervector implementation with CPU/GPU support"""

    def __init__(self, n_bits: int, buffer: np.ndarray | None = None, device: str = "cpu"):
        self.n_bits = n_bits
        self.n_words = (n_bits + 63) // 64
        self.device = device

        if device == "cpu":
            self.buf = np.zeros(self.n_words, dtype=np.uint64) if buffer is None else buffer
        else:  # GPU
            if not CUPY_AVAILABLE:
                raise RuntimeError("CuPy not available for GPU operations")
            self.buf = (
                cp.zeros(self.n_words, dtype=cp.uint64) if buffer is None else cp.asarray(buffer)
            )

    def xor_(self, other: "PackedHV") -> "PackedHV":
        """In-place XOR (binding operation)"""
        if self.device == "cpu":
            np.bitwise_xor(self.buf, other.buf, out=self.buf)
        else:
            cp.bitwise_xor(self.buf, other.buf, out=self.buf)
        return self

    def xor(self, other: "PackedHV") -> "PackedHV":
        """XOR (binding operation) returning new hypervector"""
        result = PackedHV(self.n_bits, device=self.device)
        if self.device == "cpu":
            np.bitwise_xor(self.buf, other.buf, out=result.buf)
        else:
            cp.bitwise_xor(self.buf, other.buf, out=result.buf)
        return result

    def majority(self, others: list["PackedHV"]) -> "PackedHV":
        """Majority vote for bundling operation"""
        if self.device == "cpu":
            return self._majority_cpu(others)
        else:
            return self._majority_gpu(others)

    def _majority_cpu(self, others: list["PackedHV"]) -> "PackedHV":
        """CPU-optimized majority vote using bit manipulation"""
        result = PackedHV(self.n_bits)
        n_vecs = len(others) + 1  # Include self
        threshold = n_vecs // 2

        # Use Numba for faster execution
        result.buf = _majority_vote_numba(
            self.buf, [other.buf for other in others], self.n_words, threshold
        )

        return result

    def _majority_gpu(self, others: list["PackedHV"]) -> "PackedHV":
        """GPU-optimized majority vote"""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available for GPU operations")

        result = PackedHV(self.n_bits, device="gpu")
        n_vecs = len(others) + 1

        # Stack all vectors including self
        all_vecs = cp.stack([self.buf] + [other.buf for other in others])

        # Parallel majority computation
        result.buf = _gpu_majority_vote(all_vecs, n_vecs)

        return result

    def hamming_distance(self, other: "PackedHV") -> int:
        """Compute Hamming distance using hardware popcount"""
        if self.device == "cpu":
            return _hamming_distance_numba(self.buf, other.buf, self.n_words)
        else:
            return self._hamming_gpu(other)

    def _hamming_gpu(self, other: "PackedHV") -> int:
        """GPU Hamming distance"""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available for GPU operations")

        xor_result = cp.bitwise_xor(self.buf, other.buf)
        # Count bits using CuPy
        return int(cp.unpackbits(xor_result.view(cp.uint8)).sum())

    def to_dense(self) -> np.ndarray:
        """Convert to dense binary array for compatibility"""
        if self.device == "gpu" and CUPY_AVAILABLE:
            # Move to CPU first
            buf_cpu = cp.asnumpy(self.buf)
        else:
            buf_cpu = self.buf

        return _unpack_to_dense_numba(buf_cpu, self.n_bits)

    def to_torch(self) -> torch.Tensor:
        """Convert to PyTorch tensor"""
        dense = self.to_dense()
        return torch.from_numpy(dense).float()

    @staticmethod
    def from_dense(dense: np.ndarray, device: str = "cpu") -> "PackedHV":
        """Create from dense binary array"""
        n_bits = len(dense)
        packed = PackedHV(n_bits, device=device)

        if device == "cpu":
            packed.buf = _pack_from_dense_numba(dense, (n_bits + 63) // 64)
        else:
            if not CUPY_AVAILABLE:
                raise RuntimeError("CuPy not available for GPU operations")
            # Pack on CPU then move to GPU
            buf_cpu = _pack_from_dense_numba(dense, (n_bits + 63) // 64)
            packed.buf = cp.asarray(buf_cpu)

        return packed

    @staticmethod
    def from_torch(tensor: torch.Tensor, device: str = "cpu") -> "PackedHV":
        """Create from PyTorch tensor"""
        # Binarize if not already binary
        if tensor.dtype != torch.bool:
            tensor = tensor > 0
        dense = tensor.cpu().numpy().astype(np.uint8)
        return PackedHV.from_dense(dense, device)

    def clone(self) -> "PackedHV":
        """Create a copy of this hypervector"""
        if self.device == "cpu":
            return PackedHV(self.n_bits, buffer=self.buf.copy(), device=self.device)
        else:
            return PackedHV(self.n_bits, buffer=self.buf.copy(), device=self.device)

    @property
    def memory_bytes(self) -> int:
        """Return memory usage in bytes"""
        return self.buf.nbytes


# Numba JIT compiled functions for performance
@numba.jit(nopython=True, parallel=True)
def _majority_vote_numba(self_buf, other_bufs, n_words, threshold):
    """Numba-optimized majority vote"""
    result = np.zeros(n_words, dtype=np.uint64)
    n_others = len(other_bufs)

    for word_idx in numba.prange(n_words):
        # Process 64 bits at once
        for bit in range(64):
            count = 0
            mask = np.uint64(1) << bit

            # Count bit in self
            if self_buf[word_idx] & mask:
                count += 1

            # Count bit in others
            for i in range(n_others):
                if other_bufs[i][word_idx] & mask:
                    count += 1

            # Set bit if majority
            if count > threshold:
                result[word_idx] |= mask

    return result


@numba.jit(nopython=True)
def _hamming_distance_numba(buf1, buf2, n_words):
    """Numba-optimized Hamming distance using popcount"""
    distance = 0
    for i in range(n_words):
        xor_word = buf1[i] ^ buf2[i]
        # Use bit manipulation for popcount
        distance += _popcount64(xor_word)
    return distance


@numba.jit(nopython=True)
def _popcount64(x):
    """64-bit population count"""
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    x = x + (x >> 32)
    return x & 0x7F


@numba.jit(nopython=True)
def _unpack_to_dense_numba(buf, n_bits):
    """Unpack bit-packed buffer to dense array"""
    dense = np.zeros(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        word_idx = i // 64
        bit_idx = i % 64
        dense[i] = (buf[word_idx] >> bit_idx) & 1
    return dense


@numba.jit(nopython=True)
def _pack_from_dense_numba(dense, n_words):
    """Pack dense array to bit-packed buffer"""
    buf = np.zeros(n_words, dtype=np.uint64)
    n_bits = len(dense)

    for i in range(n_bits):
        if dense[i]:
            word_idx = i // 64
            bit_idx = i % 64
            buf[word_idx] |= np.uint64(1) << bit_idx

    return buf


# Generate Hamming distance lookup table
@numba.jit(nopython=True)
def _generate_hamming_lut():
    """Generate 16-bit Hamming distance lookup table"""
    lut = np.zeros(65536, dtype=np.uint8)
    for i in range(65536):
        # Count bits
        count = 0
        x = i
        while x:
            count += x & 1
            x >>= 1
        lut[i] = count
    return lut


# Pre-compute lookup table
HAMMING_LUT = _generate_hamming_lut()


@numba.jit(nopython=True)
def fast_hamming_distance(buf1: np.ndarray, buf2: np.ndarray) -> int:
    """Ultra-fast Hamming distance using 16-bit LUT"""
    distance = 0
    for i in range(len(buf1)):
        xor_word = buf1[i] ^ buf2[i]
        # Process 16 bits at a time
        distance += HAMMING_LUT[xor_word & 0xFFFF]
        distance += HAMMING_LUT[(xor_word >> 16) & 0xFFFF]
        distance += HAMMING_LUT[(xor_word >> 32) & 0xFFFF]
        distance += HAMMING_LUT[(xor_word >> 48) & 0xFFFF]
    return distance


# GPU functions (if CuPy available)
if CUPY_AVAILABLE:

    def _gpu_majority_vote(all_vecs, n_vecs):
        """GPU-accelerated majority vote"""
        n_words = all_vecs.shape[1]
        result = cp.zeros(n_words, dtype=cp.uint64)
        threshold = n_vecs // 2

        # Process each bit position
        for bit in range(64):
            mask = cp.uint64(1) << bit
            # Count bits across all vectors
            bit_counts = ((all_vecs & mask) != 0).sum(axis=0)
            # Set bits where count > threshold
            result |= (bit_counts > threshold).astype(cp.uint64) << bit

        return result


class PackedProjection:
    """Bit-level projection for genomic features"""

    def __init__(self, input_dim: int, hv_dim: int, seed: int = 42):
        self.input_dim = input_dim
        self.hv_dim = hv_dim
        self.n_words = (hv_dim + 63) // 64

        # Generate projection masks
        rng = np.random.RandomState(seed)
        self.masks = np.zeros((input_dim, self.n_words), dtype=np.uint64)

        for i in range(input_dim):
            for j in range(self.n_words):
                self.masks[i, j] = rng.randint(0, 2**64, dtype=np.uint64)

    def encode(self, features: np.ndarray, device: str = "cpu") -> PackedHV:
        """Project features to packed hypervector"""
        result = PackedHV(self.hv_dim, device=device)

        if device == "cpu":
            result.buf = self._encode_cpu(features)
        else:
            result.buf = self._encode_gpu(features)

        return result

    def _encode_cpu(self, features: np.ndarray) -> np.ndarray:
        """CPU encoding using Numba"""
        return _project_features_numba(features, self.masks, self.n_words, len(features))

    def _encode_gpu(self, features: np.ndarray) -> cp.ndarray:
        """GPU encoding"""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available for GPU operations")

        # Move data to GPU
        features_gpu = cp.asarray(features)
        masks_gpu = cp.asarray(self.masks)

        # Compute projection
        accumulator = cp.zeros(self.n_words, dtype=cp.float32)

        for i, val in enumerate(features_gpu):
            if val > 0:
                weight = cp.minimum(val * 16, 16).astype(cp.int32)
                for j in range(self.n_words):
                    # Count bits in mask
                    mask_bits = cp.unpackbits(masks_gpu[i, j].view(cp.uint8)).sum()
                    accumulator[j] += mask_bits * weight

        # Binarize
        threshold = len(features) * 8  # Adjusted threshold
        result = cp.zeros(self.n_words, dtype=cp.uint64)

        for j in range(self.n_words):
            for bit in range(64):
                if accumulator[j] > threshold:
                    result[j] |= cp.uint64(1) << bit

        return result


@numba.jit(nopython=True, parallel=True)
def _project_features_numba(features, masks, n_words, n_features):
    """Numba-optimized feature projection"""
    accumulator = np.zeros(n_words, dtype=np.float32)

    # Accumulate weighted projections
    for i in range(n_features):
        if features[i] > 0:
            weight = min(features[i] * 16, 16)
            for j in numba.prange(n_words):
                # Count bits in mask
                mask_bits = _popcount64(masks[i, j])
                accumulator[j] += mask_bits * weight

    # Binarize
    threshold = n_features * 8
    result = np.zeros(n_words, dtype=np.uint64)

    for j in range(n_words):
        acc_val = accumulator[j]
        for bit in range(64):
            bit_threshold = threshold * (bit + 1) / 64
            if acc_val > bit_threshold:
                result[j] |= np.uint64(1) << bit

    return result


class PackedGenomicEncoder:
    """
    Genomic encoder using bit-packed hypervectors
    Drop-in replacement for GenomicEncoder with packed option
    """

    def __init__(
        self,
        dimension: int = HYPERVECTOR_DIMENSIONS["base"],
        packed: bool = True,
        device: str = "cpu",
        **kwargs,
    ):
        self.dimension = dimension
        self.packed = packed
        self.device = device

        if packed:
            # Initialize packed components
            self.projection = PackedProjection(
                input_dim=1000,
                hv_dim=dimension,  # Adjust based on your needs
            )
            self._init_packed_base_vectors()
        else:
            # Fall back to standard GenomicEncoder
            from genomevault.hypervector.encoding.genomic import GenomicEncoder

            self._fallback_encoder = GenomicEncoder(dimension=dimension, **kwargs)

    def _init_packed_base_vectors(self):
        """Initialize packed base vectors"""
        self.base_vectors = {}

        # Create orthogonal base vectors for nucleotides
        for i, base in enumerate(["A", "T", "G", "C"]):
            vec = PackedHV(self.dimension, device=self.device)
            # Set specific bits for orthogonality
            start_bit = i * (self.dimension // 4)
            end_bit = (i + 1) * (self.dimension // 4)

            # Create dense representation
            dense = np.zeros(self.dimension, dtype=np.uint8)
            dense[start_bit:end_bit] = 1

            self.base_vectors[base] = PackedHV.from_dense(dense, self.device)

        # Create random vectors for variant types
        rng = np.random.RandomState(42)
        for variant_type in ["SNP", "INS", "DEL", "DUP", "INV"]:
            dense = rng.randint(0, 2, size=self.dimension).astype(np.uint8)
            self.base_vectors[variant_type] = PackedHV.from_dense(dense, self.device)

    def encode_variant(
        self,
        chromosome: str,
        position: int,
        ref: str,
        alt: str,
        variant_type: str = "SNP",
        **kwargs,
    ) -> PackedHV | torch.Tensor:
        """Encode a single variant"""

        if not self.packed:
            return self._fallback_encoder.encode_variant(
                chromosome, position, ref, alt, variant_type, **kwargs
            )

        try:
            # Start with variant type vector
            variant_vec = self.base_vectors[variant_type].clone()

            # Create position encoding
            pos_vec = self._encode_position(position)
            variant_vec = variant_vec.xor(pos_vec)

            # Add reference allele if single nucleotide
            if len(ref) == 1 and ref in self.base_vectors:
                variant_vec = variant_vec.xor(self.base_vectors[ref])

            # Add alternative allele with permutation
            if len(alt) == 1 and alt in self.base_vectors:
                alt_vec = self._permute_packed(self.base_vectors[alt], 1)
                variant_vec = variant_vec.xor(alt_vec)

            # Add chromosome information
            chr_vec = self._chromosome_vector_packed(chromosome)
            variant_vec = self._bundle_packed([variant_vec, chr_vec])

            return variant_vec

        except Exception as e:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            raise HypervectorError(f"Failed to encode variant: {e!s}")
            raise

    def _encode_position(self, position: int) -> PackedHV:
        """Encode genomic position"""
        # Use position as seed for deterministic random vector
        rng = np.random.RandomState(position % 2**32)
        dense = rng.randint(0, 2, size=self.dimension).astype(np.uint8)
        return PackedHV.from_dense(dense, self.device)

    def _permute_packed(self, vec: PackedHV, n: int) -> PackedHV:
        """Permute packed hypervector by n positions"""
        # Convert to dense, permute, convert back
        # TODO: Implement direct bit-level permutation for efficiency
        dense = vec.to_dense()
        permuted = np.roll(dense, n)
        return PackedHV.from_dense(permuted, self.device)

    def _chromosome_vector_packed(self, chromosome: str) -> PackedHV:
        """Generate chromosome-specific packed vector"""
        chr_num = chromosome.replace("chr", "").replace("X", "23").replace("Y", "24")
        try:
            chr_idx = int(chr_num)
        except (ValueError, Exception):
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            chr_idx = 25
            raise

        rng = np.random.RandomState(chr_idx)
        dense = rng.randint(0, 2, size=self.dimension).astype(np.uint8)
        return PackedHV.from_dense(dense, self.device)

    def _bundle_packed(self, vectors: list[PackedHV]) -> PackedHV:
        """Bundle packed vectors using majority vote"""
        if len(vectors) == 1:
            return vectors[0]
        return vectors[0].majority(vectors[1:])

    def encode_genome(self, variants: list[dict]) -> PackedHV | torch.Tensor:
        """Encode complete genome"""

        if not self.packed:
            return self._fallback_encoder.encode_genome(variants)

        if not variants:
            return PackedHV(self.dimension, device=self.device)

        # Encode each variant
        variant_vectors = []
        for variant in variants:
            vec = self.encode_variant(
                chromosome=variant["chromosome"],
                position=variant["position"],
                ref=variant["ref"],
                alt=variant["alt"],
                variant_type=variant.get("type", "SNP"),
            )
            variant_vectors.append(vec)

        # Bundle all variants
        genome_vec = self._bundle_packed(variant_vectors)

        return genome_vec

    def similarity(self, vec1: PackedHV | torch.Tensor, vec2: PackedHV | torch.Tensor) -> float:
        """Calculate similarity between hypervectors"""

        if isinstance(vec1, PackedHV) and isinstance(vec2, PackedHV):
            # Use Hamming distance
            distance = vec1.hamming_distance(vec2)
            return 1.0 - (2.0 * distance / self.dimension)
        elif isinstance(vec1, torch.Tensor) and isinstance(vec2, torch.Tensor):
            # Use cosine similarity
            return torch.cosine_similarity(vec1, vec2, dim=0).item()
        else:
            # Convert to compatible format
            if isinstance(vec1, PackedHV):
                vec1 = vec1.to_torch()
            if isinstance(vec2, PackedHV):
                vec2 = vec2.to_torch()
            return torch.cosine_similarity(vec1, vec2, dim=0).item()

    @property
    def memory_efficiency(self) -> float:
        """Return memory efficiency compared to dense representation"""
        if self.packed:
            # 1 bit per dimension vs 32 bits (float32)
            return 32.0
        else:
            return 1.0


# GPU kernel compilation (if CUDA available)
if cuda.is_available():

    @cuda.jit
    def packed_xor_kernel(a, b, result):
        """GPU kernel for packed XOR"""
        idx = cuda.grid(1)
        if idx < a.shape[0]:
            result[idx] = a[idx] ^ b[idx]

    @cuda.jit
    def packed_majority_kernel(vectors, result, n_vectors):
        """GPU kernel for majority vote"""
        word_idx = cuda.grid(1)
        if word_idx < result.shape[0]:
            # Process one word across all vectors
            for bit in range(64):
                count = 0
                mask = np.uint64(1) << bit

                for vec_idx in range(n_vectors):
                    if vectors[vec_idx, word_idx] & mask:
                        count += 1

                if count > n_vectors // 2:
                    result[word_idx] |= mask
