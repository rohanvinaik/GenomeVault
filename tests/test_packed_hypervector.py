from genomevault.utils.logging import get_logger
logger = get_logger(__name__)


"""
Tests for bit-packed hypervector implementation
"""

import numpy as np
import pytest
import torch

from genomevault.hypervector.encoding import (
    GenomicEncoder,
    PackedGenomicEncoder,
    PackedHV,
)


class TestPackedHV:
    """Test PackedHV functionality"""

    def test_creation(self):
        """Test hypervector creation"""
        hv = PackedHV(10000)
        assert hv.n_bits == 10000
        assert hv.n_words == 157  # (10000 + 63) // 64
        assert hv.memory_bytes == 157 * 8  # 1256 bytes

    def test_xor_operation(self):
        """Test XOR binding operation"""
        hv1 = PackedHV(1000)
        hv2 = PackedHV(1000)

        # Set some bits
        hv1.buf[0] = 0b1010101010101010
        hv2.buf[0] = 0b1100110011001100

        result = hv1.xor(hv2)
        expected = 0b1010101010101010 ^ 0b1100110011001100
        assert result.buf[0] == expected

    def test_majority_vote(self):
        """Test majority vote bundling"""
        # Create 3 hypervectors
        hv1 = PackedHV(64)
        hv2 = PackedHV(64)
        hv3 = PackedHV(64)

        # Set bits - bit 0 appears in 2/3, bit 1 in 1/3
        hv1.buf[0] = 0b0001
        hv2.buf[0] = 0b0001
        hv3.buf[0] = 0b0010

        result = hv1.majority([hv2, hv3])
        # Bit 0 should be set (majority), bit 1 should not
        assert (result.buf[0] & 0b0001) != 0
        assert (result.buf[0] & 0b0010) == 0

    def test_hamming_distance(self):
        """Test Hamming distance calculation"""
        hv1 = PackedHV(64)
        hv2 = PackedHV(64)

        hv1.buf[0] = 0b1111000011110000
        hv2.buf[0] = 0b1010101010101010

        distance = hv1.hamming_distance(hv2)
        expected = bin(0b1111000011110000 ^ 0b1010101010101010).count("1")
        assert distance == expected

    def test_dense_conversion(self):
        """Test conversion to/from dense representation"""
        # Create dense array
        dense = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)

        # Convert to packed
        packed = PackedHV.from_dense(dense)
        assert packed.buf[0] == 0b10101010

        # Convert back to dense
        dense_recovered = packed.to_dense()
        np.testing.assert_array_equal(dense_recovered[:8], dense)

    def test_torch_compatibility(self):
        """Test PyTorch tensor conversion"""
        # Create torch tensor
        tensor = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0], dtype=torch.float32)

        # Convert to packed
        packed = PackedHV.from_torch(tensor)
        assert packed.buf[0] == 0b00101101

        # Convert back to torch
        tensor_recovered = packed.to_torch()
        expected = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0], dtype=torch.float32)
        torch.testing.assert_close(tensor_recovered[:8], expected)


class TestPackedGenomicEncoder:
    """Test genomic encoding with packed hypervectors"""

    @pytest.fixture
    def sample_variants(self):
        """Sample genomic variants for testing"""
        return [
            {
                "chromosome": "chr1",
                "position": 12345,
                "ref": "A",
                "alt": "G",
                "type": "SNP",
            },
            {
                "chromosome": "chr1",
                "position": 23456,
                "ref": "C",
                "alt": "T",
                "type": "SNP",
            },
            {
                "chromosome": "chr2",
                "position": 34567,
                "ref": "G",
                "alt": "A",
                "type": "SNP",
            },
        ]

    def test_encoder_creation(self):
        """Test encoder initialization"""
        encoder = PackedGenomicEncoder(dimension=10000, packed=True, device="cpu")
        assert encoder.dimension == 10000
        assert encoder.packed is True
        assert encoder.memory_efficiency == 32.0

    def test_variant_encoding(self, sample_variants):
        """Test single variant encoding"""
        encoder = PackedGenomicEncoder(dimension=10000)

        variant = sample_variants[0]
        hv = encoder.encode_variant(
            chromosome=variant["chromosome"],
            position=variant["position"],
            ref=variant["ref"],
            alt=variant["alt"],
            variant_type=variant["type"],
        )

        assert isinstance(hv, PackedHV)
        assert hv.n_bits == 10000

    def test_genome_encoding(self, sample_variants):
        """Test complete genome encoding"""
        encoder = PackedGenomicEncoder(dimension=10000)

        genome_hv = encoder.encode_genome(sample_variants)

        assert isinstance(genome_hv, PackedHV)
        assert genome_hv.n_bits == 10000

    def test_similarity_computation(self, sample_variants):
        """Test similarity between packed hypervectors"""
        encoder = PackedGenomicEncoder(dimension=10000)

        # Encode two similar genomes
        genome1_hv = encoder.encode_genome(sample_variants[:2])
        genome2_hv = encoder.encode_genome(sample_variants[:2])

        # Should be identical
        similarity = encoder.similarity(genome1_hv, genome2_hv)
        assert similarity == 1.0

        # Encode different genome
        genome3_hv = encoder.encode_genome(sample_variants[2:])

        # Should be less similar
        similarity2 = encoder.similarity(genome1_hv, genome3_hv)
        assert 0 < similarity2 < 1.0

    def test_memory_comparison(self, sample_variants):
        """Compare memory usage between packed and unpacked"""
        dimension = 10000

        # Packed encoder
        packed_encoder = PackedGenomicEncoder(dimension=dimension, packed=True)
        packed_hv = packed_encoder.encode_genome(sample_variants)
        packed_memory = packed_hv.memory_bytes

        # Unpacked encoder
        unpacked_encoder = GenomicEncoder(dimension=dimension)
        unpacked_hv = unpacked_encoder.encode_genome(sample_variants)
        unpacked_memory = unpacked_hv.element_size() * unpacked_hv.nelement()

        # Packed should use ~8x less memory
        compression_ratio = unpacked_memory / packed_memory
        assert compression_ratio > 7  # Allow some overhead

        print(
            f"Memory usage - Packed: {packed_memory} bytes, "
            f"Unpacked: {unpacked_memory} bytes, "
            f"Compression: {compression_ratio:.1f}x"
        )

    def test_compatibility_mode(self, sample_variants):
        """Test that packed encoder can fall back to standard mode"""
        encoder = PackedGenomicEncoder(dimension=10000, packed=False)

        # Should return torch tensor in compatibility mode
        hv = encoder.encode_variant(chromosome="chr1", position=12345, ref="A", alt="G")

        assert isinstance(hv, torch.Tensor)
        assert hv.shape[0] == 10000


class TestPerformance:
    """Performance benchmarks for packed implementation"""

    @pytest.mark.benchmark
    def test_encoding_speed(self, benchmark, sample_variants):
        """Benchmark encoding speed"""
        encoder = PackedGenomicEncoder(dimension=10000)

        def encode():
            return encoder.encode_genome(sample_variants * 100)  # 300 variants

        result = benchmark(encode)
        assert isinstance(result, PackedHV)

    @pytest.mark.benchmark
    def test_similarity_speed(self, benchmark):
        """Benchmark similarity computation"""
        dimension = 10000
        hv1 = PackedHV(dimension)
        hv2 = PackedHV(dimension)

        # Randomize bits
        rng = np.random.RandomState(42)
        hv1.buf = rng.randint(0, 2**64, size=hv1.n_words, dtype=np.uint64)
        hv2.buf = rng.randint(0, 2**64, size=hv2.n_words, dtype=np.uint64)

        def compute_similarity():
            return hv1.hamming_distance(hv2)

        result = benchmark(compute_similarity)
        assert 0 <= result <= dimension


if __name__ == "__main__":
    # Run basic tests
    logger.debug("Testing PackedHV implementation...")

    # Test creation and memory
    hv = PackedHV(10000)
    logger.debug(f"Created {hv.n_bits}-bit hypervector using {hv.memory_bytes} bytes")
    logger.debug(f"Compression vs float32: {10000 * 4 / hv.memory_bytes:.1f}x")

    # Test genomic encoding
    encoder = PackedGenomicEncoder(dimension=10000)
    variants = [
        {
            "chromosome": "chr1",
            "position": 12345,
            "ref": "A",
            "alt": "G",
            "type": "SNP",
        },
        {
            "chromosome": "chr2",
            "position": 67890,
            "ref": "C",
            "alt": "T",
            "type": "SNP",
        },
    ]

    genome_hv = encoder.encode_genome(variants)
    logger.debug(f"\nEncoded {len(variants)} variants into packed hypervector")
    logger.debug(f"Memory efficiency: {encoder.memory_efficiency}x")
