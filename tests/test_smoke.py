"""Smoke tests for GenomeVault core functionality."""

import pytest

import numpy as np
import pandas as pd


def test_imports():
    """Test that all core modules can be imported."""
    import genomevault

    assert genomevault is not None


def test_local_processing():
    """Test local processing modules."""
    from genomevault.local_processing import epigenetics, proteomics, transcriptomics

    # Create sample data
    data = pd.DataFrame(np.random.randn(10, 5))

    # Test each processor
    for processor in [epigenetics, proteomics, transcriptomics]:
        result = processor.process(data, {})
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 5)
        assert np.all(np.isfinite(result))


def test_hdc_encoding():
    """Test hyperdimensional computing encoding."""
    try:
        from genomevault.hdc import bundle, encode, similarity

        # Test encoding
        X = np.random.randn(5, 3)
        encoded = encode(X, seed=42)
        assert encoded.shape[0] == 5

        # Test bundling
        bundled = bundle(encoded, normalize=True)
        assert bundled.ndim == 1

        # Test similarity
        sim = similarity(bundled, bundled)
        assert 0.99 <= sim <= 1.01  # Should be ~1 for same vector
    except ImportError:
        pytest.skip("HDC module not available")


def test_zk_proofs():
    """Test zero-knowledge proof generation."""
    try:
        from genomevault.zk_proofs import prove, verify

        # Generate proof
        proof = prove({"data": "test"})
        assert "proof" in proof
        assert "public" in proof

        # Verify proof
        result = verify(proof)
        assert result is True
    except ImportError:
        pytest.skip("ZK proofs module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
