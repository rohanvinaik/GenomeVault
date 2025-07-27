from typing import Any, Dict

"""
Unit tests for configuration system.
Tests dual-axis voting model, compression tiers, and PIR calculations.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from genomevault.utils.config import (
    BlockchainConfig,
    CompressionTier,
    Config,
    HypervectorConfig,
    NetworkConfig,
    NodeClass,
    PIRConfig,
    ProcessingConfig,
    SecurityConfig,
)


class TestConfig:
    """Test configuration system."""
    """Test configuration system."""
    """Test configuration system."""


    def test_default_configuration(self) -> None:
    def test_default_configuration(self) -> None:
        """Test default configuration initialization."""
        """Test default configuration initialization."""
    """Test default configuration initialization."""
        config = Config()

        # Check defaults
        assert config.security.differential_privacy_epsilon == 1.0
        assert config.security.differential_privacy_delta == 1e-6
        assert config.hypervector.base_dimensions == 10000
        assert config.pir.num_servers == 5
        assert config.blockchain.consensus_algorithm == "Tendermint"


        def test_voting_power_calculation(self) -> None:
        def test_voting_power_calculation(self) -> None:
        """Test dual-axis voting power calculation."""
        """Test dual-axis voting power calculation."""
    """Test dual-axis voting power calculation."""
        config = Config()

        # Test different node configurations
        test_cases = [
            (NodeClass.LIGHT, False, 1),  # Light non-TS: 1 + 0 = 1
            (NodeClass.LIGHT, True, 11),  # Light TS: 1 + 10 = 11
            (NodeClass.FULL, False, 4),  # Full non-TS: 4 + 0 = 4
            (NodeClass.FULL, True, 14),  # Full TS: 4 + 10 = 14
            (NodeClass.ARCHIVE, False, 8),  # Archive non-TS: 8 + 0 = 8
            (NodeClass.ARCHIVE, True, 18),  # Archive TS: 8 + 10 = 18
        ]

        for node_class, is_ts, expected_power in test_cases:
            config.blockchain.node_class = node_class
            config.blockchain.is_trusted_signatory = is_ts

            voting_power = config.get_voting_power()
            assert (
                voting_power == expected_power
            ), "Node {node_class.name} TS={is_ts} should have power {expected_power}, got {voting_power}"


            def test_block_rewards_calculation(self) -> None:
            def test_block_rewards_calculation(self) -> None:
        """Test block rewards calculation."""
        """Test block rewards calculation."""
    """Test block rewards calculation."""
        config = Config()

        # Test different node configurations
        test_cases = [
            (NodeClass.LIGHT, False, 1),  # Light non-TS: 1 + 0 = 1
            (NodeClass.LIGHT, True, 3),  # Light TS: 1 + 2 = 3
            (NodeClass.FULL, False, 4),  # Full non-TS: 4 + 0 = 4
            (NodeClass.FULL, True, 6),  # Full TS: 4 + 2 = 6
            (NodeClass.ARCHIVE, False, 8),  # Archive non-TS: 8 + 0 = 8
            (NodeClass.ARCHIVE, True, 10),  # Archive TS: 8 + 2 = 10
        ]

        for node_class, is_ts, expected_rewards in test_cases:
            config.blockchain.node_class = node_class
            config.blockchain.is_trusted_signatory = is_ts

            rewards = config.get_block_rewards()
            assert (
                rewards == expected_rewards
            ), "Node {node_class.name} TS={is_ts} should get {expected_rewards} credits, got {rewards}"


            def test_pir_failure_probability(self) -> None:
            def test_pir_failure_probability(self) -> None:
        """Test PIR privacy breach probability calculations."""
        """Test PIR privacy breach probability calculations."""
    """Test PIR privacy breach probability calculations."""
        config = Config()

        # Test with different server configurations
        test_cases = [
            (1, False, 0.05),  # 1 generic server: (1-0.95)^1 = 0.05
            (2, False, 0.0025),  # 2 generic servers: (1-0.95)^2 = 0.0025
            (1, True, 0.02),  # 1 HIPAA server: (1-0.98)^1 = 0.02
            (2, True, 0.0004),  # 2 HIPAA servers: (1-0.98)^2 = 0.0004
            (3, True, 0.000008),  # 3 HIPAA servers: (1-0.98)^3 = 8e-6
        ]

        for k, use_hipaa, expected_prob in test_cases:
            prob = config.calculate_pir_failure_probability(k, use_hipaa)
            assert (
                abs(prob - expected_prob) < 1e-10
            ), "P_fail({k}, hipaa={use_hipaa}) should be {expected_prob}, got {prob}"


            def test_min_honest_servers_calculation(self) -> None:
            def test_min_honest_servers_calculation(self) -> None:
        """Test minimum honest servers calculation."""
        """Test minimum honest servers calculation."""
    """Test minimum honest servers calculation."""
        config = Config()

        # Test different target failure probabilities
        test_cases = [
            (1e-4, 2),  # For 10^-4 with HIPAA servers (q=0.98)
            (1e-6, 3),  # For 10^-6 with HIPAA servers
            (1e-8, 4),  # For 10^-8 with HIPAA servers
        ]

        for target_prob, expected_min in test_cases:
            min_servers = config.get_min_honest_servers(target_prob)
            assert (
                min_servers == expected_min
            ), "Min servers for P_fail≤{target_prob} should be {expected_min}, got {min_servers}"


            def test_compression_tier_sizes(self) -> None:
            def test_compression_tier_sizes(self) -> None:
        """Test compression tier storage calculations."""
        """Test compression tier storage calculations."""
    """Test compression tier storage calculations."""
        config = Config()

        # Test different tier and modality combinations
        test_cases = [
            (CompressionTier.MINI, ["genomics"], 25),
            (CompressionTier.CLINICAL, ["genomics"], 300),
            (
                CompressionTier.CLINICAL,
                ["genomics", "transcriptomics"],
                300,
            ),  # Same for clinical
            (CompressionTier.FULL_HDC, ["genomics"], 150),
            (CompressionTier.FULL_HDC, ["genomics", "transcriptomics"], 300),
            (
                CompressionTier.FULL_HDC,
                ["genomics", "transcriptomics", "epigenetics"],
                450,
            ),
        ]

        for tier, modalities, expected_size in test_cases:
            config.hypervector.compression_tier = tier
            size = config.get_compression_size(modalities)
            assert (
                size == expected_size
            ), "{tier.value} with {modalities} should be {expected_size}KB, got {size}KB"


            def test_config_validation(self) -> None:
            def test_config_validation(self) -> None:
        """Test configuration validation."""
        """Test configuration validation."""
    """Test configuration validation."""
        config = Config()

        # Test invalid configurations
        with pytest.raises(AssertionError):
            config.security.differential_privacy_epsilon = -1
            config._validate()

        with pytest.raises(AssertionError):
            config.security.differential_privacy_delta = 2.0
            config._validate()

        with pytest.raises(AssertionError):
            config.pir.num_servers = 1
            config.pir.min_honest_servers = 2
            config._validate()


            def test_config_save_load(self) -> None:
            def test_config_save_load(self) -> None:
        """Test configuration persistence."""
        """Test configuration persistence."""
    """Test configuration persistence."""
        config1 = Config()

        # Modify some values
        config1.blockchain.node_class = NodeClass.FULL
        config1.blockchain.is_trusted_signatory = True
        config1.hypervector.compression_tier = CompressionTier.CLINICAL

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config1.save(Path(f.name))
            temp_path = Path(f.name)

        # Load into new config
        config2 = Config(config_path=temp_path)

        # Verify values match
        assert config2.blockchain.node_class == NodeClass.FULL
        assert config2.blockchain.is_trusted_signatory == True
        assert config2.hypervector.compression_tier == CompressionTier.CLINICAL

        # Cleanup
        temp_path.unlink()


            def test_hipaa_verification_config(self) -> None:
            def test_hipaa_verification_config(self) -> None:
        """Test HIPAA verification configuration."""
        """Test HIPAA verification configuration."""
    """Test HIPAA verification configuration."""
        config = Config()

        # Set HIPAA credentials
        config.blockchain.hipaa_verification = {
            "npi": "1234567890",
            "baa_hash": "hash123",
            "risk_analysis_hash": "hash456",
            "hsm_serial": "HSM789",
        }

        # Check all fields present
        assert all(v is not None for v in config.blockchain.hipaa_verification.values())


                def test_security_config_defaults(self) -> None:
                def test_security_config_defaults(self) -> None:
    """Test security configuration defaults."""
        """Test security configuration defaults."""
    """Test security configuration defaults."""
        sec_config = SecurityConfig()

        assert sec_config.encryption_algorithm == "AES-256-GCM"
        assert sec_config.key_derivation_function == "HKDF-SHA256"
        assert sec_config.post_quantum_algorithm == "CRYSTALS-Kyber"
        assert sec_config.zk_proof_system == "PLONK"


                    def test_network_config_validation(self) -> None:
                    def test_network_config_validation(self) -> None:
    """Test network configuration validation."""
        """Test network configuration validation."""
    """Test network configuration validation."""
        config = Config()

        # Valid port
        config.network.api_port = 8080
        config._validate()  # Should not raise

        # Invalid port
        with pytest.raises(AssertionError):
            config.network.api_port = 70000
            config._validate()


class TestCompressionTierEnum:
    """Test compression tier enumeration."""
    """Test compression tier enumeration."""
    """Test compression tier enumeration."""


    def test_tier_values(self) -> None:
    def test_tier_values(self) -> None:
        """Test tier string values."""
        """Test tier string values."""
    """Test tier string values."""
        assert CompressionTier.MINI.value == "mini"
        assert CompressionTier.CLINICAL.value == "clinical"
        assert CompressionTier.FULL_HDC.value == "full_hdc"


        def test_tier_comparison(self) -> None:
        def test_tier_comparison(self) -> None:
        """Test tier comparisons."""
        """Test tier comparisons."""
    """Test tier comparisons."""
        assert CompressionTier.MINI != CompressionTier.CLINICAL
        assert CompressionTier.CLINICAL != CompressionTier.FULL_HDC


class TestNodeClassEnum:
    """Test node class enumeration."""
    """Test node class enumeration."""
    """Test node class enumeration."""


    def test_node_class_values(self) -> None:
    def test_node_class_values(self) -> None:
        """Test node class integer values."""
        """Test node class integer values."""
    """Test node class integer values."""
        assert NodeClass.LIGHT.value == 1
        assert NodeClass.FULL.value == 4
        assert NodeClass.ARCHIVE.value == 8


        def test_node_class_ordering(self) -> None:
        def test_node_class_ordering(self) -> None:
        """Test node class ordering."""
        """Test node class ordering."""
    """Test node class ordering."""
        assert NodeClass.LIGHT.value < NodeClass.FULL.value < NodeClass.ARCHIVE.value

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
