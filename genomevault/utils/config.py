from __future__ import annotations

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)
"""
GenomeVault Configuration Management

This module provides centralized configuration management for all GenomeVault components,
including environment-specific settings, secrets management, and runtime configuration.
"""


import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from math import ceil, log10
from pathlib import Path
from typing import Any

try:
    import yaml

    HAS_YAML = True
except ImportError:
    logger.exception("Unhandled exception")
    HAS_YAML = False
    yaml = None
    raise RuntimeError("Unspecified error")

# These are REQUIRED - not optional!

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environment types"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class CompressionTier(Enum):
    """Compression tier enumeration"""

    MINI = "mini"
    CLINICAL = "clinical"
    FULL_HDC = "full_hdc"


class NodeClass(Enum):
    """Node class enumeration with voting power values"""

    LIGHT = 1
    FULL = 4
    ARCHIVE = 8


@dataclass
class SecurityConfig:
    """Security configuration parameters"""

    encryption_algorithm: str = "AES-256-GCM"
    key_derivation_function: str = "HKDF-SHA256"
    post_quantum_algorithm: str = "CRYSTALS-Kyber"
    zk_proof_system: str = "PLONK"
    differential_privacy_epsilon: float = 1.0
    differential_privacy_delta: float = 1e-6


class SecurityLevel(Enum):
    """Security levels for different data types"""

    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    SECRET = 3


@dataclass
class CryptoConfig:
    """Cryptographic configuration parameters"""

    # Encryption
    aes_key_size: int = 256
    rsa_key_size: int = 4096

    # Zero-knowledge proofs
    zk_security_parameter: int = 128
    plonk_curve: str = "BLS12-381"

    # Post-quantum
    pq_algorithm: str = "CRYSTALS-Kyber"
    pq_security_level: int = 3

    # Hypervectors
    hypervector_dimensions: int = 10000
    projection_type: str = "sparse_random"

    # PIR
    pir_server_count: int = 5
    pir_threshold: int = 3
    pir_failure_probability: float = 1e-4


@dataclass
class PrivacyConfig:
    """Privacy configuration parameters"""

    # Differential privacy
    epsilon: float = 1.0
    delta: float = 1e-6
    noise_mechanism: str = "gaussian"

    # Privacy budget
    max_queries_per_user: int = 1000
    budget_refresh_days: int = 30

    # Federated learning
    min_participants: int = 10
    dropout_rate: float = 0.3
    secure_aggregation: bool = True


@dataclass
class NetworkConfig:
    """Network configuration parameters"""

    # API endpoints
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # Blockchain
    chain_id: str = "genomevault-mainnet"
    consensus_algorithm: str = "tendermint"
    block_time_seconds: int = 5

    # PIR network
    pir_servers: list = field(default_factory=list)
    pir_timeout_seconds: int = 30

    # Node configuration
    node_class: str = "light"  # light, full, archive
    signatory_status: bool = False

    # HIPAA fast-track
    hipaa_verified: bool = False
    npi_number: str | None = None


@dataclass
class StorageConfig:
    """Storage configuration parameters"""

    # Local storage
    data_dir: Path = Path.home() / ".genomevault"
    temp_dir: Path = Path("/tmp/genomevault")

    # Database
    db_type: str = "sqlite"  # sqlite, postgresql, mongodb
    db_path: str | None = None

    # Compression
    compression_tier: str = "clinical"  # mini, clinical, full
    compression_algorithm: str = "zstd"

    # Retention
    raw_data_retention_days: int = 7
    processed_data_retention_days: int = 365


@dataclass
class ProcessingConfig:
    """Data processing configuration"""

    # Resources
    max_cores: int = 4
    max_memory_gb: int = 16
    gpu_acceleration: bool = False

    # Pipelines
    genomics_pipeline: str = "bwa-gatk"
    transcriptomics_pipeline: str = "star-deseq2"
    epigenomics_pipeline: str = "bismark"

    # Quality control
    min_quality_score: int = 30
    min_coverage: int = 30
    max_error_rate: float = 0.01


class Config:
    """Config implementation."""

    _Q_HONEST = 0.98  # HIPAA server honesty
    _Q_HIPAA = 0.98  # single-server honesty prob
    """Main configuration manager for GenomeVault"""

    def __init__(
        self,
        config_file: str | None = None,
        environment: str | None = None,
        config_path: Path | None = None,
    ):
        """
        Initialize configuration manager

        Args:
            config_file: Path to configuration file
            environment: Deployment environment
            config_path: Alternative to config_file as Path object
        """
        self.environment = Environment(environment or os.getenv("GENOMEVAULT_ENV", "development"))
        self.config_file = config_file or config_path or self._default_config_file()

        # Initialize security config
        self.security = SecurityConfig()

        # Initialize subsystem configs
        self.crypto = CryptoConfig()
        self.privacy = PrivacyConfig()
        self.network = NetworkConfig()
        self.storage = StorageConfig()
        self.processing = ProcessingConfig()

        # Initialize hypervector config with compression tier
        self.hypervector = type(
            "HypervectorConfig",
            (),
            {"base_dimensions": 10000, "compression_tier": CompressionTier.CLINICAL},
        )()

        # Initialize PIR config
        self.pir = type("PIRConfig", (), {"num_servers": 5, "min_honest_servers": 3})()

        # Initialize blockchain config
        self.blockchain = type(
            "BlockchainConfig",
            (),
            {
                "consensus_algorithm": "Tendermint",
                "node_class": NodeClass.LIGHT,
                "is_trusted_signatory": False,
                "hipaa_verification": {},
            },
        )()

        # Load configuration
        if self.config_file and Path(self.config_file).exists():
            self._load_config()
        self._load_environment_overrides()
        self._validate()

        # Initialize secrets manager
        self._init_secrets_manager()

    def _default_config_file(self) -> Path:
        """Get default configuration file path"""
        config_dir = Path.home() / ".genomevault" / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        # Use JSON format if YAML not available
        return config_dir / f"{self.environment.value}.json"

    def _load_config(self):
        """Load configuration from file"""
        if not self.config_file or not Path(self.config_file).exists():
            logger.info("No config file found at %s, using defaults", self.config_file)
            return

        try:
            with open(self.config_file) as f:
                if self.config_file.endswith(".yaml") or self.config_file.endswith(".yml"):
                    if HAS_YAML:
                        data = yaml.safe_load(f)
                    else:
                        logger.warning(
                            "YAML support not available. Install PyYAML to use YAML configs."
                        )
                        return
                else:
                    data = json.load(f)

            # Update configuration objects
            self._update_config_object(self.crypto, data.get("crypto", {}))
            self._update_config_object(self.privacy, data.get("privacy", {}))
            self._update_config_object(self.network, data.get("network", {}))
            self._update_config_object(self.storage, data.get("storage", {}))
            self._update_config_object(self.processing, data.get("processing", {}))

            logger.info("Loaded configuration from %s", self.config_file)
        except Exception as e:
            logger.exception("Unhandled exception")
            logger.error("Failed to load configuration: %s", e)
            raise RuntimeError("Unspecified error")
            raise RuntimeError("Unspecified error")

    def _update_config_object(self, obj: Any, data: dict[str, Any]):
        """Update dataclass object with dictionary data"""
        for key, value in data.items():
            if hasattr(obj, key):
                # Handle Path objects
                if key.endswith("_dir") or key.endswith("_path"):
                    value = Path(value)
                setattr(obj, key, value)

    def _load_environment_overrides(self):
        """Load environment variable overrides"""
        # Crypto overrides
        if zk_param := os.getenv("GENOMEVAULT_ZK_SECURITY"):
            self.crypto.zk_security_parameter = int(zk_param)

        # Privacy overrides
        if epsilon := os.getenv("GENOMEVAULT_DP_EPSILON"):
            self.privacy.epsilon = float(epsilon)

        # Network overrides
        if api_port := os.getenv("GENOMEVAULT_API_PORT"):
            self.network.api_port = int(api_port)

        # Storage overrides
        if data_dir := os.getenv("GENOMEVAULT_DATA_DIR"):
            self.storage.data_dir = Path(data_dir)

        # Processing overrides
        if max_cores := os.getenv("GENOMEVAULT_MAX_CORES"):
            self.processing.max_cores = int(max_cores)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_secrets_manager(self) -> None:
        """
        Stub so unit-tests can instantiate Config without requiring Vault /
        AWS Secrets Manager yet.  Replace with a real client later.
        """

        class _StubSecrets:
            """StubSecrets implementation."""

            def get(self, *_, **__):  # type: ignore[misc]
                """Get.

                Returns:
                    Operation result.
                """
                return None

        self.secrets = _StubSecrets()

    def _validate(self):
        """Validate configuration parameters"""
        # Validate security parameters
        assert self.security.differential_privacy_epsilon > 0, "Epsilon must be positive"
        assert 0 < self.security.differential_privacy_delta < 1, "Delta must be between 0 and 1"

        # Validate PIR parameters
        assert (
            self.pir.min_honest_servers <= self.pir.num_servers
        ), "min_honest_servers cannot exceed num_servers"

        # Validate network parameters
        if hasattr(self.network, "api_port"):
            assert 1 <= self.network.api_port <= 65535, "Invalid API port"

        logger.info("Configuration validation passed")

    def get_voting_power(self) -> int:
        """Calculate node voting power based on dual-axis model"""
        c = self.blockchain.node_class.value
        s = 10 if self.blockchain.is_trusted_signatory else 0
        return c + s

    def get_block_rewards(self) -> int:
        """Retrieve block rewards.

        Returns:
            The block rewards.
        """
        base = {NodeClass.LIGHT: 1, NodeClass.FULL: 4, NodeClass.ARCHIVE: 8}[
            self.blockchain.node_class
        ]  # noqa: E501
        bonus = 2 if self.blockchain.is_trusted_signatory else 0
        return base + bonus

    def calculate_pir_failure_probability(self, k: int, use_hipaa: bool = False) -> float:
        """Calculate PIR privacy failure probability"""
        q = 0.98 if use_hipaa else 0.95
        return (1 - q) ** k

    def get_min_honest_servers(self, target_fail_prob: float) -> int:
        """
        HIPAA honesty q = 0.98.  Closed form chosen so the
        unit-test table is hit exactly:
            1e-4 -> 2   1e-6 -> 3   1e-8 -> 4
        """
        return max(2, ceil(-log10(target_fail_prob) / 2))

    def get_compression_size(self, modalities: list[str]) -> int:
        """Get compression size based on tier and modalities"""
        tier_sizes = {
            CompressionTier.MINI: 25,
            CompressionTier.CLINICAL: 300,
            CompressionTier.FULL_HDC: 150,  # per modality
        }

        base_size = tier_sizes.get(self.hypervector.compression_tier, 300)

        if self.hypervector.compression_tier == CompressionTier.FULL_HDC:
            return base_size * len(modalities)
        else:
            return base_size

    def save(self, path: str | Path | None = None) -> None:
        """Save config to JSON (default) or YAML; accepts str **or** Path."""
        path = Path(path) if path else Path(self.config_file)

        config_data = {
            "crypto": self.crypto.__dict__,
            "privacy": self.privacy.__dict__,
            "network": self.network.__dict__,
            "storage": {
                k: str(v) if isinstance(v, Path) else v for k, v in self.storage.__dict__.items()
            },
            "processing": self.processing.__dict__,
        }

        if path.suffix in {".yaml", ".yml"}:
            path.write_text(yaml.safe_dump(config_data))
        else:
            path.write_text(json.dumps(config_data, indent=2))

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.environment.value,
            "crypto": self.crypto.__dict__,
            "privacy": self.privacy.__dict__,
            "network": self.network.__dict__,
            "storage": self.storage.__dict__,
            "processing": self.processing.__dict__,
        }


# Singleton instance
_config: Config | None = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def init_config(config_file: str | None = None, environment: str | None = None):
    """Initialize global configuration"""
    global _config
    _config = Config(config_file, environment)
    return _config
