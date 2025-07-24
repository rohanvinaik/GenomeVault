"""
GenomeVault Configuration Management

This module provides centralized configuration management for all GenomeVault components,
including environment-specific settings, secrets management, and runtime configuration.
"""

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

# These are REQUIRED - not optional!
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environment types"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


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
    npi_number: Optional[str] = None


@dataclass
class StorageConfig:
    """Storage configuration parameters"""

    # Local storage
    data_dir: Path = Path.home() / ".genomevault"
    temp_dir: Path = Path("/tmp/genomevault")

    # Database
    db_type: str = "sqlite"  # sqlite, postgresql, mongodb
    db_path: Optional[str] = None

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
    """Main configuration manager for GenomeVault"""

    def __init__(self, config_file: Optional[str] = None, environment: Optional[str] = None):
        """
        Initialize configuration manager

        Args:
            config_file: Path to configuration file
            environment: Deployment environment
        """
        self.environment = Environment(environment or os.getenv("GENOMEVAULT_ENV", "development"))
        self.config_file = config_file or self._default_config_file()

        # Initialize subsystem configs
        self.crypto = CryptoConfig()
        self.privacy = PrivacyConfig()
        self.network = NetworkConfig()
        self.storage = StorageConfig()
        self.processing = ProcessingConfig()

        # Load configuration
        self._load_config()
        self._load_environment_overrides()
        self._validate_config()

        # Initialize secrets manager
        self._init_secrets_manager()

    def _default_config_file(self) -> Path:
        """Get default configuration file path"""
        config_dir = Path.home() / ".genomevault" / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        # Use JSON format if YAML not available
        extension = ".yaml" if HAS_YAML else ".json"
        return config_dir / "{self.environment.value}{extension}"

    def _load_config(self):
        """Load configuration from file"""
        if not self.config_file or not Path(self.config_file).exists():
            logger.info("No config file found at {self.config_file}, using defaults")
            return

        try:
            with open(self.config_file, "r") as f:
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

            logger.info("Loaded configuration from {self.config_file}")
        except Exception as e:
            logger.error("Failed to load configuration: {e}")
            raise

    def _update_config_object(self, obj: Any, data: Dict[str, Any]):
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

    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate crypto parameters
        assert self.crypto.aes_key_size in [128, 192, 256], "Invalid AES key size"
        assert self.crypto.rsa_key_size >= 2048, "RSA key size too small"
        assert self.crypto.hypervector_dimensions >= 1000, "Hypervector dimensions too small"

        # Validate privacy parameters
        assert 0 < self.privacy.epsilon <= 10, "Invalid epsilon value"
        assert 0 < self.privacy.delta < 1, "Invalid delta value"

        # Validate network parameters
        assert 1 <= self.network.api_port <= 65535, "Invalid API port"
        assert (
            len(self.network.pir_servers) >= self.crypto.pir_threshold
            if self.network.pir_servers
            else True
        )

        # Validate storage parameters
        assert self.storage.compression_tier in [
            "mini",
            "clinical",
            "full",
        ], "Invalid compression tier"

        logger.info("Configuration validation passed")

    def _init_secrets_manager(self):
        """Initialize secrets management"""
        self._master_key = self._derive_master_key()
        self._cipher = Fernet(self._master_key)

    def _derive_master_key(self) -> bytes:
        """Derive master key from environment or hardware"""
        # In production, this should use HSM or secure key management
        password = os.getenv("GENOMEVAULT_MASTER_PASSWORD", "development-password").encode()
        salt = b"genomevault-salt"  # In production, use random salt

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password))

    def encrypt_secret(self, secret: str) -> str:
        """Encrypt a secret value"""
        return self._cipher.encrypt(secret.encode()).decode()

    def decrypt_secret(self, encrypted: str) -> str:
        """Decrypt a secret value"""
        return self._cipher.decrypt(encrypted.encode()).decode()

    def get_compression_settings(self) -> Dict[str, Any]:
        """Get compression settings based on tier"""
        tiers = {
            "mini": {
                "features": 5000,
                "size_kb": 25,
                "description": "Most-studied SNPs only",
            },
            "clinical": {
                "features": 120000,
                "size_kb": 300,
                "description": "ACMG + PharmGKB variants",
            },
            "full": {
                "features": "all",
                "size_kb": 200,  # per modality
                "description": "Full HDC vectors",
            },
        }
        return tiers.get(self.storage.compression_tier, tiers["clinical"])

    def get_node_voting_power(self) -> int:
        """Calculate node voting power based on dual-axis model"""
        class_weights = {"light": 1, "full": 4, "archive": 8}
        c = class_weights.get(self.network.node_class, 1)
        s = 10 if self.network.signatory_status else 0
        return c + s

    def get_block_rewards(self) -> int:
        """Calculate block reward credits"""
        class_weights = {"light": 1, "full": 4, "archive": 8}
        c = class_weights.get(self.network.node_class, 1)
        ts_bonus = 2 if self.network.signatory_status else 0
        return c + ts_bonus

    def get_pir_failure_probability(self) -> float:
        """Calculate PIR privacy failure probability"""
        if not self.network.pir_servers:
            return 1.0

        # P_fail(k,q) = (1-q)^k
        q = 0.98 if self.network.hipaa_verified else 0.95
        k = len([s for s in self.network.pir_servers if s.get("trusted", False)])
        return (1 - q) ** k if k > 0 else 1.0

    def save(self, path: Optional[str] = None):
        """Save current configuration to file"""
        path = path or self.config_file

        config_data = {
            "crypto": self.crypto.__dict__,
            "privacy": self.privacy.__dict__,
            "network": self.network.__dict__,
            "storage": {
                k: str(v) if isinstance(v, Path) else v for k, v in self.storage.__dict__.items()
            },
            "processing": self.processing.__dict__,
        }

        with open(path, "w") as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                if HAS_YAML:
                    yaml.dump(config_data, f, default_flow_style=False)
                else:
                    # Fall back to JSON if YAML not available
                    json.dump(config_data, f, indent=2)
            else:
                json.dump(config_data, f, indent=2)

        logger.info("Saved configuration to {path}")

    def to_dict(self) -> Dict[str, Any]:
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
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def init_config(config_file: Optional[str] = None, environment: Optional[str] = None):
    """Initialize global configuration"""
    global _config
    _config = Config(config_file, environment)
    return _config
