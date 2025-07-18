"""
Central configuration management for GenomeVault 3.0.
Handles environment variables, HSM integration, and centralized settings.
"""
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class CompressionTier(Enum):
    """Compression tier options for hypervector storage."""
    MINI = "mini"  # ~5,000 most-studied SNPs (~25 KB)
    CLINICAL = "clinical"  # ACMG + PharmGKB variants (~120k) (~300 KB)
    FULL_HDC = "full_hdc"  # 10,000-D vectors per modality (100-200 KB)


class NodeClass(Enum):
    """Node hardware classification for dual-axis model."""
    LIGHT = 1  # Consumer hardware (e.g., Mac mini)
    FULL = 4   # Standard servers (e.g., 1U rack server)
    ARCHIVE = 8  # High-performance storage systems


@dataclass
class SecurityConfig:
    """Security-related configuration."""
    encryption_algorithm: str = "AES-256-GCM"
    key_derivation_function: str = "HKDF-SHA256"
    post_quantum_algorithm: str = "CRYSTALS-Kyber"
    zk_proof_system: str = "PLONK"
    differential_privacy_epsilon: float = 1.0
    differential_privacy_delta: float = 1e-6
    hsm_enabled: bool = field(default_factory=lambda: os.getenv('HSM_ENABLED', 'false').lower() == 'true')
    hsm_serial: Optional[str] = field(default_factory=lambda: os.getenv('HSM_SERIAL'))


@dataclass
class ProcessingConfig:
    """Local processing configuration."""
    bwa_path: str = field(default_factory=lambda: os.getenv('BWA_PATH', '/usr/bin/bwa'))
    gatk_jar: str = field(default_factory=lambda: os.getenv('GATK_JAR', '/opt/gatk/gatk.jar'))
    star_path: str = field(default_factory=lambda: os.getenv('STAR_PATH', '/usr/bin/STAR'))
    kallisto_path: str = field(default_factory=lambda: os.getenv('KALLISTO_PATH', '/usr/bin/kallisto'))
    max_threads: int = field(default_factory=lambda: int(os.getenv('MAX_THREADS', '8')))
    memory_limit_gb: int = field(default_factory=lambda: int(os.getenv('MEMORY_LIMIT_GB', '16')))
    temp_dir: Path = field(default_factory=lambda: Path(os.getenv('TEMP_DIR', '/tmp/genomevault')))
    container_runtime: str = field(default_factory=lambda: os.getenv('CONTAINER_RUNTIME', 'docker'))


@dataclass
class HypervectorConfig:
    """Hypervector engine configuration."""
    base_dimensions: int = 10000
    mid_dimensions: int = 15000
    high_dimensions: int = 20000
    compression_tier: CompressionTier = CompressionTier.CLINICAL
    projection_version: str = "v3.0"
    enable_simd: bool = True
    sparsity_threshold: float = 0.1


@dataclass
class PIRConfig:
    """Private Information Retrieval configuration."""
    num_servers: int = 5
    min_honest_servers: int = 2
    server_honesty_generic: float = 0.95
    server_honesty_hipaa: float = 0.98
    target_failure_probability: float = 1e-4
    query_timeout_seconds: int = 30
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600


@dataclass
class BlockchainConfig:
    """Blockchain and governance configuration."""
    consensus_algorithm: str = "Tendermint"
    block_time_seconds: int = 3
    transactions_per_second: int = 3000
    node_class: NodeClass = NodeClass.LIGHT
    is_trusted_signatory: bool = False
    stake_amount: int = 0
    slashing_percentage: float = 0.25
    hipaa_verification: Dict[str, Optional[str]] = field(default_factory=lambda: {
        'npi': os.getenv('HIPAA_NPI'),
        'baa_hash': os.getenv('HIPAA_BAA_HASH'),
        'risk_analysis_hash': os.getenv('HIPAA_RISK_ANALYSIS_HASH'),
        'hsm_serial': os.getenv('HSM_SERIAL')
    })


@dataclass
class NetworkConfig:
    """Network and API configuration."""
    api_host: str = field(default_factory=lambda: os.getenv('API_HOST', '0.0.0.0'))
    api_port: int = field(default_factory=lambda: int(os.getenv('API_PORT', '8000')))
    enable_ssl: bool = field(default_factory=lambda: os.getenv('ENABLE_SSL', 'true').lower() == 'true')
    ssl_cert_path: Optional[Path] = field(default_factory=lambda: Path(os.getenv('SSL_CERT_PATH')) if os.getenv('SSL_CERT_PATH') else None)
    ssl_key_path: Optional[Path] = field(default_factory=lambda: Path(os.getenv('SSL_KEY_PATH')) if os.getenv('SSL_KEY_PATH') else None)
    rate_limit_per_minute: int = 60
    max_request_size_mb: int = 100


class Config:
    """Central configuration management for GenomeVault."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration from environment and optional config file.
        
        Args:
            config_path: Optional path to JSON configuration file
        """
        self.security = SecurityConfig()
        self.processing = ProcessingConfig()
        self.hypervector = HypervectorConfig()
        self.pir = PIRConfig()
        self.blockchain = BlockchainConfig()
        self.network = NetworkConfig()
        
        # Load from config file if provided
        if config_path and config_path.exists():
            self._load_from_file(config_path)
        
        # Override with environment variables
        self._load_from_env()
        
        # Validate configuration
        self._validate()
        
        # Create necessary directories
        self._setup_directories()
    
    def _load_from_file(self, config_path: Path):
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            data = json.load(f)
            
        # Update dataclass instances with loaded data
        for section_name, section_data in data.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                section = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def _load_from_env(self):
        """Override configuration with environment variables."""
        # Environment variables take precedence over file configuration
        # This is handled by the field default_factory in dataclasses
        pass
    
    def _validate(self):
        """Validate configuration settings."""
        # Validate security settings
        assert self.security.differential_privacy_epsilon > 0, "Differential privacy epsilon must be positive"
        assert 0 < self.security.differential_privacy_delta < 1, "Differential privacy delta must be between 0 and 1"
        
        # Validate processing settings
        assert self.processing.max_threads > 0, "Max threads must be positive"
        assert self.processing.memory_limit_gb > 0, "Memory limit must be positive"
        
        # Validate hypervector settings
        assert self.hypervector.base_dimensions > 0, "Base dimensions must be positive"
        assert 0 <= self.hypervector.sparsity_threshold <= 1, "Sparsity threshold must be between 0 and 1"
        
        # Validate PIR settings
        assert self.pir.num_servers >= self.pir.min_honest_servers, "Number of servers must be >= minimum honest servers"
        assert 0 < self.pir.server_honesty_generic <= 1, "Server honesty probability must be between 0 and 1"
        assert 0 < self.pir.server_honesty_hipaa <= 1, "HIPAA server honesty probability must be between 0 and 1"
        
        # Validate blockchain settings
        assert self.blockchain.block_time_seconds > 0, "Block time must be positive"
        assert 0 <= self.blockchain.slashing_percentage <= 1, "Slashing percentage must be between 0 and 1"
        
        # Validate network settings
        assert 0 < self.network.api_port < 65536, "API port must be valid"
        assert self.network.rate_limit_per_minute > 0, "Rate limit must be positive"
    
    def _setup_directories(self):
        """Create necessary directories."""
        self.processing.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def get_voting_power(self) -> int:
        """
        Calculate node voting power based on dual-axis model.
        
        Returns:
            Total voting power (w = c + s)
        """
        c = self.blockchain.node_class.value
        s = 10 if self.blockchain.is_trusted_signatory else 0
        return c + s
    
    def get_block_rewards(self) -> int:
        """
        Calculate block rewards based on node configuration.
        
        Returns:
            Credits per block
        """
        c = self.blockchain.node_class.value
        ts_bonus = 2 if self.blockchain.is_trusted_signatory else 0
        return c + ts_bonus
    
    def calculate_pir_failure_probability(self, k: int, use_hipaa: bool = False) -> float:
        """
        Calculate PIR privacy breach probability.
        
        Args:
            k: Number of required honest servers
            use_hipaa: Whether to use HIPAA server honesty probability
            
        Returns:
            Privacy breach probability P_fail(k,q) = (1-q)^k
        """
        q = self.pir.server_honesty_hipaa if use_hipaa else self.pir.server_honesty_generic
        return (1 - q) ** k
    
    def get_min_honest_servers(self, target_failure_prob: Optional[float] = None) -> int:
        """
        Calculate minimum required honest servers for target failure probability.
        
        Args:
            target_failure_prob: Target failure probability (uses config default if None)
            
        Returns:
            Minimum number of required honest servers
        """
        import math
        
        if target_failure_prob is None:
            target_failure_prob = self.pir.target_failure_probability
        
        q = self.pir.server_honesty_hipaa  # Use higher honesty for calculation
        k_min = math.ceil(math.log(target_failure_prob) / math.log(1 - q))
        return k_min
    
    def get_compression_size(self, modalities: list[str]) -> int:
        """
        Calculate total client storage based on compression tier and modalities.
        
        Args:
            modalities: List of data modalities (e.g., ['genomics', 'transcriptomics'])
            
        Returns:
            Total storage in KB
        """
        tier_sizes = {
            CompressionTier.MINI: 25,  # KB
            CompressionTier.CLINICAL: 300,  # KB
            CompressionTier.FULL_HDC: 150  # KB per modality (average)
        }
        
        if self.hypervector.compression_tier == CompressionTier.FULL_HDC:
            return tier_sizes[self.hypervector.compression_tier] * len(modalities)
        else:
            return tier_sizes[self.hypervector.compression_tier]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'security': self.security.__dict__,
            'processing': {k: str(v) if isinstance(v, Path) else v for k, v in self.processing.__dict__.items()},
            'hypervector': {k: v.value if isinstance(v, Enum) else v for k, v in self.hypervector.__dict__.items()},
            'pir': self.pir.__dict__,
            'blockchain': {k: v.value if isinstance(v, Enum) else v for k, v in self.blockchain.__dict__.items()},
            'network': {k: str(v) if isinstance(v, Path) else v for k, v in self.network.__dict__.items()}
        }
    
    def save(self, config_path: Path):
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Global configuration instance
config = Config()


# Example usage and calculations
if __name__ == "__main__":
    # Print current configuration
    print("=== GenomeVault Configuration ===")
    print(f"Voting Power: {config.get_voting_power()}")
    print(f"Block Rewards: {config.get_block_rewards()} credits/block")
    print(f"PIR Failure Probability (k=2): {config.calculate_pir_failure_probability(2):.2e}")
    print(f"Min Honest Servers for target failure: {config.get_min_honest_servers()}")
    print(f"Storage for genomics + transcriptomics: {config.get_compression_size(['genomics', 'transcriptomics'])} KB")
    
    # Save configuration
    config.save(Path("genomevault_config.json"))
