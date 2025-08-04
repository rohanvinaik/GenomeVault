"""
Core configuration management for GenomeVault
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Central configuration for GenomeVault system"""

    # Node Configuration
    node_id: str = Field("node_001", env="NODE_ID")
    node_type: str = Field("light", env="NODE_TYPE")
    signatory_status: bool = Field(False, env="SIGNATORY_STATUS")
    NODE_CLASS_WEIGHT: int = Field(1, env="NODE_CLASS_WEIGHT")

    # Network Configuration
    pir_servers: str = Field(
        "localhost:9001,localhost:9002,localhost:9003", env="PIR_SERVERS"
    )
    blockchain_rpc: str = Field("http://localhost:8545", env="BLOCKCHAIN_RPC")
    ipfs_api: str = Field("http://localhost:5001", env="IPFS_API")

    # Security Configuration
    redis_password: str = Field("genomevault", env="REDIS_PASSWORD")
    jwt_secret: str = Field("test_secret", env="JWT_SECRET")
    encryption_key: str = Field("test_key", env="ENCRYPTION_KEY")

    # HIPAA Configuration
    npi_number: str | None = Field(None, env="NPI_NUMBER")
    baa_hash: str | None = Field(None, env="BAA_HASH")
    risk_analysis_hash: str | None = Field(None, env="RISK_ANALYSIS_HASH")
    hsm_serial: str | None = Field(None, env="HSM_SERIAL")

    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_workers: int = Field(4, env="API_WORKERS")

    # Processing Configuration
    max_genome_size: str = Field("3GB", env="MAX_GENOME_SIZE")
    compression_tier: str = Field("clinical", env="COMPRESSION_TIER")
    enable_gpu: bool = Field(False, env="ENABLE_GPU")

    # Hypervector dimensions
    hypervector_dim: int = 10000
    mini_tier_size: int = 25 * 1024  # 25KB
    clinical_tier_size: int = 300 * 1024  # 300KB
    full_tier_size: int = 200 * 1024  # 200KB per modality

    # ZK Proof Configuration
    proof_size: int = 384  # bytes
    verification_time: int = 25  # milliseconds
    post_quantum_security_level: int = 256  # bits

    # PIR Configuration
    pir_privacy_threshold: float = 0.98
    pir_latency_target: int = 350  # milliseconds

    # Clinical Features
    enable_diabetes_pilot: bool = Field(False, env="ENABLE_DIABETES_PILOT")
    enable_pharmacogenomics: bool = Field(False, env="ENABLE_PHARMACOGENOMICS")
    enable_trial_matching: bool = Field(False, env="ENABLE_TRIAL_MATCHING")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @field_validator("node_type")
    def validate_node_type(cls, v):
        if v not in ["light", "full", "archive"]:
            raise ValueError("node_type must be one of: light, full, archive")
        return v

    @field_validator("compression_tier")
    def validate_compression_tier(cls, v):
        if v not in ["mini", "clinical", "full"]:
            raise ValueError("compression_tier must be one of: mini, clinical, full")
        return v

    @property
    def total_voting_power(self) -> int:
        """Calculate total voting power: w = c + s"""
        signatory_weight = 10 if self.signatory_status else 0
        return self.NODE_CLASS_WEIGHT + signatory_weight

    @property
    def credits_per_block(self) -> int:
        """Calculate credits earned per block"""
        base_credits = self.NODE_CLASS_WEIGHT
        signatory_bonus = 2 if self.signatory_status else 0
        return base_credits + signatory_bonus

    @property
    def pir_server_list(self) -> list[str]:
        """Parse PIR servers into a list"""
        return [s.strip() for s in self.pir_servers.split(",")]

    def is_hipaa_compliant(self) -> bool:
        """Check if all HIPAA requirements are met"""
        return all(
            [self.npi_number, self.baa_hash, self.risk_analysis_hash, self.hsm_serial]
        )


@lru_cache
def get_config() -> Config:
    """Get cached configuration instance"""
    return Config()
