"""
GenomeVault Core Configuration
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic import BaseSettings, Field, validator


class GenomeVaultConfig(BaseSettings):
    """Core configuration for GenomeVault."""

    # API Configuration
    api_host: str = Field(default="127.0.0.1", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_prefix: str = Field(default="/api/v1", description="API prefix")

    # Security
    secret_key: str = Field(default="", description="Secret key for JWT")
    allowed_hosts: str = Field(default="*", description="Allowed hosts")
    cors_origins: str = Field(default="*", description="CORS origins")

    # Database
    database_url: Optional[str] = Field(default=None, description="Database URL")
    redis_url: Optional[str] = Field(default=None, description="Redis URL")

    # Blockchain
    node_type: str = Field(default="light", description="Node type")
    blockchain_network: str = Field(default="testnet", description="Blockchain network")
    consensus_algorithm: str = Field(
        default="tendermint", description="Consensus algorithm"
    )

    # PIR Configuration
    pir_server_count: int = Field(default=5, description="Number of PIR servers")
    pir_privacy_threshold: float = Field(
        default=0.98, description="PIR privacy threshold"
    )

    # ZK Proofs
    zk_circuit_path: str = Field(default="./circuits", description="ZK circuit path")
    proof_timeout: int = Field(default=300, description="Proof timeout in seconds")

    # Hypervector
    hypervector_dimensions: int = Field(
        default=10000, description="Hypervector dimensions"
    )
    compression_tier: str = Field(default="clinical", description="Compression tier")

    # Local Processing
    processing_threads: int = Field(default=4, description="Processing threads")
    temp_dir: str = Field(default="/tmp/genomevault", description="Temporary directory")

    # Monitoring
    enable_monitoring: bool = Field(default=True, description="Enable monitoring")
    log_level: str = Field(default="INFO", description="Log level")

    # HIPAA
    hipaa_compliance: bool = Field(default=True, description="HIPAA compliance mode")
    encryption_at_rest: bool = Field(default=True, description="Encryption at rest")

    @validator("node_type")
    @classmethod
    def validate_node_type(cls, v):
        """Validate node type."""
        allowed = ["light", "full", "archive"]
        if v not in allowed:
            raise ValueError(f"Node type must be one of {allowed}")
        return v

    @validator("compression_tier")
    @classmethod
    def validate_compression_tier(cls, v):
        """Validate compression tier."""
        allowed = ["mini", "clinical", "full"]
        if v not in allowed:
            raise ValueError(f"Compression tier must be one of {allowed}")
        return v

    class Config:
        env_prefix = "GENOMEVAULT_"
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_config() -> GenomeVaultConfig:
    """Get cached configuration instance."""
    return GenomeVaultConfig()


# Global config instance
config = get_config()

# Environment-specific settings
ENVIRONMENTS = {
    "development": {
        "debug": True,
        "log_level": "DEBUG",
    },
    "production": {
        "debug": False,
        "log_level": "INFO",
    },
    "testing": {
        "debug": True,
        "log_level": "DEBUG",
    },
}


def get_environment_config(env: str = None):
    """Get environment-specific configuration."""
    if env is None:
        env = os.getenv("GENOMEVAULT_ENVIRONMENT", "development")
    return ENVIRONMENTS.get(env, ENVIRONMENTS["development"])
