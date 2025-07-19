"""
GenomeVault Utilities Package

Core utilities for configuration, logging, and encryption.
"""

from .config import (
    Config,
    CryptoConfig,
    Environment,
    NetworkConfig,
    PrivacyConfig,
    ProcessingConfig,
    SecurityLevel,
    StorageConfig,
    get_config,
    init_config,
)
from .encryption import (
    AESGCMCipher,
    ChaCha20Poly1305,
    EncryptionManager,
    KeyDerivation,
    RSAEncryption,
    SecureRandom,
    ThresholdCrypto,
    ThresholdShare,
    generate_secure_key,
    secure_hash,
)
from .logging import (
    GenomeVaultLogger,
    LogEvent,
    PrivacyLevel,
    configure_logging,
    get_logger,
    log_genomic_operation,
    log_operation,
)

__all__ = [
    # Config
    'Config',
    'get_config',
    'init_config',
    'Environment',
    'SecurityLevel',
    'CryptoConfig',
    'PrivacyConfig',
    'NetworkConfig',
    'StorageConfig',
    'ProcessingConfig',
    
    # Logging
    'get_logger',
    'configure_logging',
    'log_operation',
    'log_genomic_operation',
    'LogEvent',
    'PrivacyLevel',
    'GenomeVaultLogger',
    
    # Encryption
    'AESGCMCipher',
    'ChaCha20Poly1305',
    'RSAEncryption',
    'ThresholdCrypto',
    'ThresholdShare',
    'KeyDerivation',
    'SecureRandom',
    'EncryptionManager',
    'generate_secure_key',
    'secure_hash'
]

# Version info
__version__ = '3.0.0'
__author__ = 'GenomeVault Team'
