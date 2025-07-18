"""
GenomeVault Utilities Package

Core utilities for configuration, logging, and encryption.
"""

from .config import (
    Config, 
    get_config, 
    init_config,
    Environment,
    SecurityLevel,
    CryptoConfig,
    PrivacyConfig,
    NetworkConfig,
    StorageConfig,
    ProcessingConfig
)

from .logging import (
    get_logger,
    configure_logging,
    log_operation,
    log_genomic_operation,
    LogEvent,
    PrivacyLevel,
    GenomeVaultLogger
)

from .encryption import (
    AESGCMCipher,
    ChaCha20Poly1305,
    RSAEncryption,
    ThresholdCrypto,
    ThresholdShare,
    KeyDerivation,
    SecureRandom,
    EncryptionManager,
    generate_secure_key,
    secure_hash
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
