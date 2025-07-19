"""
GenomeVault Utilities Package

Core utilities for configuration, logging, and encryption.
"""

# Import only what's actually needed and used
from .config import Config, get_config

from .encryption import (
    AESGCMCipher,
    generate_secure_key,
    secure_hash,
)

from .logging import (
    get_logger,
    audit_logger,
    performance_logger,
    security_logger,
)

# Don't import modules that have heavy dependencies unless needed
# from . import backup  # Has boto3 dependency
# from . import monitoring  # May have other dependencies

__all__ = [
    # Config
    "Config",
    "get_config",
    # Logging
    "get_logger",
    "audit_logger",
    "performance_logger",
    "security_logger",
    # Encryption
    "AESGCMCipher",
    "generate_secure_key",
    "secure_hash",
]

# Version info
__version__ = "3.0.0"
__author__ = "GenomeVault Team"
