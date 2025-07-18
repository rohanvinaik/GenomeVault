"""
GenomeVault Core Module

This module contains the core configuration and shared components
used throughout the GenomeVault system.
"""

__version__ = "3.0.0"
__author__ = "GenomeVault Team"

from .config import Config
from .constants import *
from .exceptions import *

__all__ = ["Config", "GenomeVaultError", "ValidationError", "PrivacyError"]
