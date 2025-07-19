"""GenomeVault - Privacy-preserving genomic data management platform."""

__version__ = "0.1.0"
__author__ = "GenomeVault Team"
__email__ = "team@genomevault.org"

# Import key components for easier access
from genomevault.core.config import Config
from genomevault.core.constants import *
from genomevault.core.exceptions import *

__all__ = ["Config"]
