"""
GenomeVault 3.0

A revolutionary privacy-preserving genomic data platform that enables secure analysis 
and research while maintaining complete individual data sovereignty.
"""

# Version information
__version__ = '3.0.0'
__author__ = 'GenomeVault Team'
__license__ = 'Apache License 2.0'

# Avoid circular imports by not importing submodules at package level
# Users should import directly from submodules:
# from genomevault.local_processing import SequencingProcessor
# from genomevault.core.config import get_config

__all__ = [
    '__version__',
    '__author__',
    '__license__'
]

# Package metadata
__description__ = """
GenomeVault 3.0 is a comprehensive platform for privacy-preserving genomic data analysis.
It combines hyperdimensional computing, zero-knowledge cryptography, and federated AI to
enable population-scale genomic research while ensuring individuals maintain complete
sovereignty over their biological information.
"""

__features__ = [
    "Complete privacy through mathematical guarantees",
    "Multi-omics data integration (genomics, transcriptomics, epigenomics, proteomics)",
    "Zero-knowledge proofs for verifiable computations",
    "Information-theoretic private information retrieval",
    "Federated learning for distributed AI",
    "Blockchain governance with dual-axis node model",
    "Post-quantum cryptographic security",
    "Continuous knowledge integration"
]

# Quick start message
def _print_welcome():
    """Print welcome message when package is imported interactively"""
    import sys
    if hasattr(sys, 'ps1'):  # Interactive mode
        print(f"GenomeVault {__version__} - Privacy-Preserving Genomic Intelligence")
        print("Documentation: https://docs.genomevault.io")
        print("Quick start: from genomevault.core.config import get_config")

# Don't run welcome in non-interactive mode to avoid test issues
import sys
if hasattr(sys, 'ps1'):
    _print_welcome()
