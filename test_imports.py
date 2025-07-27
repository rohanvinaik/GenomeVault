#!/usr/bin/env python3
import logging

"""Test GenomeVault imports"""

print("Testing GenomeVault imports...")

try:
    from genomevault.core.config import Config

    print("✓ Config imported")
except ImportError as e:
    print(f"✗ Config import failed: {e}")

try:
    from genomevault.local_processing import SequencingProcessor

    print("✓ SequencingProcessor imported")
except ImportError as e:
    print(f"✗ SequencingProcessor import failed: {e}")

try:
    from genomevault.local_processing.phenotypes import PhenotypeProcessor

    print("✓ PhenotypeProcessor imported")
except ImportError as e:
    print(f"✗ PhenotypeProcessor import failed: {e}")

try:
    from genomevault.utils.logging import get_logger

    print("✓ get_logger imported")
except ImportError as e:
    print(f"✗ get_logger import failed: {e}")

try:
    from genomevault.utils.config import get_config

    print("✓ get_config imported")
except ImportError as e:
    print(f"✗ get_config import failed: {e}")

print("\nImport test complete!")
