"""
Path configuration for GenomeVault.

All paths should be configured here rather than hardcoded.
"""

import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
KEYS_DIR = PROJECT_ROOT / "keys"
CIRCUITS_DIR = PROJECT_ROOT / "circuits"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ENCRYPTED_DATA_DIR = DATA_DIR / "encrypted"

# User directories (configurable)
USER_HOME = Path.home()
USER_DESKTOP = USER_HOME / "Desktop"
USER_DOCUMENTS = USER_HOME / "Documents"
USER_DOWNLOADS = USER_HOME / "Downloads"

# Environment-based paths
GENOMEVAULT_HOME = Path(os.environ.get('GENOMEVAULT_HOME', str(PROJECT_ROOT)))
GENOMEVAULT_DATA = Path(os.environ.get('GENOMEVAULT_DATA', str(DATA_DIR)))
GENOMEVAULT_LOGS = Path(os.environ.get('GENOMEVAULT_LOGS', str(LOGS_DIR)))

# Temporary directories
TEMP_DIR = Path(os.environ.get('TMPDIR', '/tmp')) / 'genomevault'
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Create all directories
for dir_path in [
    DATA_DIR, RESULTS_DIR, LOGS_DIR, KEYS_DIR, 
    CIRCUITS_DIR, CONFIGS_DIR, RAW_DATA_DIR,
    PROCESSED_DATA_DIR, ENCRYPTED_DATA_DIR
]:
    dir_path.mkdir(parents=True, exist_ok=True)

def get_data_path(filename: str, data_type: str = 'raw') -> Path:
    """Get path for a data file."""
    if data_type == 'raw':
        return RAW_DATA_DIR / filename
    elif data_type == 'processed':
        return PROCESSED_DATA_DIR / filename
    elif data_type == 'encrypted':
        return ENCRYPTED_DATA_DIR / filename
    else:
        return DATA_DIR / filename

def get_result_path(filename: str) -> Path:
    """Get path for a result file."""
    return RESULTS_DIR / filename

def get_log_path(filename: str) -> Path:
    """Get path for a log file."""
    return LOGS_DIR / filename

def get_config_path(filename: str) -> Path:
    """Get path for a config file."""
    return CONFIGS_DIR / filename
