"""Path configuration for GenomeVault project."""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"
LOGS_DIR = PROJECT_ROOT / "logs"
ATTIC_DIR = PROJECT_ROOT / "attic"

# Package directories
GENOMEVAULT_DIR = PROJECT_ROOT / "genomevault"
TESTS_DIR = PROJECT_ROOT / "tests"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Config directories
CONFIG_DIR = GENOMEVAULT_DIR / "config"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Cache subdirectories
PIR_CACHE_DIR = CACHE_DIR / "pir"
HD_CACHE_DIR = CACHE_DIR / "hypervectors"
ZK_CACHE_DIR = CACHE_DIR / "zk_proofs"


# Ensure critical directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [DATA_DIR, CACHE_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


# Initialize on import
ensure_directories()
