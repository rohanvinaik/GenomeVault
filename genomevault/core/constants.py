"""
System-wide constants for GenomeVault
"""

from enum import Enum, IntEnum


class NodeType(str, Enum):
    """Node hardware classes"""

    LIGHT = "light"
    FULL = "full"
    ARCHIVE = "archive"


class NodeClassWeight(IntEnum):
    """Hardware class weights (c values)"""

    LIGHT = 1
    FULL = 4
    ARCHIVE = 8


class SignatoryWeight(IntEnum):
    """Trust signatory weight (s values)"""

    NON_SIGNER = 0
    TRUSTED_SIGNATORY = 10


class CompressionTier(str, Enum):
    """Data compression tiers"""

    MINI = "mini"  # ~25KB - 5,000 most-studied SNPs
    CLINICAL = "clinical"  # ~300KB - ACMG + PharmGKB variants
    FULL = "full"  # 100-200KB per modality


# Hypervector dimensions by tier
HYPERVECTOR_DIMENSIONS = {
    "base": 10000,
    "mid": 15000,
    "high": 20000,
}


# Biological data types
class OmicsType(str, Enum):
    """Multi-omics data types"""

    GENOMIC = "genomic"
    TRANSCRIPTOMIC = "transcriptomic"
    EPIGENETIC = "epigenetic"
    PROTEOMIC = "proteomic"
    PHENOTYPIC = "phenotypic"


# Privacy constants
MIN_PIR_SERVERS = 3
PIR_THRESHOLD = 0.98  # Server honesty assumption
PRIVACY_FAILURE_THRESHOLD = 1e-4  # Maximum acceptable P_fail

# ZK Proof constants
PROOF_SIZE_BYTES = 384
MAX_VERIFICATION_TIME_MS = 25
RECURSIVE_PROOF_DEPTH = 5

# Blockchain constants
BLOCK_TIME_SECONDS = 6
CREDITS_PER_BLOCK_BASE = 1
CREDITS_SIGNATORY_BONUS = 2
AUDIT_SLASH_PERCENTAGE = 25

# Clinical thresholds
GLUCOSE_THRESHOLD_MG_DL = 126  # Diabetes threshold
HBA1C_THRESHOLD_PERCENT = 6.5
GENETIC_RISK_SCORE_THRESHOLD = 0.8

# Reference genome versions
REFERENCE_GENOME_VERSION = "GRCh38"
REFERENCE_GENOME_PATCH = "p14"

# File size limits
MAX_VCF_SIZE_GB = 3
MAX_BAM_SIZE_GB = 50
MAX_FASTQ_SIZE_GB = 100

# API rate limits
API_RATE_LIMIT_PER_MINUTE = 60
API_RATE_LIMIT_BURST = 100

# Cryptographic parameters
AES_KEY_SIZE = 256
RSA_KEY_SIZE = 4096
ECC_CURVE = "secp256k1"
POST_QUANTUM_ALGORITHM = "CRYSTALS-Dilithium3"

# Network timeouts (milliseconds)
PIR_QUERY_TIMEOUT_MS = 1000
BLOCKCHAIN_TIMEOUT_MS = 30000
API_TIMEOUT_MS = 60000

# Healthcare standards
HL7_VERSION = "2.5.1"
FHIR_VERSION = "R4"
LOINC_VERSION = "2.73"
SNOMED_VERSION = "2023-09-01"

# Federated learning parameters
FL_MIN_PARTICIPANTS = 10
FL_ROUNDS = 100
FL_LEARNING_RATE = 0.01
FL_DIFFERENTIAL_PRIVACY_EPSILON = 1.0

# Population genetics thresholds
MIN_ALLELE_FREQUENCY = 0.01
HARDY_WEINBERG_P_VALUE = 0.05
LINKAGE_DISEQUILIBRIUM_R2 = 0.8

# TDA parameters
PERSISTENCE_THRESHOLD = 0.1
MAX_HOMOLOGY_DIMENSION = 2
FILTRATION_STEPS = 100

# Error messages
ERROR_MESSAGES = {
    "INVALID_GENOME": "Invalid genome format or size",
    "PRIVACY_VIOLATION": "Operation would violate privacy guarantees",
    "INSUFFICIENT_SERVERS": "Not enough PIR servers available",
    "PROOF_VERIFICATION_FAILED": "Zero-knowledge proof verification failed",
    "UNAUTHORIZED": "Unauthorized access attempt",
    "HIPAA_COMPLIANCE_REQUIRED": "HIPAA compliance verification required",
}

# HDC Error Handling Configuration
HDC_ERROR_CONFIG = {
    "dimension_caps": {
        "mini": 50000,
        "clinical": 100000,
        "research": 150000,
        "full": 200000,
    },
    "default_epsilon": 0.01,
    "default_delta_exp": 15,
    "ecc_enabled_default": True,
    "ecc_parity_g": 3,
    "max_repeats": 100,
    "presets": {
        "fast": {"epsilon": 0.02, "delta_exp": 10, "ecc": False},
        "balanced": {"epsilon": 0.01, "delta_exp": 15, "ecc": True},
        "high_accuracy": {"epsilon": 0.005, "delta_exp": 20, "ecc": True},
        "clinical_standard": {"epsilon": 0.001, "delta_exp": 25, "ecc": True},
    },
}
