"""
Custom exceptions for GenomeVault
"""


class GenomeVaultError(Exception):
    """Base exception for all GenomeVault errors"""


class ValidationError(GenomeVaultError):
    """Raised when data validation fails"""


class PrivacyError(GenomeVaultError):
    """Raised when an operation would violate privacy guarantees"""


class CryptographicError(GenomeVaultError):
    """Raised when cryptographic operations fail"""


class ProofError(CryptographicError):
    """Raised when zero-knowledge proof generation or verification fails"""


class CircuitError(ProofError):
    """Raised when circuit operations fail"""


class PIRError(GenomeVaultError):
    """Raised when Private Information Retrieval fails"""


class BlockchainError(GenomeVaultError):
    """Raised when blockchain operations fail"""


class HIPAAComplianceError(GenomeVaultError):
    """Raised when HIPAA compliance requirements are not met"""


class CompressionError(GenomeVaultError):
    """Raised when data compression/decompression fails"""


class HypervectorError(GenomeVaultError):
    """Raised when hypervector operations fail"""


class BindingError(HypervectorError):
    """Raised when hypervector binding operations fail"""


class EncodingError(HypervectorError):
    """Raised when hypervector encoding operations fail"""


class MappingError(HypervectorError):
    """Raised when hypervector mapping operations fail"""


class NetworkError(GenomeVaultError):
    """Raised when network operations fail"""


class StorageError(GenomeVaultError):
    """Raised when storage operations fail"""


class AuthenticationError(GenomeVaultError):
    """Raised when authentication fails"""


class AuthorizationError(GenomeVaultError):
    """Raised when authorization fails"""


class RateLimitError(GenomeVaultError):
    """Raised when rate limits are exceeded"""


class ConfigurationError(GenomeVaultError):
    """Raised when configuration is invalid"""


class ProcessingError(GenomeVaultError):
    """Raised when data processing fails"""


class ClinicalError(GenomeVaultError):
    """Raised when clinical operations fail"""


class ResearchError(GenomeVaultError):
    """Raised when research operations fail"""
