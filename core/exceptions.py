"""
Custom exceptions for GenomeVault
"""


class GenomeVaultError(Exception):
    """Base exception for all GenomeVault errors"""
    pass


class ValidationError(GenomeVaultError):
    """Raised when data validation fails"""
    pass


class PrivacyError(GenomeVaultError):
    """Raised when an operation would violate privacy guarantees"""
    pass


class CryptographicError(GenomeVaultError):
    """Raised when cryptographic operations fail"""
    pass


class ProofError(CryptographicError):
    """Raised when zero-knowledge proof generation or verification fails"""
    pass


class PIRError(GenomeVaultError):
    """Raised when Private Information Retrieval fails"""
    pass


class BlockchainError(GenomeVaultError):
    """Raised when blockchain operations fail"""
    pass


class HIPAAComplianceError(GenomeVaultError):
    """Raised when HIPAA compliance requirements are not met"""
    pass


class CompressionError(GenomeVaultError):
    """Raised when data compression/decompression fails"""
    pass


class HypervectorError(GenomeVaultError):
    """Raised when hypervector operations fail"""
    pass


class NetworkError(GenomeVaultError):
    """Raised when network operations fail"""
    pass


class StorageError(GenomeVaultError):
    """Raised when storage operations fail"""
    pass


class AuthenticationError(GenomeVaultError):
    """Raised when authentication fails"""
    pass


class AuthorizationError(GenomeVaultError):
    """Raised when authorization fails"""
    pass


class RateLimitError(GenomeVaultError):
    """Raised when rate limits are exceeded"""
    pass


class ConfigurationError(GenomeVaultError):
    """Raised when configuration is invalid"""
    pass


class ProcessingError(GenomeVaultError):
    """Raised when data processing fails"""
    pass


class ClinicalError(GenomeVaultError):
    """Raised when clinical operations fail"""
    pass


class ResearchError(GenomeVaultError):
    """Raised when research operations fail"""
    pass
