"""Core exceptions for GenomeVault."""


class GenomeVaultError(Exception):
    """Base exception for GenomeVault."""

    pass


class HypervectorError(GenomeVaultError):
    """Exception raised for hypervector operations."""

    pass


class ZKProofError(GenomeVaultError):
    """Exception raised for zero-knowledge proof operations."""

    pass


class ValidationError(GenomeVaultError):
    """Exception raised for validation failures."""

    pass


class ConfigurationError(GenomeVaultError):
    """Exception raised for configuration issues."""

    pass


class ProjectionError(GenomeVaultError):
    """Exception raised for projection operations."""

    pass
