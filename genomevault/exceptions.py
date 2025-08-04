class GenomeVaultError(Exception):
    pass


class ConfigError(GenomeVaultError):
    pass


class ValidationError(GenomeVaultError):
    pass


class ComputeError(GenomeVaultError):
    pass


class HypervectorError(GenomeVaultError):
    """Exception raised for hypervector operations."""

    pass


class ZKProofError(GenomeVaultError):
    """Exception raised for zero-knowledge proof operations."""

    pass
