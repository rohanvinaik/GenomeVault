class GenomeVaultError(Exception):
    pass


class ConfigError(GenomeVaultError):
    pass


class ValidationError(GenomeVaultError):
    pass


class ComputeError(GenomeVaultError):
    pass
