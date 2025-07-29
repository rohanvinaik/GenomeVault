from __future__ import annotations

from typing import Dict, Optional


class GenomeVaultError(Exception):
    """Base exception for GenomeVault."""

    def __init__(self, message: str = "", *, context: dict | None = None):
        super().__init__(message)
        self.context = context or {}

    def __str__(self) -> str:
        base = super().__str__() or self.__class__.__name__
        if self.context:
            return f"{self.__class__.__name__}: {base} | context={self.context}"
        return f"{self.__class__.__name__}: {base}"


class ConfigurationError(GenomeVaultError): ...


class ValidationError(GenomeVaultError): ...


class ProjectionError(GenomeVaultError): ...


class EncodingError(GenomeVaultError): ...


class ZKProofError(GenomeVaultError): ...


class PIRProtocolError(GenomeVaultError): ...


class LedgerError(GenomeVaultError): ...


class FederatedError(GenomeVaultError): ...


class APISchemaError(GenomeVaultError): ...


__all__ = [
    "GenomeVaultError",
    "ConfigurationError",
    "ValidationError",
    "ProjectionError",
    "EncodingError",
    "ZKProofError",
    "PIRProtocolError",
    "LedgerError",
    "FederatedError",
    "APISchemaError",
]
