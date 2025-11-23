"""Private Information Retrieval implementations for pir."""

from .core import PIRConfig, PIRClient, SimplePIRServer, SimplePIR, create_pir_system
from .servers import (
    PIRServer,
    ShardHealth,
    ShardManager,
    ShardedPIRServer,
    FECEncoder,
)
from .engine import PIREngine
from .secure_wrapper import SecurePIRServer, SecurePIRWrapper
from .it_pir_protocol import PIRParameters, PIRProtocol, BatchPIRProtocol

# Create aliases for backward compatibility and expected names
ITPrivateInformationRetrieval = PIRProtocol  # Alias for tests expecting this name
ITPIRProtocol = PIRProtocol  # Another common alias

__all__ = [
    "BatchPIRProtocol",
    "FECEncoder",
    "ITPrivateInformationRetrieval",  # Alias
    "ITPIRProtocol",  # Alias
    "PIRClient",
    "PIRConfig",
    "PIREngine",
    "PIRParameters",
    "PIRProtocol",
    "PIRServer",
    "SecurePIRServer",
    "SecurePIRWrapper",
    "ShardHealth",
    "ShardManager",
    "ShardedPIRServer",
    "SimplePIR",
    "SimplePIRServer",
    "create_pir_system",
]
