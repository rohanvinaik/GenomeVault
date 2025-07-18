"""
GenomeVault Blockchain Package

This package implements the blockchain layer including:
- Consensus mechanisms
- Smart contracts
- Governance (DAO)
- HIPAA fast-track verification
- Node management
"""

from .node import BlockchainNode, NodeType
from .governance import (
    GovernanceSystem,
    ProposalType,
    ProposalStatus,
    CommitteeType,
    HIPAAOracle
)

# Import HIPAA components if available
try:
    from .hipaa import (
        HIPAAVerifier,
        HIPAACredentials,
        VerificationStatus,
        NPIRegistry
    )
    from .hipaa.integration import (
        HIPAANodeIntegration,
        HIPAAGovernanceIntegration
    )
    HIPAA_AVAILABLE = True
except ImportError:
    HIPAA_AVAILABLE = False

__all__ = [
    # Node management
    'BlockchainNode',
    'NodeType',
    
    # Governance
    'GovernanceSystem',
    'ProposalType',
    'ProposalStatus',
    'CommitteeType',
    'HIPAAOracle',
]

# Add HIPAA exports if available
if HIPAA_AVAILABLE:
    __all__.extend([
        'HIPAAVerifier',
        'HIPAACredentials',
        'VerificationStatus',
        'NPIRegistry',
        'HIPAANodeIntegration',
        'HIPAAGovernanceIntegration'
    ])

__version__ = '0.1.0'
