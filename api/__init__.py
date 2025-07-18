"""
GenomeVault API Package

Core network API implementation using FastAPI.
Provides endpoints for:
- Network topology discovery
- Credit management
- Audit challenges
- Processing pipelines
- Hypervector operations
- Zero-knowledge proofs
"""

from .app import app, get_app_instance
from .main import (
    TopologyRequest,
    TopologyResponse,
    CreditRedeemRequest,
    CreditRedeemResponse,
    AuditChallengeRequest,
    AuditChallengeResponse
)

# Import routers
from .routers.topology import router as topology_router
from .routers.credit import router as credit_router

__all__ = [
    # Application
    'app',
    'get_app_instance',
    
    # Request/Response models
    'TopologyRequest',
    'TopologyResponse',
    'CreditRedeemRequest',
    'CreditRedeemResponse',
    'AuditChallengeRequest',
    'AuditChallengeResponse',
    
    # Routers
    'topology_router',
    'credit_router'
]

# Version info
__version__ = '1.0.0'
__author__ = 'GenomeVault Team'
