"""
Private Information Retrieval (PIR) system for GenomeVault.

This package implements information-theoretic PIR for secure genomic data access
without revealing what data is being accessed.
"""

# Client components
from .client import (
    PIRClient,
    PIRServer,
    PIRQuery,
    PIRResponse
)

from .client.query_builder import (
    PIRQueryBuilder,
    GenomicQuery,
    QueryResult,
    QueryType
)

# Server components
from .server.pir_server import (
    PIRServer as PIRServerInstance,
    TrustedSignatoryServer,
    DatabaseShard
)

from .server.shard_manager import (
    ShardManager,
    ShardMetadata,
    ShardDistribution
)

# Network components
from .network.coordinator import (
    PIRNetworkCoordinator,
    NetworkTopology,
    ServerHealth
)

# Reference data components
from .reference_data.manager import (
    ReferenceDataManager,
    GenomicRegion,
    PangenomeNode,
    PangenomeEdge,
    VariantAnnotation,
    ReferenceDataType
)

__all__ = [
    # Client
    'PIRClient',
    'PIRServer',
    'PIRQuery',
    'PIRResponse',
    'PIRQueryBuilder',
    'GenomicQuery',
    'QueryResult',
    'QueryType',
    
    # Server
    'PIRServerInstance',
    'TrustedSignatoryServer',
    'DatabaseShard',
    'ShardManager',
    'ShardMetadata',
    'ShardDistribution',
    
    # Network
    'PIRNetworkCoordinator',
    'NetworkTopology',
    'ServerHealth',
    
    # Reference data
    'ReferenceDataManager',
    'GenomicRegion',
    'PangenomeNode',
    'PangenomeEdge',
    'VariantAnnotation',
    'ReferenceDataType'
]

# Version info
__version__ = '1.0.0'
__author__ = 'GenomeVault Team'
