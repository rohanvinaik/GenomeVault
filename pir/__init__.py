"""
Private Information Retrieval (PIR) system for GenomeVault.

This package implements information-theoretic PIR for secure genomic data access
without revealing what data is being accessed.
"""

# Client components
from .client import PIRClient, PIRQuery, PIRResponse, PIRServer
from .client.query_builder import GenomicQuery, PIRQueryBuilder, QueryResult, QueryType

# Network components
from .network.coordinator import NetworkTopology, PIRNetworkCoordinator, ServerHealth

# Reference data components
from .reference_data.manager import (
    GenomicRegion,
    PangenomeEdge,
    PangenomeNode,
    ReferenceDataManager,
    ReferenceDataType,
    VariantAnnotation,
)

# Server components
from .server.pir_server import DatabaseShard
from .server.pir_server import PIRServer as PIRServerInstance
from .server.pir_server import TrustedSignatoryServer
from .server.shard_manager import ShardDistribution, ShardManager, ShardMetadata

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
