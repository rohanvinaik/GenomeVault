"""
GenomeVault Advanced Analysis Package

Research modules and AI integration for advanced genomic analysis:
- Hyperdimensional AI research
- Federated learning
- Topological data analysis
- Graph algorithms for population genomics
- Differential equation models
"""

# Federated learning components
from .federated_learning.client import FederatedClient
from .federated_learning.coordinator import FederatedCoordinator

# TDA components (when implemented)
try:
    from .tda.persistence import (
        PersistenceDiagram,
        compute_persistence,
        plot_persistence_diagram
    )
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False

__all__ = [
    # Federated learning
    'FederatedClient',
    'FederatedCoordinator',
]

# Add TDA exports if available
if TDA_AVAILABLE:
    __all__.extend([
        'PersistenceDiagram',
        'compute_persistence',
        'plot_persistence_diagram'
    ])

# Version info
__version__ = '1.0.0'
__author__ = 'GenomeVault Team'

# Module status
MODULE_STATUS = {
    'federated_learning': 'partial',
    'hypervector_engine': 'planned',
    'graph_algorithms': 'planned',
    'tda': 'partial',
    'differential_models': 'planned',
    'ai_integration': 'planned',
    'population_genomics': 'planned'
}
