"""
KAN (Kolmogorov-Arnold Network) Hybrid Architecture for GenomeVault

This module implements the enhanced KAN-HD hybrid architecture that combines:
- Kolmogorov-Arnold Networks for extreme compression (10-500x)
- Hyperdimensional computing for privacy preservation
- Hierarchical multi-modal encoding
- Federated learning capabilities
- Scientific interpretability analysis
- Real-time performance tuning
"""

from .compression import AdaptiveKANCompressor, KANCompressor

# Enhanced hybrid encoders
from .enhanced_hybrid_encoder import CompressionStrategy, EnhancedKANHybridEncoder

# Federated learning
from .federated_kan import (
    FederatedKANCoordinator,
    FederatedKANParticipant,
    FederatedUpdate,
    FederationConfig,
)

# Hierarchical encoding
from .hierarchical_encoding import (
    AdaptiveDimensionalityCalculator,
    DataModality,
    EncodingSpecification,
    HierarchicalHypervectorEncoder,
    MultiResolutionVector,
)
from .hybrid_encoder import KANHybridEncoder, StreamingKANHybridEncoder

# Core KAN components
from .kan_layer import KANLayer, LinearKAN, SplineFunction

# Scientific interpretability
from .scientific_interpretability import (
    BiologicalFunction,
    DiscoveredFunction,
    InterpretableKANHybridEncoder,
    KANFunctionAnalyzer,
    PatternAnalysis,
)

__all__ = [
    # Core components
    "KANLayer",
    "LinearKAN",
    "SplineFunction",
    "KANCompressor",
    "AdaptiveKANCompressor",
    # Enhanced encoders
    "EnhancedKANHybridEncoder",
    "KANHybridEncoder",
    "StreamingKANHybridEncoder",
    "CompressionStrategy",
    # Hierarchical encoding
    "HierarchicalHypervectorEncoder",
    "EncodingSpecification",
    "DataModality",
    "MultiResolutionVector",
    "AdaptiveDimensionalityCalculator",
    # Federated learning
    "FederatedKANCoordinator",
    "FederatedKANParticipant",
    "FederationConfig",
    "FederatedUpdate",
    # Scientific interpretability
    "InterpretableKANHybridEncoder",
    "KANFunctionAnalyzer",
    "BiologicalFunction",
    "DiscoveredFunction",
    "PatternAnalysis",
]
