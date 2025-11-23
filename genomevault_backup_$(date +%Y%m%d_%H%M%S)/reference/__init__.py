"""
GenomeVault Reference Genome Management

This module provides utilities for managing reference genomes,
including:
- Probabilistic alignment with exponential certainty decay
- Byzantine Consensus Privacy Stack
- Superposition Consensus with graph-based multiple paths
- Hierarchical SNP/indel detection (1-nt, 2-nt, 3+-nt)
- Statistical significance testing for alignment patterns
- Advanced indel detection with iterative realignment
- Comprehensive alignment challenge detection (SVs, CNVs, repeats, artifacts)

See: docs/guides/PROBABILISTIC_ALIGNMENT_PRIVACY_STACK.md for complete guide
"""

from .byzantine_consensus_builder import (
    ByzantineConsensusBuilder,
    ConsensusBase,
    build_consensus_reference,
)

from .superposition_consensus_builder import (
    SuperpositionConsensusBuilder,
    SuperpositionPath,
    SuperpositionNode,
    PopulationVariant,
    ConservedRegion,
    VariableRegion,
    PathSelectionStrategy,
    build_superposition_consensus,
)

from .user_alignment_randomizer import (
    UserAlignmentRandomizer,
    AlignmentParameters,
    create_user_randomizer,
)

from .rolling_reference_pool import (
    RollingReferencePool,
    QueryRecord,
    GenomeReference,
    PoolStatistics,
    UpdateStrategy,
    PoolUpdateMethod,
)

from .probabilistic_alignment_system import (
    ProbabilisticAligner,
    SNPDatabase,
    ChromosomeSNPIndex,
    SNPRecord,
    AlignmentCertainty,
    IndelCandidate,
)

from .advanced_indel_detection import (
    AdvancedIndelDetector,
    IndelSignature,
    IndelType,
    IndelDatabase,
    SmithWatermanAligner,
    HaplotypeCandidate,
)

from .comprehensive_alignment_engine import (
    ComprehensiveAlignmentEngine,
    AlignmentChallenge,
    AlignmentChallengeType,
    StructuralVariantDetector,
    RepetitiveElementHandler,
    LowComplexityRegionAnalyzer,
    CopyNumberAnalyzer,
    SequencingArtifactFilter,
    AlignmentAmbiguityResolver,
    BiologicalComplexityHandler,
)

__all__ = [
    # Byzantine Consensus
    'ByzantineConsensusBuilder',
    'ConsensusBase',
    'build_consensus_reference',

    # Superposition Consensus (Graph-Based)
    'SuperpositionConsensusBuilder',
    'SuperpositionPath',
    'SuperpositionNode',
    'PopulationVariant',
    'ConservedRegion',
    'VariableRegion',
    'PathSelectionStrategy',
    'build_superposition_consensus',

    # User-Specific Alignment Randomization (SHA-256² Security)
    'UserAlignmentRandomizer',
    'AlignmentParameters',
    'create_user_randomizer',

    # Rolling Reference Pool (Entropy-Based Rotation)
    'RollingReferencePool',
    'QueryRecord',
    'GenomeReference',
    'PoolStatistics',
    'UpdateStrategy',
    'PoolUpdateMethod',

    # Probabilistic Alignment (Basic)
    'ProbabilisticAligner',
    'SNPDatabase',
    'ChromosomeSNPIndex',
    'SNPRecord',
    'AlignmentCertainty',
    'IndelCandidate',

    # Advanced Indel Detection
    'AdvancedIndelDetector',
    'IndelSignature',
    'IndelType',
    'IndelDatabase',
    'SmithWatermanAligner',
    'HaplotypeCandidate',

    # Comprehensive Alignment Engine
    'ComprehensiveAlignmentEngine',
    'AlignmentChallenge',
    'AlignmentChallengeType',
    'StructuralVariantDetector',
    'RepetitiveElementHandler',
    'LowComplexityRegionAnalyzer',
    'CopyNumberAnalyzer',
    'SequencingArtifactFilter',
    'AlignmentAmbiguityResolver',
    'BiologicalComplexityHandler',
]
