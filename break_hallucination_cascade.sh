#!/bin/bash
# Manual cleanup of __init__.py files to break the hallucination cascade

echo "🧹 Breaking the Hallucination Cascade!"
echo "====================================="

cd /Users/rohanvinaik/genomevault

# 1. Fix core/__init__.py
echo "Fixing core/__init__.py..."
cat > core/__init__.py << 'EOF'
"""
GenomeVault Core Package
"""

from .config import Config, get_config
from .constants import OmicsType
from .exceptions import (
    GenomeVaultError,
    ValidationError,
    PrivacyError,
    CryptographicError,
    ProofError,
    PIRError,
    BlockchainError,
    HIPAAComplianceError,
    CompressionError,
    HypervectorError,
    BindingError,
    NetworkError,
    StorageError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    ConfigurationError,
    ProcessingError,
    ClinicalError,
    ResearchError,
)

__all__ = [
    'Config',
    'get_config',
    'OmicsType',
    'GenomeVaultError',
    'ValidationError',
    'PrivacyError',
    'CryptographicError',
    'ProofError',
    'PIRError',
    'BlockchainError',
    'HIPAAComplianceError',
    'CompressionError',
    'HypervectorError',
    'BindingError',
    'NetworkError',
    'StorageError',
    'AuthenticationError',
    'AuthorizationError',
    'RateLimitError',
    'ConfigurationError',
    'ProcessingError',
    'ClinicalError',
    'ResearchError',
]
EOF

# 2. Fix local_processing/__init__.py
echo "Fixing local_processing/__init__.py..."
cat > local_processing/__init__.py << 'EOF'
"""
GenomeVault Local Processing Package
"""

from .sequencing import (
    SequencingProcessor,
    DifferentialStorage,
    GenomicProfile,
    Variant,
    QualityMetrics,
)

from .transcriptomics import (
    TranscriptomicsProcessor,
    ExpressionProfile,
    GeneExpression,
)

from .epigenetics import (
    EpigeneticsProcessor,
    MethylationProfile,
    MethylationSite,
)

from .proteomics import (
    ProteomicsProcessor,
    ProteinProfile,
    ProteinAbundance,
)

from .phenotypes import (
    PhenotypeProcessor,
    ClinicalProfile,
    Phenotype,
)

__all__ = [
    # Sequencing
    'SequencingProcessor',
    'DifferentialStorage',
    'GenomicProfile',
    'Variant',
    'QualityMetrics',
    
    # Transcriptomics
    'TranscriptomicsProcessor',
    'ExpressionProfile',
    'GeneExpression',
    
    # Epigenetics
    'EpigeneticsProcessor',
    'MethylationProfile',
    'MethylationSite',
    
    # Proteomics
    'ProteomicsProcessor',
    'ProteinProfile',
    'ProteinAbundance',
    
    # Phenotypes
    'PhenotypeProcessor',
    'ClinicalProfile',
    'Phenotype',
]
EOF

# 3. Fix zk_proofs/__init__.py
echo "Fixing zk_proofs/__init__.py..."
cat > zk_proofs/__init__.py << 'EOF'
"""
GenomeVault Zero-Knowledge Proofs Package
"""

from .prover import (
    Prover,
    Circuit,
    Proof,
    CircuitLibrary,
)

from .verifier import (
    Verifier,
    VerificationResult,
)

# Import submodules
from . import circuits

__all__ = [
    'Prover',
    'Circuit', 
    'Proof',
    'CircuitLibrary',
    'Verifier',
    'VerificationResult',
    'circuits',
]
EOF

# 4. Fix pir/__init__.py (if it exists)
if [ -d "pir" ]; then
    echo "Fixing pir/__init__.py..."
    cat > pir/__init__.py << 'EOF'
"""
GenomeVault Private Information Retrieval Package
"""

from .client import PIRClient
from .server import PIRServer
from .coordinator import PIRCoordinator

__all__ = [
    'PIRClient',
    'PIRServer',
    'PIRCoordinator',
]
EOF
fi

# 5. Fix blockchain/__init__.py (if it exists)
if [ -d "blockchain" ]; then
    echo "Fixing blockchain/__init__.py..."
    cat > blockchain/__init__.py << 'EOF'
"""
GenomeVault Blockchain Package
"""

from .node import Node
from .consensus import ConsensusEngine
from .contracts import SmartContract
from .governance import GovernanceModule

__all__ = [
    'Node',
    'ConsensusEngine',
    'SmartContract',
    'GovernanceModule',
]
EOF
fi

echo ""
echo "✅ Cleaned up __init__.py files!"
echo ""
echo "🧪 Running import test..."
python3 << 'EOF'
try:
    print("Testing imports after cleanup...")
    from core.config import get_config
    print("✅ core.config works")
    
    from utils import get_logger
    print("✅ utils works")
    
    from local_processing import SequencingProcessor
    print("✅ local_processing works")
    
    from hypervector_transform import HypervectorEncoder
    print("✅ hypervector_transform works")
    
    from zk_proofs import Prover
    print("✅ zk_proofs works")
    
    print("\n✅ All imports successful!")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
EOF

echo ""
echo "🎯 Running minimal test..."
python3 minimal_test.py
