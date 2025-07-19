#!/bin/bash
# Complete solution for GenomeVault

echo "🚀 GenomeVault Complete Solution"
echo "================================"

cd /Users/rohanvinaik/genomevault

# Step 1: Install ALL dependencies including structlog
echo "📦 Installing all dependencies..."
pip install -q structlog scikit-learn biopython pysam pynacl pyyaml uvicorn web3 eth-account seaborn
pip install -q pydantic-settings pydantic>=2.0.0

# Step 2: Fix the config import issue in zk_proofs/prover.py
echo "🔧 Fixing config imports..."
if [ -f "zk_proofs/prover.py" ]; then
    # Replace the config import with the correct one
    sed -i '' 's/from utils.config import config/from core.config import get_config\nconfig = get_config()/' zk_proofs/prover.py
fi

# Step 3: Fix the logging imports in zk_proofs/prover.py
echo "🔧 Fixing logging imports..."
if [ -f "zk_proofs/prover.py" ]; then
    # Update logging imports
    sed -i '' 's/from utils.logging import audit_logger, logger, performance_logger/from utils.logging import get_logger\nlogger = get_logger(__name__)\naudit_logger = logger\nperformance_logger = logger/' zk_proofs/prover.py
fi

# Step 4: Update utils/__init__.py to avoid config conflicts
echo "🔧 Updating utils/__init__.py..."
cat > utils/__init__.py << 'EOF'
"""
GenomeVault Utilities Package

Core utilities for configuration, logging, and encryption.
"""

# Import config from core instead of utils to avoid conflicts
from core.config import Config, get_config

from .encryption import (
    AESGCMCipher,
    ChaCha20Poly1305,
    EncryptionManager,
    KeyDerivation,
    RSAEncryption,
    SecureRandom,
    ThresholdCrypto,
    ThresholdShare,
    generate_secure_key,
    secure_hash,
)
from .logging import (
    GenomeVaultLogger,
    LogEvent,
    PrivacyLevel,
    configure_logging,
    get_logger,
    log_genomic_operation,
    log_operation,
)

__all__ = [
    # Config
    'Config',
    'get_config',
    
    # Logging
    'get_logger',
    'configure_logging',
    'log_operation',
    'log_genomic_operation',
    'LogEvent',
    'PrivacyLevel',
    'GenomeVaultLogger',
    
    # Encryption
    'AESGCMCipher',
    'ChaCha20Poly1305',
    'RSAEncryption',
    'ThresholdCrypto',
    'ThresholdShare',
    'KeyDerivation',
    'SecureRandom',
    'EncryptionManager',
    'generate_secure_key',
    'secure_hash'
]

# Version info
__version__ = '3.0.0'
__author__ = 'GenomeVault Team'
EOF

# Step 5: Create a minimal test to verify everything works
echo "🧪 Creating minimal test..."
cat > minimal_test.py << 'EOF'
#!/usr/bin/env python3
"""Minimal test to verify GenomeVault is working"""

try:
    # Test basic imports
    from core.config import Config, get_config
    print("✅ Core config imported")
    
    from utils import get_logger
    print("✅ Utils imported")
    
    from local_processing.sequencing import SequencingProcessor
    print("✅ Sequencing module imported")
    
    from hypervector_transform.encoding import HypervectorEncoder
    print("✅ Hypervector module imported")
    
    from zk_proofs.prover import Prover
    print("✅ ZK proofs module imported")
    
    # Test creating instances
    config = get_config()
    print(f"✅ Config created: node_type={config.node_type}")
    
    logger = get_logger(__name__)
    print("✅ Logger created")
    
    print("\n✅ All imports successful!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF

chmod +x minimal_test.py

# Step 6: Run the minimal test
echo ""
echo "🧪 Running minimal test..."
python3 minimal_test.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🧪 Running pytest..."
    python -m pytest tests/test_simple.py -v
else
    echo ""
    echo "❌ Minimal test failed"
fi

echo ""
echo "✅ Complete!"
echo ""
echo "If tests still fail, try:"
echo "  python3 -m pip install --upgrade pip"
echo "  pip install -r requirements.txt --force-reinstall"
