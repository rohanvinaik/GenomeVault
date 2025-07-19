#!/bin/bash
# Final fix script for GenomeVault - includes structlog

echo "🚀 GenomeVault Final Fix"
echo "========================"

cd /Users/rohanvinaik/genomevault

# Step 1: Install structlog and other missing dependencies
echo "📦 Installing structlog and remaining dependencies..."
pip install -q structlog

# Step 2: Fix the config import conflict
echo "🔧 Fixing config import conflicts..."

# The issue is that we have utils/config.py and core/config.py
# Let's update the utils/__init__.py to use core.config instead

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

# Step 3: Fix the zk_proofs/prover.py import
echo "🔧 Fixing zk_proofs imports..."

# Check if the file exists and fix it
if [ -f "zk_proofs/prover.py" ]; then
    sed -i '' 's/from utils.config import config/from core.config import get_config/' zk_proofs/prover.py
    sed -i '' 's/from utils import config/from core.config import get_config/' zk_proofs/prover.py
fi

# Step 4: Run tests
echo ""
echo "🧪 Testing imports..."
python3 simple_test.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🧪 Running pytest..."
    python -m pytest tests/test_simple.py -v
else
    echo ""
    echo "❌ Import test failed. Checking what's still missing..."
    python3 -c "import structlog; print('✅ structlog installed')" 2>/dev/null || echo "❌ structlog not installed"
fi

echo ""
echo "✅ Fix complete!"
