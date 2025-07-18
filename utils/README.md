# GenomeVault Utilities

Core utility modules providing configuration management, logging, and cryptographic services for the GenomeVault platform.

## Modules

### config.py
- **Purpose**: Centralized configuration management
- **Features**:
  - Environment-specific settings (development, staging, production)
  - Secure secrets management with encryption
  - Configuration validation
  - Support for YAML/JSON config files
  - Environment variable overrides
  - Dual-axis node model calculations
  - PIR failure probability computations

### logging.py
- **Purpose**: Privacy-aware logging system
- **Features**:
  - Structured JSON logging
  - Privacy filter for sensitive data redaction
  - Audit trail logging for compliance
  - Metrics collection and batching
  - Operation context tracking
  - Standardized event types
  - Integration with monitoring systems

### encryption.py
- **Purpose**: Cryptographic utilities and key management
- **Features**:
  - AES-256-GCM authenticated encryption
  - ChaCha20-Poly1305 AEAD encryption
  - RSA-4096 for key exchange
  - Shamir's Secret Sharing for threshold cryptography
  - Key derivation (PBKDF2, HKDF)
  - Secure random generation
  - Post-quantum readiness hooks

## Usage Examples

### Configuration
```python
from genomevault.utils import get_config, init_config

# Initialize configuration
config = init_config(environment="production")

# Access configuration values
hypervector_dims = config.crypto.hypervector_dimensions
epsilon = config.privacy.epsilon

# Calculate node voting power
voting_power = config.get_node_voting_power()

# Get PIR failure probability
p_fail = config.get_pir_failure_probability()
```

### Logging
```python
from genomevault.utils import get_logger, LogEvent

logger = get_logger(__name__)

# Basic logging
logger.info("Starting genomic processing")

# Operation context
with logger.operation_context("variant_analysis", user_id="user123"):
    # Perform operation
    pass

# Log standardized events
logger.log_event(LogEvent.PROCESSING_START, "Starting variant analysis")

# Audit logging
logger.audit.log_computation(
    user_id="user123",
    operation="prs_calculation",
    privacy_budget_used=0.1
)
```

### Encryption
```python
from genomevault.utils import AESGCMCipher, ThresholdCrypto, EncryptionManager

# AES-GCM encryption
key = AESGCMCipher.generate_key()
ciphertext, nonce, tag = AESGCMCipher.encrypt(b"sensitive data", key)
plaintext = AESGCMCipher.decrypt(ciphertext, key, nonce, tag)

# Threshold secret sharing
secret = b"master_secret_key"
shares = ThresholdCrypto.split_secret(secret, threshold=3, total_shares=5)
# Need any 3 shares to reconstruct
reconstructed = ThresholdCrypto.reconstruct_secret(shares[:3])

# Encryption manager
manager = EncryptionManager()
key = manager.generate_key("data-key-1", "AES-GCM")
encrypted = manager.encrypt_data(b"genomic data", "data-key-1")
decrypted = manager.decrypt_data(encrypted)
```

## Security Considerations

1. **Key Storage**: In production, use Hardware Security Modules (HSM) for key storage
2. **Secrets Management**: Never commit secrets to version control
3. **Audit Logging**: Ensure audit logs are tamper-proof and regularly backed up
4. **Privacy Compliance**: Configure privacy filters appropriately for your jurisdiction

## Configuration File Format

Example configuration file (`~/.genomevault/config/production.yaml`):

```yaml
crypto:
  aes_key_size: 256
  hypervector_dimensions: 10000
  pir_server_count: 5
  pir_threshold: 3

privacy:
  epsilon: 1.0
  delta: 0.000001
  max_queries_per_user: 1000

network:
  api_port: 8080
  node_class: "full"
  signatory_status: true
  hipaa_verified: true

storage:
  compression_tier: "clinical"
  data_dir: "/var/genomevault/data"

processing:
  max_cores: 16
  max_memory_gb: 64
  gpu_acceleration: true
```

## Environment Variables

Configuration can be overridden using environment variables:

- `GENOMEVAULT_ENV`: Set environment (development/staging/production)
- `GENOMEVAULT_ZK_SECURITY`: Zero-knowledge security parameter
- `GENOMEVAULT_DP_EPSILON`: Differential privacy epsilon
- `GENOMEVAULT_API_PORT`: API server port
- `GENOMEVAULT_DATA_DIR`: Data storage directory
- `GENOMEVAULT_MAX_CORES`: Maximum CPU cores to use
- `GENOMEVAULT_MASTER_PASSWORD`: Master password for secrets encryption
