# Key Management Policy

## Overview
This document outlines key management procedures for GenomeVault in production environments.

## Key Types

### 1. Master Keys
- **Purpose**: Encrypt data encryption keys (DEKs)
- **Rotation**: Every 90 days
- **Storage**: Hardware Security Module (HSM) or AWS KMS/Azure Key Vault

### 2. Data Encryption Keys (DEKs)
- **Purpose**: Encrypt genomic data
- **Rotation**: Per session or daily
- **Storage**: Encrypted with master key

### 3. ZK Transcript Keys
- **Purpose**: Zero-knowledge proof generation
- **Rotation**: Per proof or weekly
- **Storage**: Secure enclave

## Key Generation

```python
from genomevault.crypto import KeyManager

# Initialize key manager
km = KeyManager(hsm_enabled=True)

# Generate new master key
master_key = km.generate_master_key(
    algorithm="AES-256-GCM",
    purpose="data_encryption"
)

# Rotate keys
km.rotate_key(master_key.id, grace_period_days=7)
```

## Key Storage

### Development
```yaml
storage:
  type: filesystem
  path: ./keys/
  encryption: password
```

### Production
```yaml
storage:
  type: hsm
  provider: aws_kms  # or azure_key_vault, gcp_kms
  region: us-east-1
  key_id: arn:aws:kms:...
```

## Key Rotation Schedule

| Key Type | Rotation Period | Grace Period |
|----------|----------------|--------------|
| Master | 90 days | 7 days |
| DEK | 24 hours | 1 hour |
| ZK Transcript | 7 days | 1 day |
| TLS Certificates | 365 days | 30 days |

## Emergency Procedures

### Key Compromise
1. Immediately rotate affected keys
2. Re-encrypt all affected data
3. Audit access logs
4. Notify security team
5. Document incident

### Key Recovery
1. Restore from secure backup
2. Verify key integrity
3. Test decryption
4. Update key registry

## Compliance

This policy meets requirements for:
- HIPAA (45 CFR 164.312(a)(2)(iv))
- GDPR Article 32
- SOC 2 Type II
