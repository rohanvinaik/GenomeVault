# Key Management Policy

## Version 1.0 - DRAFT

### 1. Purpose
Define cryptographic key management for GenomeVault.

### 2. Scope
All cryptographic keys used for PHI protection.

### 3. Key Hierarchy

```
Master Key (HSM)
├── Key Encryption Keys (KEK)
│   ├── Data Encryption Keys (DEK)
│   └── Database Encryption Keys
└── Signing Keys
    ├── Audit Log Signing
    └── API Token Signing
```

### 4. Key Lifecycle

| Phase | Duration | Action |
|-------|----------|--------|
| Generation | Day 0 | Create in HSM/KMS |
| Distribution | Day 1-7 | Secure distribution |
| Usage | Day 8-90 | Active use |
| Rotation | Day 90 | Generate new key |
| Archive | Day 91-365 | Retain for decryption only |
| Destruction | Day 365+ | Secure deletion |

### 5. Key Storage

**Development**: 
- File-based with password protection
- Local key directory

**Production**:
- Hardware Security Module (HSM)
- Cloud KMS (AWS KMS, Azure Key Vault)
- FIPS 140-2 Level 2 minimum

### 6. Emergency Procedures

1. **Key Compromise**:
   - Immediate key rotation
   - Re-encrypt affected data
   - Audit log review
   - Incident report

2. **Key Recovery**:
   - Restore from secure backup
   - Verify integrity
   - Test decryption
   - Update key registry

### 7. Compliance

Meets requirements for:
- HIPAA (45 CFR § 164.312(a)(2)(iv))
- NIST SP 800-57
- FIPS 140-2
