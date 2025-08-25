#!/bin/bash
set -e

echo "📋 Fixing compliance documentation"
echo "=================================="

# Rename misleading files
if [ -f "docs/HIPAA_IMPLEMENTATION_COMPLETE.md" ]; then
    echo "Renaming misleading HIPAA document..."
    mv docs/HIPAA_IMPLEMENTATION_COMPLETE.md docs/HIPAA_READINESS_PLAN.md
fi

# Create accurate compliance structure
mkdir -p docs/compliance/{plans,templates,artifacts}

# Create HIPAA Readiness Plan (not "complete")
cat > docs/compliance/plans/HIPAA_READINESS_PLAN.md << 'EOF'
# HIPAA Readiness Plan

## Current Status: Technical Architecture Ready

**Important**: GenomeVault provides HIPAA-compliant architecture but is **NOT** certified for production PHI handling.

## Completed Technical Components

### ✅ Security Architecture
- AES-256-GCM encryption at rest
- TLS 1.3 for data in transit
- Zero-knowledge proof system for privacy

### ✅ Access Control Framework
- Role-based access control (RBAC)
- API key management
- Session management

### ✅ Audit Logging System
- Comprehensive audit trails
- Tamper-evident logs
- 7-year retention capability

## Required Before PHI Processing

### 1. Legal Requirements
- [ ] Business Associate Agreements (BAAs)
- [ ] Privacy Officer designation
- [ ] Security Officer designation
- [ ] Workforce training program

### 2. Risk Assessment
- [ ] Formal risk assessment
- [ ] Vulnerability scanning
- [ ] Penetration testing
- [ ] Remediation plan

### 3. Policies & Procedures
- [ ] Information security policy
- [ ] Incident response plan
- [ ] Disaster recovery plan
- [ ] Employee training records

### 4. Technical Implementations
- [ ] Encryption key management with HSM
- [ ] Automated backup system
- [ ] Integrity monitoring
- [ ] Production audit logging

## Timeline to HIPAA Compliance

| Phase | Tasks | Duration | Status |
|-------|-------|----------|--------|
| 1. Assessment | Risk assessment, gap analysis | 2-4 weeks | Not started |
| 2. Remediation | Address identified gaps | 4-8 weeks | Not started |
| 3. Documentation | Policies, procedures, training | 2-4 weeks | Not started |
| 4. Testing | Security testing, audits | 2-3 weeks | Not started |
| 5. Certification | Third-party assessment | 2-4 weeks | Not started |

**Total estimated time**: 3-6 months

## Contact

For compliance consulting: [compliance@genomevault.example.com]
EOF

# Create BAA Template
cat > docs/compliance/templates/BAA_TEMPLATE.md << 'EOF'
# Business Associate Agreement Template

**DRAFT - Requires Legal Review**

This Business Associate Agreement ("Agreement") is entered into as of __________ ("Effective Date")

## Definitions

- **Protected Health Information (PHI)**: As defined in 45 CFR § 160.103
- **Business Associate**: [Your Organization]
- **Covered Entity**: [Healthcare Provider/Plan]

## Obligations of Business Associate

1. **Use and Disclosure**: Business Associate shall not use or disclose PHI except as permitted by this Agreement.

2. **Safeguards**: Business Associate shall use appropriate safeguards to prevent unauthorized use or disclosure.

3. **Reporting**: Business Associate shall report any breach within 72 hours.

4. **Subcontractors**: Business Associate shall ensure subcontractors agree to similar restrictions.

## Security Requirements

Business Associate shall implement:
- Administrative safeguards (45 CFR § 164.308)
- Physical safeguards (45 CFR § 164.310)
- Technical safeguards (45 CFR § 164.312)

## Term and Termination

This Agreement shall remain in effect until terminated.

---
**Note**: This is a template only. Consult legal counsel before use.
EOF

# Create Risk Register Template
cat > docs/compliance/artifacts/RISK_REGISTER.md << 'EOF'
# Risk Register

## Risk Assessment Matrix

| ID | Risk | Likelihood | Impact | Score | Mitigation | Status |
|----|------|------------|--------|-------|------------|--------|
| R001 | Unauthorized PHI access | Medium | High | 6 | RBAC implementation | In Progress |
| R002 | Data breach | Low | Critical | 8 | Encryption + monitoring | Implemented |
| R003 | Key compromise | Low | High | 4 | HSM integration | Planned |
| R004 | Audit log tampering | Low | High | 4 | Immutable logs | Implemented |
| R005 | Insider threat | Medium | High | 6 | Access monitoring | Planned |

## Risk Scoring

- **Likelihood**: Low (1), Medium (2), High (3)
- **Impact**: Low (1), Medium (2), High (3), Critical (4)
- **Score**: Likelihood × Impact

## Risk Treatment

| Score | Action Required |
|-------|----------------|
| 1-3 | Accept risk, monitor |
| 4-6 | Mitigate within 90 days |
| 7-9 | Mitigate within 30 days |
| 10+ | Immediate action required |

## Review Schedule

- Quarterly risk assessment review
- Annual comprehensive assessment
- Incident-triggered reviews
EOF

# Create Key Management Policy
cat > docs/compliance/artifacts/KEY_MANAGEMENT_POLICY.md << 'EOF'
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
EOF

# Create Logging and Retention Policy
cat > docs/compliance/artifacts/LOGGING_RETENTION_POLICY.md << 'EOF'
# Audit Logging and Retention Policy

## Logging Requirements

### Events to Log

**Required by HIPAA**:
- User authentication (success/failure)
- PHI access attempts
- Data modifications
- Administrative actions
- System events

### Log Format

```json
{
  "timestamp": "ISO-8601",
  "event_type": "string",
  "user_id": "string",
  "resource": "string",
  "action": "string",
  "result": "success|failure",
  "ip_address": "string",
  "session_id": "string",
  "details": {}
}
```

## Retention Periods

| Log Type | Retention | Justification |
|----------|-----------|---------------|
| Access logs | 7 years | HIPAA requirement |
| System logs | 1 year | Operational needs |
| Security events | 7 years | Compliance/forensics |
| Performance logs | 90 days | Capacity planning |

## Implementation

```python
from genomevault.compliance import AuditLogger

logger = AuditLogger(
    retention_days=2555,  # 7 years
    encryption=True,
    tamper_evident=True
)

logger.log_phi_access(
    user_id="usr_123",
    resource="patient_record",
    action="read",
    result="success"
)
```

## Storage and Protection

- Encrypted at rest (AES-256)
- Tamper-evident (hash chain)
- Replicated backup
- Immutable after write
- Regular integrity checks
EOF

echo "✅ Compliance documentation restructured"
echo ""
echo "Created structure:"
echo "  docs/compliance/"
echo "  ├── plans/"
echo "  │   └── HIPAA_READINESS_PLAN.md"
echo "  ├── templates/"
echo "  │   └── BAA_TEMPLATE.md"
echo "  └── artifacts/"
echo "      ├── RISK_REGISTER.md"
echo "      ├── KEY_MANAGEMENT_POLICY.md"
echo "      └── LOGGING_RETENTION_POLICY.md"
echo ""
echo "These documents are marked as DRAFT and require:"
echo "  1. Legal review"
echo "  2. Customization for your organization"
echo "  3. Formal approval process"