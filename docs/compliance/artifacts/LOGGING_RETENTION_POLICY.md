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
