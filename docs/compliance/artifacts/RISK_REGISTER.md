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
