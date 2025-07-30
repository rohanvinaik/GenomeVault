# Data Governance & PII (Phase 8)

This phase adds:
- **PII detection** (email, phone, IPv4, SSN) and **redaction/tokenization**
- **Policy-based classification** via `etc/policies/classification.json`
- **Consent management** (in-memory) and **audit events** (backed by ledger)
- **Access guards** (FastAPI dependencies)
- **Governance API**: consent grant/revoke/check, DSAR export/erase, ROPA
- **CLI**: `scripts/pii_scan.py`

## Configuration
- `GV_PII_SECRET` — HMAC secret for deterministic tokens (set in prod!)
- `GV_PSEUDONYM_STORE` — optional JSON path to persist token→original (sensitive!)
- `GV_POLICY_PATH` — path to classification policy JSON (default `etc/policies/classification.json`)

## Example
```python
from genomevault.governance.pii.redact import tokenize_text
tok = tokenize_text("Email john@example.com")
```

## Warning
- The provided SSN regex is for **US only**; adjust or remove based on your use case.
- Persisting original values in a pseudonym store is **sensitive**; prefer non-reversible tokens in production.
- Replace the in-memory ConsentStore with a DB-backed implementation for real systems.
