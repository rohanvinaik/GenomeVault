# Federated + Ledger (Phase 3) — Notes & Hardening Checklist

This implementation provides minimal scaffolds to enable end-to-end flow and API surface.

## Federated
- **FedAvgAggregator** with optional **per-update L2 clipping**.
- **Not included (out of scope for scaffold):**
  - Secure aggregation (e.g., additive masking w/ pairwise masks)
  - Byzantine robustness (e.g., Krum/Multi-Krum, coordinate-wise median, trimmed mean)
  - Differential privacy (e.g., per-update clipping + Gaussian noise w/ accounting)
  - Client attestation & signature verification

### Recommended Hardening
- Use **secure aggregation** library/protocol to hide individual updates.
- Add **robust aggregation** (Multi-Krum) for outlier/malicious updates.
- Integrate **DP-SGD** and account ε using RDP accountant.
- Require **client signatures** and verify **model shape** server-side.
- Add **versioning** for model/optimizer state.

## Ledger
- In-memory, hash-chained **append-only log** (not a consensus chain).
- Use it to track model versions, proof references, data-policy events, etc.

### Recommended Hardening
- Move to **content-addressed storage** (e.g., IPFS) for blobs; store CIDs in ledger.
- Back ledger by **SQLite/Postgres** with integrity constraints.
- Add **signatures** on entries and enforce **authorization** on API.
- Consider **consensus** or append-only, tamper-evident log (e.g., Trillian/Merkle).

## Endpoints
- `/federated/aggregate` — submit client updates and get aggregated weights
- `/ledger/append` | `/ledger/verify` | `/ledger/entries`

## Testing
Run focused tests:
```bash
pytest -q tests/federated/test_aggregator.py
pytest -q tests/api/test_federated_endpoints.py
pytest -q tests/ledger/test_ledger_store.py
pytest -q tests/api/test_ledger_endpoints.py
```
