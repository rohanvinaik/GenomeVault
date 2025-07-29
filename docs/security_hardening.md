# Security Hardening (Phase 7)

## What was added
- **Security headers** and **CORS** helper
- **MAX_BODY_SIZE** enforcement (default 10 MiB)
- **API key auth** (`X-API-Key`, keys from `GV_API_KEYS`)
- **Rate limiting** (in-memory token bucket)
- **SBOM generation** (CycloneDX) + CI artifact
- **SAST** (Bandit), **Dependency audit** (pip-audit)
- **Container scan** (Trivy), **DAST** (OWASP ZAP baseline)

## Configuration
- `.env` variables:
  - `GV_API_KEYS=devkey1,devkey2`
  - `MAX_BODY_SIZE=10485760`
  - `RATE_LIMIT_RPS=5.0`
  - `RATE_LIMIT_BURST=10`

## Production notes
- Replace in-memory rate limiting with Redis/NGINX/Gateway.
- Rotate API keys and migrate to OIDC/JWT if needed.
- Enable HSTS only when serving over HTTPS.
- Review ZAP rules allowlist under `.zap/rules.tsv` (create as needed).
