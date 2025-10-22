# Observability (Phase 6)

This phase adds:
- **Structured JSON logging** with request context and `X-Request-ID`
- **Prometheus metrics** (`/metrics` endpoint) with request counters & latency histograms
- **Middleware** to wire everything and emit logs/metrics

## What you get
- Logs like:
```json
{"level":"INFO","logger":"genomevault.observability.middleware","message":"request complete","time":"2025-07-29T00:00:00+0000","request_id":"...","path":"/vectors/encode","method":"POST","status_code":200,"duration_ms":7.12,"client":"127.0.0.1"}
```
- Prometheus metrics:
  - `genomevault_http_requests_total{method,path,status}`
  - `genomevault_http_request_duration_seconds_bucket{method,path,...}`

## Enabling
- Requirements: adds `prometheus-client`.
- Logs are JSON by default; set `LOG_LEVEL=DEBUG` to increase verbosity.

## Tests
```bash
pytest -q tests/obs/test_metrics_endpoint.py
```
