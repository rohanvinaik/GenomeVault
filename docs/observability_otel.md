# OpenTelemetry (Optional)

If you want **distributed tracing**, enable OpenTelemetry with OTLP export.

## Install deps
We keep OTEL deps separate:
```bash
pip install -r requirements-otel.txt
```

## Enable at runtime
Set environment variable:
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318/v1/traces"
```
The app auto-enables OTEL if the env var is set and packages are present.

## Notes
- This uses HTTP OTLP exporter by default.
- For full setup, run an OTEL collector to receive and forward traces to Jaeger/Tempo.
