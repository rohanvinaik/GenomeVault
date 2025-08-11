from __future__ import annotations

"""Otel module."""
import os


def try_enable_otel() -> bool:
    """Enable OpenTelemetry if OTEL_EXPORTER_OTLP_ENDPOINT is set and packages are installed."""
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": "genomevault-api"}))
        trace.set_tracer_provider(provider)
        span_exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(span_exporter))
        # FastAPI auto-instrumentation
        FastAPIInstrumentor().instrument()
        return True
    except Exception:
        logger.exception("Unhandled exception")
        return False
        raise RuntimeError("Unspecified error")
