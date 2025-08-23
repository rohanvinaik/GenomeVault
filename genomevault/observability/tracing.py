"""
OpenTelemetry distributed tracing for GenomeVault.

Implements end-to-end tracing across API, PIR, and ZK proof chains
with detailed span attributes and context propagation.
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any, Callable
from functools import wraps
import json

from opentelemetry import trace, baggage, context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.zipkin.json import ZipkinExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient, GrpcInstrumentorServer
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode, SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


# Configuration from environment
ENABLE_TRACING = os.getenv("ENABLE_TRACING", "true").lower() == "true"
TRACING_BACKEND = os.getenv("TRACING_BACKEND", "otlp")  # otlp, jaeger, zipkin, console
OTLP_ENDPOINT = os.getenv("OTLP_ENDPOINT", "localhost:4317")
JAEGER_ENDPOINT = os.getenv("JAEGER_ENDPOINT", "localhost:14268")
ZIPKIN_ENDPOINT = os.getenv("ZIPKIN_ENDPOINT", "http://localhost:9411/api/v2/spans")
SERVICE_NAME_ENV = os.getenv("SERVICE_NAME", "genomevault")
SERVICE_VERSION_ENV = os.getenv("SERVICE_VERSION", "1.0.0")


class SpanType(Enum):
    """Types of spans."""

    API = "api"
    PIR_QUERY = "pir_query"
    PIR_RESPONSE = "pir_response"
    ZK_PROOF = "zk_proof"
    HV_ENCODING = "hv_encoding"
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL = "external"


@dataclass
class TracingConfig:
    """Tracing configuration."""

    enabled: bool = ENABLE_TRACING
    backend: str = TRACING_BACKEND
    service_name: str = SERVICE_NAME_ENV
    service_version: str = SERVICE_VERSION_ENV
    sample_rate: float = 1.0
    max_attributes: int = 128
    max_events: int = 128
    max_links: int = 128
    max_attribute_length: int = 1024


class TracingManager:
    """
    Manager for distributed tracing.

    Handles trace initialization, span creation, and context propagation
    across the entire GenomeVault system.
    """

    def __init__(self, config: Optional[TracingConfig] = None):
        """
        Initialize tracing manager.

        Args:
            config: Tracing configuration
        """
        self.config = config or TracingConfig()
        self.tracer_provider: Optional[TracerProvider] = None
        self.tracer: Optional[trace.Tracer] = None

        if self.config.enabled:
            self._initialize_tracing()

    def _initialize_tracing(self) -> None:
        """Initialize OpenTelemetry tracing."""
        # Create resource
        resource = Resource.create(
            {
                SERVICE_NAME: self.config.service_name,
                SERVICE_VERSION: self.config.service_version,
                "deployment.environment": os.getenv("GENOMEVAULT_ENV", "development"),
                "host.name": os.getenv("HOSTNAME", "unknown"),
            }
        )

        # Create tracer provider
        self.tracer_provider = TracerProvider(
            resource=resource, active_span_processor=self._create_span_processor()
        )

        # Set global tracer provider
        trace.set_tracer_provider(self.tracer_provider)

        # Get tracer
        self.tracer = trace.get_tracer(
            instrumenting_module_name=__name__, instrumenting_library_version="1.0.0"
        )

        # Set propagator
        set_global_textmap(TraceContextTextMapPropagator())

        # Auto-instrument libraries
        self._instrument_libraries()

    def _create_span_processor(self) -> SpanProcessor:
        """Create span processor based on backend."""
        if self.config.backend == "otlp":
            exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)
        elif self.config.backend == "jaeger":
            exporter = JaegerExporter(
                agent_host_name=JAEGER_ENDPOINT.split(":")[0],
                agent_port=int(JAEGER_ENDPOINT.split(":")[1]) if ":" in JAEGER_ENDPOINT else 14268,
            )
        elif self.config.backend == "zipkin":
            exporter = ZipkinExporter(endpoint=ZIPKIN_ENDPOINT)
        else:
            exporter = ConsoleSpanExporter()

        return BatchSpanProcessor(exporter)

    def _instrument_libraries(self) -> None:
        """Auto-instrument common libraries."""
        try:
            # FastAPI
            FastAPIInstrumentor.instrument(tracer_provider=self.tracer_provider)

            # Requests
            RequestsInstrumentor().instrument(tracer_provider=self.tracer_provider)

            # SQLAlchemy
            SQLAlchemyInstrumentor().instrument(tracer_provider=self.tracer_provider)

            # Redis
            RedisInstrumentor().instrument(tracer_provider=self.tracer_provider)

            # gRPC
            GrpcInstrumentorClient().instrument(tracer_provider=self.tracer_provider)
            GrpcInstrumentorServer().instrument(tracer_provider=self.tracer_provider)
        except Exception as e:
            # Don't fail if instrumentation fails
            import logging

            logging.warning(f"Failed to instrument some libraries: {e}")

    @contextmanager
    def span(
        self,
        name: str,
        span_type: SpanType = SpanType.API,
        attributes: Optional[Dict[str, Any]] = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ):
        """
        Create a new span.

        Args:
            name: Span name
            span_type: Type of span
            attributes: Span attributes
            kind: Span kind
        """
        if not self.config.enabled or not self.tracer:
            yield None
            return

        # Start span
        with self.tracer.start_as_current_span(
            name=name, kind=kind, attributes=self._prepare_attributes(span_type, attributes)
        ) as span:
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def _prepare_attributes(
        self, span_type: SpanType, attributes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare span attributes.

        Args:
            span_type: Type of span
            attributes: Custom attributes

        Returns:
            Prepared attributes
        """
        base_attrs = {
            "span.type": span_type.value,
            "service.name": self.config.service_name,
            "service.version": self.config.service_version,
        }

        if attributes:
            # Filter and sanitize attributes
            for key, value in attributes.items():
                # Convert complex types to strings
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)[: self.config.max_attribute_length]
                elif not isinstance(value, (str, int, float, bool)):
                    value = str(value)[: self.config.max_attribute_length]

                # Skip PHI attributes
                if not self._is_phi_attribute(key):
                    base_attrs[key] = value

        return base_attrs

    def _is_phi_attribute(self, key: str) -> bool:
        """
        Check if attribute contains PHI.

        Args:
            key: Attribute key

        Returns:
            True if PHI attribute
        """
        phi_keywords = [
            "ssn",
            "social_security",
            "patient_id",
            "mrn",
            "email",
            "phone",
            "address",
            "dob",
            "date_of_birth",
            "genomic_data",
            "variant",
            "diagnosis",
            "medication",
        ]

        key_lower = key.lower()
        return any(keyword in key_lower for keyword in phi_keywords)

    def trace_pir_query(self, server_id: int, query_id: str, query_type: str) -> Any:
        """
        Trace PIR query.

        Args:
            server_id: Server ID
            query_id: Query ID
            query_type: Query type
        """
        return self.span(
            name=f"pir_query_{server_id}",
            span_type=SpanType.PIR_QUERY,
            attributes={
                "pir.server_id": server_id,
                "pir.query_id": query_id,
                "pir.query_type": query_type,
            },
            kind=SpanKind.CLIENT,
        )

    def trace_pir_response(self, server_id: int, query_id: str, response_size: int) -> Any:
        """
        Trace PIR response.

        Args:
            server_id: Server ID
            query_id: Query ID
            response_size: Response size
        """
        return self.span(
            name=f"pir_response_{server_id}",
            span_type=SpanType.PIR_RESPONSE,
            attributes={
                "pir.server_id": server_id,
                "pir.query_id": query_id,
                "pir.response_size": response_size,
            },
            kind=SpanKind.SERVER,
        )

    def trace_zk_proof(self, proof_type: str, circuit_size: int, operation: str) -> Any:
        """
        Trace ZK proof operation.

        Args:
            proof_type: Type of proof
            circuit_size: Circuit size
            operation: Operation (generate/verify)
        """
        return self.span(
            name=f"zk_{operation}_{proof_type}",
            span_type=SpanType.ZK_PROOF,
            attributes={
                "zk.proof_type": proof_type,
                "zk.circuit_size": circuit_size,
                "zk.operation": operation,
            },
            kind=SpanKind.INTERNAL,
        )

    def trace_hv_encoding(self, dimension: int, data_type: str, input_size: int) -> Any:
        """
        Trace hypervector encoding.

        Args:
            dimension: Vector dimension
            data_type: Data type
            input_size: Input size
        """
        return self.span(
            name=f"hv_encode_{data_type}",
            span_type=SpanType.HV_ENCODING,
            attributes={
                "hv.dimension": dimension,
                "hv.data_type": data_type,
                "hv.input_size": input_size,
            },
            kind=SpanKind.INTERNAL,
        )

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        Add event to current span.

        Args:
            name: Event name
            attributes: Event attributes
        """
        span = trace.get_current_span()
        if span and span.is_recording():
            span.add_event(name=name, attributes=self._prepare_attributes(SpanType.API, attributes))

    def set_attribute(self, key: str, value: Any) -> None:
        """
        Set attribute on current span.

        Args:
            key: Attribute key
            value: Attribute value
        """
        if self._is_phi_attribute(key):
            return  # Skip PHI attributes

        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute(key, value)

    def get_trace_id(self) -> Optional[str]:
        """
        Get current trace ID.

        Returns:
            Trace ID if available
        """
        span = trace.get_current_span()
        if span and span.is_recording():
            span_context = span.get_span_context()
            return format(span_context.trace_id, "032x")
        return None

    def get_span_id(self) -> Optional[str]:
        """
        Get current span ID.

        Returns:
            Span ID if available
        """
        span = trace.get_current_span()
        if span and span.is_recording():
            span_context = span.get_span_context()
            return format(span_context.span_id, "016x")
        return None

    def inject_context(self, carrier: Dict[str, str]) -> None:
        """
        Inject trace context into carrier.

        Args:
            carrier: Carrier dictionary (e.g., HTTP headers)
        """
        from opentelemetry.propagate import inject

        inject(carrier)

    def extract_context(self, carrier: Dict[str, str]) -> context.Context:
        """
        Extract trace context from carrier.

        Args:
            carrier: Carrier dictionary

        Returns:
            Extracted context
        """
        from opentelemetry.propagate import extract

        return extract(carrier)


# Global tracing manager
tracing_manager = TracingManager()


# =============================================================================
# Decorators
# =============================================================================


def trace_function(span_type: SpanType = SpanType.API, name: Optional[str] = None):
    """
    Decorator to trace function execution.

    Args:
        span_type: Type of span
        name: Custom span name
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not tracing_manager.config.enabled:
                return func(*args, **kwargs)

            span_name = name or f"{func.__module__}.{func.__name__}"

            with tracing_manager.span(
                name=span_name,
                span_type=span_type,
                attributes={
                    "function.module": func.__module__,
                    "function.name": func.__name__,
                },
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def trace_async_function(span_type: SpanType = SpanType.API, name: Optional[str] = None):
    """
    Decorator to trace async function execution.

    Args:
        span_type: Type of span
        name: Custom span name
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not tracing_manager.config.enabled:
                return await func(*args, **kwargs)

            span_name = name or f"{func.__module__}.{func.__name__}"

            with tracing_manager.span(
                name=span_name,
                span_type=span_type,
                attributes={
                    "function.module": func.__module__,
                    "function.name": func.__name__,
                },
            ):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Context Propagation
# =============================================================================


class TraceContext:
    """
    Helper for trace context management.
    """

    @staticmethod
    def get_current_trace_id() -> Optional[str]:
        """Get current trace ID."""
        return tracing_manager.get_trace_id()

    @staticmethod
    def get_current_span_id() -> Optional[str]:
        """Get current span ID."""
        return tracing_manager.get_span_id()

    @staticmethod
    def create_child_span(name: str, **kwargs) -> Any:
        """
        Create child span.

        Args:
            name: Span name
            **kwargs: Additional span arguments
        """
        return tracing_manager.span(name, **kwargs)

    @staticmethod
    def add_baggage(key: str, value: str) -> None:
        """
        Add baggage item.

        Args:
            key: Baggage key
            value: Baggage value
        """
        ctx = baggage.set_baggage(key, value)
        context.attach(ctx)

    @staticmethod
    def get_baggage(key: str) -> Optional[str]:
        """
        Get baggage item.

        Args:
            key: Baggage key

        Returns:
            Baggage value if exists
        """
        return baggage.get_baggage(key)


# =============================================================================
# Export
# =============================================================================

__all__ = [
    "TracingManager",
    "tracing_manager",
    "TracingConfig",
    "SpanType",
    "trace_function",
    "trace_async_function",
    "TraceContext",
    "ENABLE_TRACING",
]
