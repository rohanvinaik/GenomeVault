"""
OpenTelemetry tracing for GenomeVault.

Comprehensive distributed tracing for privacy-preserving genomic operations.
Includes automatic instrumentation for FastAPI, SQLAlchemy, Redis, and custom
business logic spans for HDC, ZK proofs, and PIR operations.
"""

import os
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
from contextlib import contextmanager

try:
    from opentelemetry import trace, baggage
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.semconv.trace import SpanAttributes
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.util.http import get_excluded_urls

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

    # Create mock classes when OpenTelemetry is not available
    class MockTrace:
        class Tracer:
            def start_as_current_span(self, name):
                return MockSpan()

        class Status:
            def __init__(self, code, message=""):
                pass

        class StatusCode:
            OK = "OK"
            ERROR = "ERROR"

    class MockSpan:
        def __init__(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_attribute(self, key, value):
            pass

        def record_exception(self, exc):
            pass

        def set_status(self, status):
            pass

        def is_recording(self):
            return False

        def get_span_context(self):
            return MockSpanContext()

    class MockSpanContext:
        def __init__(self):
            self.is_valid = False

    trace = MockTrace()
    Status = MockTrace.Status
    StatusCode = MockTrace.StatusCode

logger = logging.getLogger(__name__)

# Service information
SERVICE_NAME = "genomevault"
SERVICE_VERSION = "1.0.0"
SERVICE_NAMESPACE = "genomics"


# Custom span attributes for GenomeVault
class GenomeVaultAttributes:
    """Custom attributes for GenomeVault spans."""

    # HDC attributes
    HDC_DIMENSION = "genomevault.hdc.dimension"
    HDC_VARIANT_COUNT = "genomevault.hdc.variant_count"
    HDC_ENCODING_ID = "genomevault.hdc.encoding_id"
    HDC_METRIC_TYPE = "genomevault.hdc.metric_type"
    HDC_SIMILARITY_SCORE = "genomevault.hdc.similarity_score"

    # ZK Proof attributes
    ZK_CIRCUIT_TYPE = "genomevault.zk.circuit_type"
    ZK_PROOF_ID = "genomevault.zk.proof_id"
    ZK_PUBLIC_INPUTS_COUNT = "genomevault.zk.public_inputs_count"
    ZK_PRIVATE_INPUTS_COUNT = "genomevault.zk.private_inputs_count"
    ZK_FALLBACK_USED = "genomevault.zk.fallback_used"
    ZK_VERIFICATION_RESULT = "genomevault.zk.verification_result"

    # PIR attributes
    PIR_DATABASE_SIZE = "genomevault.pir.database_size"
    PIR_QUERY_INDEX = "genomevault.pir.query_index"
    PIR_SERVERS_COUNT = "genomevault.pir.servers_count"
    PIR_BYZANTINE_DETECTED = "genomevault.pir.byzantine_detected"
    PIR_SETUP_ID = "genomevault.pir.setup_id"

    # Privacy attributes
    PRIVACY_LEVEL = "genomevault.privacy.level"
    PRIVACY_TECHNIQUE = "genomevault.privacy.technique"
    PHI_PRESENT = "genomevault.privacy.phi_present"

    # Genomic data attributes
    GENOMIC_VARIANT_TYPE = "genomevault.genomic.variant_type"
    GENOMIC_CHROMOSOME = "genomevault.genomic.chromosome"
    GENOMIC_POSITION = "genomevault.genomic.position"
    GENOMIC_SAMPLE_ID = "genomevault.genomic.sample_id"


class TracingManager:
    """Manages OpenTelemetry tracing for GenomeVault."""

    def __init__(
        self,
        service_name: str = SERVICE_NAME,
        service_version: str = SERVICE_VERSION,
        jaeger_endpoint: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        sampling_rate: float = 1.0,
        enable_console_exporter: bool = False,
    ):
        """Initialize tracing manager.

        Args:
            service_name: Name of the service
            service_version: Version of the service
            jaeger_endpoint: Jaeger collector endpoint
            otlp_endpoint: OTLP exporter endpoint
            sampling_rate: Sampling rate (0.0 to 1.0)
            enable_console_exporter: Whether to enable console exporter
        """
        self.service_name = service_name
        self.service_version = service_version
        self.jaeger_endpoint = jaeger_endpoint or os.getenv("JAEGER_ENDPOINT")
        self.otlp_endpoint = otlp_endpoint or os.getenv("OTLP_ENDPOINT")
        self.sampling_rate = sampling_rate
        self.enable_console_exporter = enable_console_exporter

        self.tracer_provider: Optional[TracerProvider] = None
        self.tracer: Optional[trace.Tracer] = None

        if OTEL_AVAILABLE:
            self._setup_tracing()
        else:
            logger.warning("OpenTelemetry not available. Tracing disabled.")

    def _setup_tracing(self):
        """Set up OpenTelemetry tracing."""
        if not OTEL_AVAILABLE:
            return

        # Create resource
        resource = Resource.create(
            {
                ResourceAttributes.SERVICE_NAME: self.service_name,
                ResourceAttributes.SERVICE_VERSION: self.service_version,
                ResourceAttributes.SERVICE_NAMESPACE: SERVICE_NAMESPACE,
                "service.instance.id": os.getenv("HOSTNAME", "localhost"),
                "deployment.environment": os.getenv("ENVIRONMENT", "development"),
            }
        )

        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(self.tracer_provider)

        # Add exporters
        self._add_exporters()

        # Get tracer
        self.tracer = trace.get_tracer(__name__, self.service_version)

        # Set up auto-instrumentation
        self._setup_auto_instrumentation()

        logger.info(f"OpenTelemetry tracing initialized for {self.service_name}")

    def _add_exporters(self):
        """Add span exporters."""
        if not self.tracer_provider:
            return

        # Jaeger exporter
        if self.jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name=(
                    self.jaeger_endpoint.split(":")[0]
                    if ":" in self.jaeger_endpoint
                    else self.jaeger_endpoint
                ),
                agent_port=(
                    int(self.jaeger_endpoint.split(":")[1])
                    if ":" in self.jaeger_endpoint
                    else 14268
                ),
                collector_endpoint=(
                    f"http://{self.jaeger_endpoint}/api/traces"
                    if not self.jaeger_endpoint.startswith("http")
                    else self.jaeger_endpoint
                ),
            )
            self.tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
            logger.info(f"Added Jaeger exporter: {self.jaeger_endpoint}")

        # OTLP exporter
        if self.otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.otlp_endpoint,
                headers={"Authorization": f"Bearer {os.getenv('OTLP_TOKEN', '')}"},
            )
            self.tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"Added OTLP exporter: {self.otlp_endpoint}")

        # Console exporter for debugging
        if self.enable_console_exporter:
            from opentelemetry.exporter.console import ConsoleSpanExporter

            console_exporter = ConsoleSpanExporter()
            self.tracer_provider.add_span_processor(SimpleSpanProcessor(console_exporter))
            logger.info("Added console exporter for debugging")

    def _setup_auto_instrumentation(self):
        """Set up automatic instrumentation."""
        try:
            # FastAPI instrumentation
            FastAPIInstrumentor.instrument(
                excluded_urls=get_excluded_urls("OTEL_PYTHON_FASTAPI_EXCLUDED_URLS")
            )

            # SQLAlchemy instrumentation
            SQLAlchemyInstrumentor().instrument()

            # Redis instrumentation
            RedisInstrumentor().instrument()

            # HTTP client instrumentation
            RequestsInstrumentor().instrument()
            HTTPXClientInstrumentor().instrument()

            logger.info("Auto-instrumentation configured")

        except Exception as e:
            logger.warning(f"Failed to set up some auto-instrumentation: {e}")

    def get_tracer(self) -> Optional[Any]:
        """Get the tracer instance."""
        return self.tracer

    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None,
        privacy_level: str = "medium",
    ):
        """Context manager for tracing operations.

        Args:
            operation_name: Name of the operation
            attributes: Custom attributes to add to span
            privacy_level: Privacy level (low/medium/high)
        """
        if not self.tracer:
            yield None
            return

        with self.tracer.start_as_current_span(operation_name) as span:
            try:
                # Add common attributes
                span.set_attribute(GenomeVaultAttributes.PRIVACY_LEVEL, privacy_level)

                # Add custom attributes
                if attributes:
                    for key, value in attributes.items():
                        if value is not None:
                            span.set_attribute(key, str(value))

                yield span

                # Mark as successful
                span.set_status(Status(StatusCode.OK))

            except Exception as e:
                # Record error
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def trace_hdc_encoding(
        self, dimension: int, variant_count: int, encoding_id: Optional[str] = None
    ):
        """Decorator for tracing HDC encoding operations."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.trace_operation(
                    f"hdc.encoding.{func.__name__}",
                    attributes={
                        GenomeVaultAttributes.HDC_DIMENSION: dimension,
                        GenomeVaultAttributes.HDC_VARIANT_COUNT: variant_count,
                        GenomeVaultAttributes.HDC_ENCODING_ID: encoding_id,
                        GenomeVaultAttributes.PRIVACY_TECHNIQUE: "hyperdimensional_computing",
                    },
                    privacy_level="high",
                ):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def trace_zk_proof(
        self,
        circuit_type: str,
        proof_id: Optional[str] = None,
        public_inputs_count: Optional[int] = None,
    ):
        """Decorator for tracing ZK proof operations."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.trace_operation(
                    f"zk.proof.{func.__name__}",
                    attributes={
                        GenomeVaultAttributes.ZK_CIRCUIT_TYPE: circuit_type,
                        GenomeVaultAttributes.ZK_PROOF_ID: proof_id,
                        GenomeVaultAttributes.ZK_PUBLIC_INPUTS_COUNT: public_inputs_count,
                        GenomeVaultAttributes.PRIVACY_TECHNIQUE: "zero_knowledge_proofs",
                    },
                    privacy_level="high",
                ):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def trace_pir_query(
        self, database_size: int, query_index: Optional[int] = None, servers_count: int = 3
    ):
        """Decorator for tracing PIR query operations."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.trace_operation(
                    f"pir.query.{func.__name__}",
                    attributes={
                        GenomeVaultAttributes.PIR_DATABASE_SIZE: database_size,
                        GenomeVaultAttributes.PIR_QUERY_INDEX: query_index,
                        GenomeVaultAttributes.PIR_SERVERS_COUNT: servers_count,
                        GenomeVaultAttributes.PRIVACY_TECHNIQUE: "private_information_retrieval",
                    },
                    privacy_level="high",
                ):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def add_genomic_context(
        self,
        variant_type: Optional[str] = None,
        chromosome: Optional[str] = None,
        position: Optional[int] = None,
        sample_id: Optional[str] = None,
        phi_present: bool = True,
    ):
        """Add genomic context to current span."""
        if not self.tracer:
            return

        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            if variant_type:
                current_span.set_attribute(GenomeVaultAttributes.GENOMIC_VARIANT_TYPE, variant_type)
            if chromosome:
                current_span.set_attribute(GenomeVaultAttributes.GENOMIC_CHROMOSOME, chromosome)
            if position:
                current_span.set_attribute(GenomeVaultAttributes.GENOMIC_POSITION, position)
            if sample_id:
                current_span.set_attribute(GenomeVaultAttributes.GENOMIC_SAMPLE_ID, sample_id)

            current_span.set_attribute(GenomeVaultAttributes.PHI_PRESENT, phi_present)

    def add_baggage(self, key: str, value: str):
        """Add baggage to current context."""
        if OTEL_AVAILABLE:
            baggage.set_baggage(key, value)

    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage from current context."""
        if OTEL_AVAILABLE:
            return baggage.get_baggage(key)
        return None


# Global tracing manager instance
_tracing_manager: Optional[TracingManager] = None


def get_tracing_manager() -> Optional[TracingManager]:
    """Get or create the global tracing manager."""
    global _tracing_manager

    if not OTEL_AVAILABLE:
        return None

    if _tracing_manager is None:
        # Configure from environment variables
        jaeger_endpoint = os.getenv("JAEGER_ENDPOINT")
        otlp_endpoint = os.getenv("OTLP_ENDPOINT")
        sampling_rate = float(os.getenv("OTEL_SAMPLING_RATE", "1.0"))
        enable_console = os.getenv("OTEL_ENABLE_CONSOLE", "false").lower() == "true"

        # Only initialize if at least one exporter is configured
        if jaeger_endpoint or otlp_endpoint or enable_console:
            _tracing_manager = TracingManager(
                jaeger_endpoint=jaeger_endpoint,
                otlp_endpoint=otlp_endpoint,
                sampling_rate=sampling_rate,
                enable_console_exporter=enable_console,
            )
        else:
            logger.info("No tracing exporters configured. Tracing disabled.")

    return _tracing_manager


def get_tracer() -> Optional[Any]:
    """Get the OpenTelemetry tracer."""
    manager = get_tracing_manager()
    return manager.get_tracer() if manager else None


# Convenience decorators using the global tracing manager
def trace_hdc_operation(dimension: int, variant_count: int, encoding_id: Optional[str] = None):
    """Convenience decorator for HDC operations."""
    manager = get_tracing_manager()
    if manager:
        return manager.trace_hdc_encoding(dimension, variant_count, encoding_id)

    def no_op_decorator(func):
        return func

    return no_op_decorator


def trace_zk_operation(circuit_type: str, proof_id: Optional[str] = None):
    """Convenience decorator for ZK operations."""
    manager = get_tracing_manager()
    if manager:
        return manager.trace_zk_proof(circuit_type, proof_id)

    def no_op_decorator(func):
        return func

    return no_op_decorator


def trace_pir_operation(database_size: int, query_index: Optional[int] = None):
    """Convenience decorator for PIR operations."""
    manager = get_tracing_manager()
    if manager:
        return manager.trace_pir_query(database_size, query_index)

    def no_op_decorator(func):
        return func

    return no_op_decorator


@contextmanager
def trace_operation(operation_name: str, **attributes):
    """Convenience context manager for tracing operations."""
    manager = get_tracing_manager()
    if manager:
        with manager.trace_operation(operation_name, attributes):
            yield
    else:
        yield
