"""
Enhanced observability middleware for GenomeVault.

Combines structured logging, Prometheus metrics, OpenTelemetry tracing,
and performance timing in a single middleware component.
"""

import time
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from genomevault.observability.logging.structured import (
    get_structured_logger,
    set_request_context,
    generate_request_id,
)
from genomevault.observability.metrics.prometheus import get_metrics_collector
from genomevault.observability.tracing.opentelemetry import get_tracing_manager


class EnhancedObservabilityMiddleware(BaseHTTPMiddleware):
    """Enhanced middleware that provides comprehensive observability."""

    def __init__(self, app, service_name: str = "genomevault-api"):
        """Initialize middleware.

        Args:
            app: FastAPI application instance
            service_name: Service name for tracing and metrics
        """
        super().__init__(app)
        self.service_name = service_name
        self.logger = get_structured_logger(__name__)
        self.metrics_collector = get_metrics_collector()
        self.tracing_manager = get_tracing_manager()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request with comprehensive observability.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            HTTP response with observability headers
        """
        # Generate or extract request ID
        request_id = request.headers.get("x-request-id")
        if not request_id:
            request_id = generate_request_id()

        # Store request ID in request state for handlers
        request.state.request_id = request_id

        # Extract user context if available
        user_id = request.headers.get("x-user-id")
        client_ip = request.client.host if request.client else "unknown"

        # Set structured logging context
        set_request_context(request_id, user_id)

        # Get route information for metrics
        route = request.scope.get("route")
        endpoint_path = (
            getattr(route, "path_format", request.url.path) if route else request.url.path
        )

        # Start timing
        start_time = time.perf_counter()

        # Determine operation type for better tracing
        operation_name = f"{request.method} {endpoint_path}"
        privacy_level = self._determine_privacy_level(endpoint_path)

        # Set up tracing context
        trace_attributes = {
            "http.method": request.method,
            "http.url": str(request.url),
            "http.route": endpoint_path,
            "http.client_ip": client_ip,
            "http.user_agent": request.headers.get("user-agent", ""),
            "genomevault.request_id": request_id,
        }

        if user_id:
            trace_attributes["genomevault.user_id"] = user_id

        # Process request within tracing context
        if self.tracing_manager:
            with self.tracing_manager.trace_operation(
                operation_name, attributes=trace_attributes, privacy_level=privacy_level
            ) as span:
                return await self._process_request(
                    request,
                    call_next,
                    request_id,
                    user_id,
                    client_ip,
                    endpoint_path,
                    start_time,
                    span,
                )
        else:
            return await self._process_request(
                request, call_next, request_id, user_id, client_ip, endpoint_path, start_time, None
            )

    async def _process_request(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
        request_id: str,
        user_id: Optional[str],
        client_ip: str,
        endpoint_path: str,
        start_time: float,
        span=None,
    ) -> Response:
        """Process the actual request with error handling."""
        response = None
        status_code = 500  # Default to error
        error_occurred = False

        try:
            # Add request size information if available
            request_size = None
            if "content-length" in request.headers:
                try:
                    request_size = int(request.headers["content-length"])
                except (ValueError, TypeError):
                    pass

            # Log request start
            self.logger.info(
                f"Request started: {request.method} {request.url.path}",
                http_method=request.method,
                http_path=request.url.path,
                http_user_agent=request.headers.get("user-agent", ""),
                client_ip=client_ip,
                request_size=request_size,
            )

            # Process request
            response = await call_next(request)
            status_code = response.status_code

        except Exception as e:
            error_occurred = True
            status_code = 500

            # Record error in tracing
            if span:
                span.record_exception(e)

            # Log error
            self.logger.error(
                f"Request failed: {request.method} {request.url.path}",
                error_type=type(e).__name__,
                error_message=str(e),
                http_method=request.method,
                http_path=request.url.path,
                client_ip=client_ip,
            )

            # Record error metrics
            self.metrics_collector.record_error(
                error_type=type(e).__name__, component="middleware", severity="error"
            )

            # Re-raise the exception
            raise

        finally:
            # Calculate duration
            duration = time.perf_counter() - start_time

            # Get response size if available
            response_size = None
            if response and "content-length" in response.headers:
                try:
                    response_size = int(response.headers["content-length"])
                except (ValueError, TypeError):
                    pass

            # Record HTTP metrics
            self.metrics_collector.record_http_request(
                method=request.method,
                endpoint=endpoint_path,
                status_code=status_code,
                duration=duration,
                request_size=request_size,
                response_size=response_size,
                component="api",
            )

            # Log request completion
            if not error_occurred:
                self.logger.log_api_request(
                    method=request.method,
                    path=request.url.path,
                    status_code=status_code,
                    duration=duration,
                    client_ip=client_ip,
                    response_size=response_size,
                )

        # Add observability headers to response
        if response:
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration * 1000:.2f}ms"

            # Add trace ID if available
            if self.tracing_manager and span:
                span_context = span.get_span_context()
                if span_context.is_valid:
                    from opentelemetry.trace import format_trace_id

                    response.headers["X-Trace-ID"] = format_trace_id(span_context.trace_id)

        return response

    def _determine_privacy_level(self, path: str) -> str:
        """Determine privacy level based on endpoint path."""
        # High privacy endpoints
        if any(
            pattern in path.lower()
            for pattern in ["/zk/", "/pir/", "/clinical/", "/variants/", "/genomic/"]
        ):
            return "high"

        # Medium privacy endpoints
        if any(pattern in path.lower() for pattern in ["/hdc/", "/encode/", "/federated/"]):
            return "medium"

        # Low privacy endpoints (health, metrics, etc.)
        return "low"


class PerformanceTimingMiddleware(BaseHTTPMiddleware):
    """Lightweight middleware focused only on performance timing."""

    def __init__(self, app):
        """Initialize performance timing middleware."""
        super().__init__(app)
        self.metrics_collector = get_metrics_collector()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Add performance timing to requests."""
        start_time = time.perf_counter()

        # Process request
        response = await call_next(request)

        # Calculate and record timing
        duration = time.perf_counter() - start_time

        # Get route for metrics
        route = request.scope.get("route")
        endpoint_path = (
            getattr(route, "path_format", request.url.path) if route else request.url.path
        )

        # Record basic HTTP metrics
        self.metrics_collector.record_http_request(
            method=request.method,
            endpoint=endpoint_path,
            status_code=response.status_code,
            duration=duration,
            component="api",
        )

        # Add timing header
        response.headers["X-Response-Time"] = f"{duration * 1000:.2f}ms"

        return response


def add_enhanced_observability_middleware(app, service_name: str = "genomevault-api"):
    """Add enhanced observability middleware to FastAPI app.

    Args:
        app: FastAPI application instance
        service_name: Service name for observability
    """
    app.add_middleware(EnhancedObservabilityMiddleware, service_name=service_name)


def add_performance_timing_middleware(app):
    """Add lightweight performance timing middleware.

    Args:
        app: FastAPI application instance
    """
    app.add_middleware(PerformanceTimingMiddleware)
