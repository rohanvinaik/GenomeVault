"""
Logging middleware for GenomeVault API Gateway.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from genomevault.observability.logging import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive logging middleware for API requests and responses.
    
    Provides:
    - Request/response logging
    - Performance metrics
    - Error tracking
    - Security event logging
    - Audit trails
    """
    
    def __init__(
        self, 
        app,
        log_requests: bool = True,
        log_responses: bool = True,
        log_request_body: bool = False,
        log_response_body: bool = False,
        sensitive_headers: Optional[set] = None
    ):
        """
        Initialize logging middleware.
        
        Args:
            app: FastAPI application instance
            log_requests: Whether to log incoming requests
            log_responses: Whether to log outgoing responses
            log_request_body: Whether to log request body (be careful with sensitive data)
            log_response_body: Whether to log response body (be careful with sensitive data)
            sensitive_headers: Set of header names to mask in logs
        """
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        
        # Default sensitive headers to mask
        self.sensitive_headers = sensitive_headers or {
            "authorization",
            "x-api-key",
            "cookie",
            "x-auth-token",
            "x-session-id"
        }
        
        # Paths to exclude from detailed logging
        self.exclude_paths = {
            "/health",
            "/health/liveness", 
            "/health/readiness",
            "/metrics",
            "/favicon.ico"
        }
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with comprehensive logging.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or route handler
            
        Returns:
            HTTP response
        """
        # Generate request ID if not already present
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.perf_counter()
        
        # Skip detailed logging for excluded paths
        skip_detailed_logging = request.url.path in self.exclude_paths
        
        # Log incoming request
        if self.log_requests and not skip_detailed_logging:
            await self._log_request(request, request_id)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.perf_counter() - start_time
            
            # Log response
            if self.log_responses and not skip_detailed_logging:
                await self._log_response(request, response, request_id, process_time)
            
            # Log performance metrics
            await self._log_performance_metrics(request, response, process_time, request_id)
            
            return response
            
        except Exception as exc:
            # Log error
            process_time = time.perf_counter() - start_time
            await self._log_error(request, exc, request_id, process_time)
            raise
    
    async def _log_request(self, request: Request, request_id: str):
        """
        Log incoming request details.
        
        Args:
            request: HTTP request
            request_id: Request identifier
        """
        # Extract client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Extract user context if available
        user_id = getattr(request.state, "user_id", None)
        auth_method = getattr(request.state, "auth_method", None)
        
        # Prepare request headers (mask sensitive ones)
        headers = self._mask_sensitive_headers(dict(request.headers))
        
        # Prepare log data
        log_data = {
            "event": "request_started",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.url.query) if request.url.query else None,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "user_id": user_id,
            "auth_method": auth_method,
            "headers": headers,
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length")
        }
        
        # Add request body if configured and safe to do so
        if self.log_request_body and await self._is_safe_to_log_body(request):
            try:
                # Note: This consumes the request body, so we need to be careful
                # In a production system, you might want to use a different approach
                pass  # Skip body logging for now to avoid consuming the stream
            except Exception as e:
                log_data["body_read_error"] = str(e)
        
        # Log the request
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra=log_data
        )
    
    async def _log_response(
        self, 
        request: Request, 
        response: Response, 
        request_id: str, 
        process_time: float
    ):
        """
        Log outgoing response details.
        
        Args:
            request: HTTP request
            response: HTTP response
            request_id: Request identifier
            process_time: Processing time in seconds
        """
        # Prepare response headers (mask sensitive ones)
        headers = self._mask_sensitive_headers(dict(response.headers))
        
        # Prepare log data
        log_data = {
            "event": "request_completed",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time_ms": round(process_time * 1000, 2),
            "response_headers": headers,
            "content_type": response.headers.get("content-type"),
            "content_length": response.headers.get("content-length")
        }
        
        # Add user context if available
        if hasattr(request.state, "user_id"):
            log_data["user_id"] = request.state.user_id
        
        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = "error"
        elif response.status_code >= 400:
            log_level = "warning"
        else:
            log_level = "info"
        
        # Log the response
        getattr(logger, log_level)(
            f"Request completed: {request.method} {request.url.path} -> {response.status_code}",
            extra=log_data
        )
    
    async def _log_performance_metrics(
        self,
        request: Request,
        response: Response, 
        process_time: float,
        request_id: str
    ):
        """
        Log performance metrics.
        
        Args:
            request: HTTP request
            response: HTTP response
            process_time: Processing time in seconds
            request_id: Request identifier
        """
        # Prepare metrics data
        metrics_data = {
            "event": "performance_metrics",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time_ms": round(process_time * 1000, 2),
            "endpoint": self._get_endpoint_pattern(request.url.path)
        }
        
        # Add user context for performance analysis
        if hasattr(request.state, "user_id"):
            metrics_data["user_id"] = request.state.user_id
        
        if hasattr(request.state, "rate_limit_tier"):
            metrics_data["rate_limit_tier"] = request.state.rate_limit_tier
        
        # Log slow requests as warnings
        if process_time > 5.0:  # More than 5 seconds
            logger.warning(
                f"Slow request detected: {process_time:.2f}s",
                extra=metrics_data
            )
        else:
            logger.debug("Performance metrics", extra=metrics_data)
    
    async def _log_error(
        self,
        request: Request,
        error: Exception,
        request_id: str,
        process_time: float
    ):
        """
        Log request processing errors.
        
        Args:
            request: HTTP request
            error: Exception that occurred
            request_id: Request identifier
            process_time: Processing time before error
        """
        error_data = {
            "event": "request_error",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "process_time_ms": round(process_time * 1000, 2)
        }
        
        # Add user context if available
        if hasattr(request.state, "user_id"):
            error_data["user_id"] = request.state.user_id
        
        logger.error(
            f"Request error: {type(error).__name__}: {str(error)}",
            extra=error_data,
            exc_info=True
        )
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address, considering proxy headers.
        
        Args:
            request: HTTP request
            
        Returns:
            Client IP address
        """
        # Check for forwarded IP headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _mask_sensitive_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Mask sensitive headers for logging.
        
        Args:
            headers: Original headers dictionary
            
        Returns:
            Headers with sensitive values masked
        """
        masked_headers = {}
        for key, value in headers.items():
            if key.lower() in self.sensitive_headers:
                # Mask the value but keep some characters for debugging
                if len(value) > 8:
                    masked_headers[key] = value[:4] + "*" * (len(value) - 8) + value[-4:]
                else:
                    masked_headers[key] = "*" * len(value)
            else:
                masked_headers[key] = value
        
        return masked_headers
    
    async def _is_safe_to_log_body(self, request: Request) -> bool:
        """
        Check if it's safe to log the request body.
        
        Args:
            request: HTTP request
            
        Returns:
            True if safe to log body, False otherwise
        """
        # Don't log bodies for sensitive endpoints
        sensitive_paths = {"/auth", "/login", "/register", "/password"}
        if any(request.url.path.startswith(path) for path in sensitive_paths):
            return False
        
        # Don't log large bodies
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 1024:  # 1KB limit
            return False
        
        # Don't log binary content
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith(("application/json", "application/x-www-form-urlencoded", "text/")):
            return False
        
        return True
    
    def _get_endpoint_pattern(self, path: str) -> str:
        """
        Get normalized endpoint pattern for metrics aggregation.
        
        Args:
            path: Request path
            
        Returns:
            Normalized endpoint pattern
        """
        # Replace path parameters with placeholders for better aggregation
        # This is a simple implementation - you might want more sophisticated pattern matching
        
        import re
        
        # Replace UUIDs with placeholder
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{uuid}', path, flags=re.IGNORECASE)
        
        # Replace numeric IDs with placeholder
        path = re.sub(r'/\d+', '/{id}', path)
        
        # Replace other common patterns
        path = re.sub(r'/[a-zA-Z0-9]{20,}', '/{hash}', path)
        
        return path