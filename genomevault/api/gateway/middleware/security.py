"""
Security middleware for GenomeVault API Gateway.
"""

from __future__ import annotations

import re
from typing import Set

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from genomevault.observability.logging import get_logger

logger = get_logger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware providing various security headers and protections.

    Features:
    - Security headers (HSTS, CSP, etc.)
    - Input sanitization
    - XSS protection
    - Request size limits
    - IP filtering
    """

    def __init__(
        self,
        app,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        enable_hsts: bool = True,
        enable_csp: bool = True,
        blocked_ips: Set[str] = None,
    ):
        """
        Initialize security middleware.

        Args:
            app: FastAPI application instance
            max_request_size: Maximum request size in bytes
            enable_hsts: Whether to enable HSTS header
            enable_csp: Whether to enable CSP header
            blocked_ips: Set of blocked IP addresses
        """
        super().__init__(app)
        self.max_request_size = max_request_size
        self.enable_hsts = enable_hsts
        self.enable_csp = enable_csp
        self.blocked_ips = blocked_ips or set()

        # XSS patterns to detect in inputs
        self.xss_patterns = [
            re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
            re.compile(r"javascript:", re.IGNORECASE),
            re.compile(r"on\w+\s*=", re.IGNORECASE),
            re.compile(r"<iframe[^>]*>.*?</iframe>", re.IGNORECASE | re.DOTALL),
            re.compile(r"<object[^>]*>.*?</object>", re.IGNORECASE | re.DOTALL),
            re.compile(r"<embed[^>]*>", re.IGNORECASE),
        ]

        # SQL injection patterns
        self.sql_patterns = [
            re.compile(r"\b(union|select|insert|update|delete|drop|alter|create)\b", re.IGNORECASE),
            re.compile(r"--\s"),
            re.compile(r"/\*.*?\*/", re.DOTALL),
            re.compile(r"\'\s*(or|and)\s*\'.+\'\s*=\s*\'", re.IGNORECASE),
        ]

        # Path traversal patterns
        self.path_traversal_patterns = [
            re.compile(r"\.\."),
            re.compile(r"/etc/passwd"),
            re.compile(r"/proc/"),
            re.compile(r"\\\\"),
        ]

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Apply security measures to incoming requests.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware or route handler

        Returns:
            HTTP response with security headers
        """
        # Check IP blocking
        client_ip = self._get_client_ip(request)
        if client_ip in self.blocked_ips:
            logger.warning(f"Blocked request from IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "type": "SecurityError",
                    "code": "GV_IP_BLOCKED",
                    "message": "Access denied",
                },
            )

        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            logger.warning(
                f"Request size too large: {content_length} bytes from {client_ip}",
                extra={
                    "client_ip": client_ip,
                    "content_length": content_length,
                    "max_allowed": self.max_request_size,
                },
            )
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "type": "SecurityError",
                    "code": "GV_REQUEST_TOO_LARGE",
                    "message": f"Request size exceeds maximum allowed size of {self.max_request_size} bytes",
                },
            )

        # Sanitize and validate request
        security_violation = await self._check_security_violations(request)
        if security_violation:
            logger.warning(
                f"Security violation detected: {security_violation}",
                extra={
                    "client_ip": client_ip,
                    "violation_type": security_violation,
                    "path": request.url.path,
                    "method": request.method,
                },
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "type": "SecurityError",
                    "code": "GV_SECURITY_VIOLATION",
                    "message": "Request contains potentially malicious content",
                },
            )

        # Process request
        response = await call_next(request)

        # Add security headers
        self._add_security_headers(response)

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        if request.client:
            return request.client.host

        return "unknown"

    async def _check_security_violations(self, request: Request) -> str:
        """
        Check for security violations in the request.

        Args:
            request: HTTP request

        Returns:
            Violation type if found, None otherwise
        """
        # Check URL path
        path = request.url.path
        query = str(request.url.query) if request.url.query else ""

        # Check for path traversal
        if self._contains_pattern(path + query, self.path_traversal_patterns):
            return "path_traversal"

        # Check for XSS
        if self._contains_pattern(path + query, self.xss_patterns):
            return "xss_attempt"

        # Check for SQL injection
        if self._contains_pattern(path + query, self.sql_patterns):
            return "sql_injection"

        # Check headers
        for header_name, header_value in request.headers.items():
            # Skip common headers that might contain these patterns legitimately
            if header_name.lower() in [
                "user-agent",
                "accept",
                "accept-encoding",
                "accept-language",
            ]:
                continue

            if self._contains_pattern(header_value, self.xss_patterns):
                return "xss_in_headers"

            if self._contains_pattern(header_value, self.sql_patterns):
                return "sql_injection_in_headers"

        return None

    def _contains_pattern(self, text: str, patterns: list) -> bool:
        """Check if text contains any of the given patterns."""
        for pattern in patterns:
            if pattern.search(text):
                return True
        return False

    def _add_security_headers(self, response: Response):
        """
        Add security headers to response.

        Args:
            response: HTTP response
        """
        # Strict Transport Security
        if self.enable_hsts:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        # Content Security Policy
        if self.enable_csp:
            csp_policy = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' https:; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )
            response.headers["Content-Security-Policy"] = csp_policy

        # X-Frame-Options
        response.headers["X-Frame-Options"] = "DENY"

        # X-Content-Type-Options
        response.headers["X-Content-Type-Options"] = "nosniff"

        # X-XSS-Protection
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions Policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "speaker=()"
        )

        # Remove server header
        response.headers.pop("server", None)

        # Add custom security header
        response.headers["X-Security-Framework"] = "GenomeVault-Gateway"

    def add_blocked_ip(self, ip: str):
        """
        Add IP to blocked list.

        Args:
            ip: IP address to block
        """
        self.blocked_ips.add(ip)
        logger.info(f"Added IP to blocklist: {ip}")

    def remove_blocked_ip(self, ip: str):
        """
        Remove IP from blocked list.

        Args:
            ip: IP address to unblock
        """
        self.blocked_ips.discard(ip)
        logger.info(f"Removed IP from blocklist: {ip}")

    def get_blocked_ips(self) -> Set[str]:
        """
        Get list of blocked IPs.

        Returns:
            Set of blocked IP addresses
        """
        return self.blocked_ips.copy()

    def sanitize_input(self, text: str) -> str:
        """
        Sanitize input text by removing potentially dangerous content.

        Args:
            text: Input text to sanitize

        Returns:
            Sanitized text
        """
        if not text:
            return text

        # Remove script tags
        for pattern in self.xss_patterns:
            text = pattern.sub("", text)

        # Basic HTML entity encoding for remaining suspicious characters
        replacements = {
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#x27;",
            "&": "&amp;",
        }

        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        return text
