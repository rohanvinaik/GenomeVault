"""
Advanced CORS Security Configuration for GenomeVault.

Implements sophisticated Cross-Origin Resource Sharing (CORS) policies
with enhanced security features for protecting genomic and clinical data.
Includes dynamic origin validation, preflight caching, and security headers.
"""

import os
import re
from typing import List, Set, Optional, Dict, Callable, Union
from urllib.parse import urlparse
from enum import Enum

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import PlainTextResponse
from starlette.status import HTTP_200_OK, HTTP_403_FORBIDDEN, HTTP_204_NO_CONTENT


class CORSSecurityLevel(str, Enum):
    """Security levels for CORS configuration."""
    
    PERMISSIVE = "permissive"    # Development/testing
    STANDARD = "standard"        # Production with known origins
    STRICT = "strict"           # High security with minimal access
    CLINICAL = "clinical"       # Maximum security for PHI data


class OriginValidator:
    """Validates origins against security policies."""
    
    def __init__(self, 
                 allowed_origins: List[str],
                 security_level: CORSSecurityLevel = CORSSecurityLevel.STANDARD):
        """Initialize origin validator.
        
        Args:
            allowed_origins: List of allowed origins (supports wildcards)
            security_level: Security level for validation
        """
        self.allowed_origins = set(allowed_origins)
        self.security_level = security_level
        self.origin_patterns = self._compile_patterns(allowed_origins)
        
        # Security-level specific configurations
        self.strict_https_only = security_level in [CORSSecurityLevel.STRICT, CORSSecurityLevel.CLINICAL]
        self.allow_localhost = security_level in [CORSSecurityLevel.PERMISSIVE, CORSSecurityLevel.STANDARD]
        self.require_port_match = security_level == CORSSecurityLevel.CLINICAL
    
    def _compile_patterns(self, origins: List[str]) -> List[re.Pattern]:
        """Compile origin patterns for efficient matching."""
        patterns = []
        for origin in origins:
            if '*' in origin:
                # Convert wildcard pattern to regex
                pattern = origin.replace('*', r'[a-zA-Z0-9\-]+')
                pattern = f"^{pattern}$"
                patterns.append(re.compile(pattern))
        return patterns
    
    def is_origin_allowed(self, origin: str) -> tuple[bool, str]:
        """Check if origin is allowed.
        
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        if not origin:
            return False, "No origin header provided"
        
        # Parse the origin
        try:
            parsed = urlparse(origin)
        except Exception:
            return False, "Invalid origin format"
        
        # Security level checks
        if self.strict_https_only and parsed.scheme != 'https':
            if not (self.allow_localhost and parsed.hostname in ['localhost', '127.0.0.1', '::1']):
                return False, "HTTPS required"
        
        # Exact match check
        if origin in self.allowed_origins:
            return True, "Exact match"
        
        # Wildcard pattern matching
        for pattern in self.origin_patterns:
            if pattern.match(origin):
                return True, "Pattern match"
        
        # Special handling for localhost in development
        if self.allow_localhost and self._is_localhost_origin(parsed):
            return True, "Localhost allowed"
        
        return False, "Origin not in allowed list"
    
    def _is_localhost_origin(self, parsed_origin) -> bool:
        """Check if origin is a localhost variant."""
        localhost_hosts = ['localhost', '127.0.0.1', '::1']
        return parsed_origin.hostname in localhost_hosts
    
    def get_allowed_origins_for_response(self) -> Union[str, List[str]]:
        """Get origins to return in Access-Control-Allow-Origin header."""
        # For security, never return wildcard in production
        if self.security_level in [CORSSecurityLevel.STRICT, CORSSecurityLevel.CLINICAL]:
            return list(self.allowed_origins)
        
        # Return specific origins for standard security
        return list(self.allowed_origins)


class CORSSecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced CORS middleware with security features."""
    
    def __init__(self,
                 app,
                 allowed_origins: List[str],
                 allowed_methods: List[str] = None,
                 allowed_headers: List[str] = None,
                 exposed_headers: List[str] = None,
                 allow_credentials: bool = True,
                 max_age: int = 86400,  # 24 hours
                 security_level: CORSSecurityLevel = CORSSecurityLevel.STANDARD,
                 enable_preflight_caching: bool = True):
        """Initialize CORS security middleware.
        
        Args:
            app: FastAPI application
            allowed_origins: List of allowed origins
            allowed_methods: HTTP methods to allow
            allowed_headers: Headers to allow in requests
            exposed_headers: Headers to expose in responses
            allow_credentials: Whether to allow credentials
            max_age: Preflight cache duration in seconds
            security_level: Security level for CORS policies
            enable_preflight_caching: Whether to cache preflight responses
        """
        super().__init__(app)
        
        self.origin_validator = OriginValidator(allowed_origins, security_level)
        self.security_level = security_level
        
        # Configure allowed methods based on security level
        if allowed_methods is None:
            if security_level == CORSSecurityLevel.CLINICAL:
                allowed_methods = ['GET', 'POST']  # Minimal methods for clinical
            else:
                allowed_methods = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'HEAD']
        
        self.allowed_methods = set(allowed_methods)
        
        # Configure allowed headers
        if allowed_headers is None:
            base_headers = [
                'Accept',
                'Accept-Language',
                'Content-Language',
                'Content-Type',
                'Authorization',
                'X-API-Key',
                'X-Request-ID'
            ]
            
            # Add additional headers for less restrictive security levels
            if security_level in [CORSSecurityLevel.PERMISSIVE, CORSSecurityLevel.STANDARD]:
                base_headers.extend([
                    'X-Requested-With',
                    'X-CSRF-Token',
                    'Cache-Control'
                ])
            
            allowed_headers = base_headers
        
        self.allowed_headers = set(h.lower() for h in allowed_headers)
        
        # Configure exposed headers
        if exposed_headers is None:
            base_exposed = [
                'X-Request-ID',
                'X-Response-Time',
                'X-RateLimit-Remaining',
                'X-RateLimit-Reset'
            ]
            
            # Limit exposed headers for clinical security
            if security_level == CORSSecurityLevel.CLINICAL:
                base_exposed = ['X-Request-ID']
            
            exposed_headers = base_exposed
        
        self.exposed_headers = exposed_headers
        self.allow_credentials = allow_credentials
        self.max_age = max_age
        self.enable_preflight_caching = enable_preflight_caching
        
        # Preflight cache (in-memory for simplicity)
        self._preflight_cache: Dict[str, Dict] = {}
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Handle CORS for all requests."""
        
        origin = request.headers.get('origin')
        
        # Handle preflight requests
        if request.method == 'OPTIONS':
            return await self._handle_preflight(request, origin)
        
        # Process actual request
        response = await call_next(request)
        
        # Add CORS headers to response
        self._add_cors_headers(response, request, origin)
        
        return response
    
    async def _handle_preflight(self, request: Request, origin: Optional[str]) -> Response:
        """Handle CORS preflight requests."""
        
        # Check if origin is allowed
        if origin:
            allowed, reason = self.origin_validator.is_origin_allowed(origin)
            if not allowed:
                # Log security event
                print(f"CORS preflight denied: {reason} for origin: {origin}")
                return PlainTextResponse(
                    "CORS preflight denied",
                    status_code=HTTP_403_FORBIDDEN
                )
        
        # Check if method is allowed
        requested_method = request.headers.get('access-control-request-method')
        if requested_method and requested_method not in self.allowed_methods:
            return PlainTextResponse(
                "Method not allowed",
                status_code=HTTP_403_FORBIDDEN
            )
        
        # Check if headers are allowed
        requested_headers = request.headers.get('access-control-request-headers')
        if requested_headers:
            headers = [h.strip().lower() for h in requested_headers.split(',')]
            for header in headers:
                if header not in self.allowed_headers:
                    return PlainTextResponse(
                        f"Header not allowed: {header}",
                        status_code=HTTP_403_FORBIDDEN
                    )
        
        # Create preflight response
        response = Response(status_code=HTTP_204_NO_CONTENT)
        
        # Add CORS headers
        if origin:
            response.headers['access-control-allow-origin'] = origin
        
        response.headers['access-control-allow-methods'] = ', '.join(self.allowed_methods)
        response.headers['access-control-allow-headers'] = ', '.join(self.allowed_headers)
        
        if self.exposed_headers:
            response.headers['access-control-expose-headers'] = ', '.join(self.exposed_headers)
        
        if self.allow_credentials:
            response.headers['access-control-allow-credentials'] = 'true'
        
        if self.enable_preflight_caching:
            response.headers['access-control-max-age'] = str(self.max_age)
        
        # Add security headers
        self._add_security_headers(response)
        
        return response
    
    def _add_cors_headers(self, response: Response, request: Request, origin: Optional[str]):
        """Add CORS headers to actual responses."""
        
        if not origin:
            return
        
        # Check if origin is allowed
        allowed, reason = self.origin_validator.is_origin_allowed(origin)
        if not allowed:
            # Don't add CORS headers for disallowed origins
            print(f"CORS response denied: {reason} for origin: {origin}")
            return
        
        # Add CORS headers
        response.headers['access-control-allow-origin'] = origin
        
        if self.exposed_headers:
            response.headers['access-control-expose-headers'] = ', '.join(self.exposed_headers)
        
        if self.allow_credentials:
            response.headers['access-control-allow-credentials'] = 'true'
        
        # Add security headers
        self._add_security_headers(response)
    
    def _add_security_headers(self, response: Response):
        """Add additional security headers."""
        
        # Add Vary header for proper caching
        vary_header = response.headers.get('vary', '')
        if 'origin' not in vary_header.lower():
            if vary_header:
                vary_header += ', Origin'
            else:
                vary_header = 'Origin'
            response.headers['vary'] = vary_header
        
        # Add security headers based on security level
        if self.security_level in [CORSSecurityLevel.STRICT, CORSSecurityLevel.CLINICAL]:
            # Strict security headers
            response.headers['x-frame-options'] = 'DENY'
            response.headers['x-content-type-options'] = 'nosniff'
            response.headers['referrer-policy'] = 'strict-origin-when-cross-origin'
            
            # Clinical level gets additional headers
            if self.security_level == CORSSecurityLevel.CLINICAL:
                response.headers['x-permitted-cross-domain-policies'] = 'none'
                response.headers['cross-origin-resource-policy'] = 'cross-origin'


def create_cors_middleware(
    allowed_origins: Optional[List[str]] = None,
    security_level: Optional[CORSSecurityLevel] = None
) -> Callable:
    """Create CORS middleware with environment-based configuration.
    
    Args:
        allowed_origins: Override allowed origins
        security_level: Override security level
    
    Returns:
        CORS middleware factory function
    """
    
    # Get configuration from environment
    if allowed_origins is None:
        env_origins = os.getenv('GENOMEVAULT_CORS_ORIGINS', '')
        if env_origins:
            allowed_origins = [origin.strip() for origin in env_origins.split(',') if origin.strip()]
        else:
            # Default origins based on environment
            env = os.getenv('GENOMEVAULT_ENV', 'development').lower()
            if env == 'production':
                allowed_origins = [
                    'https://genomevault.com',
                    'https://api.genomevault.com',
                    'https://clinical.genomevault.com'
                ]
            elif env == 'staging':
                allowed_origins = [
                    'https://staging.genomevault.com',
                    'https://staging-api.genomevault.com',
                    'http://localhost:3000',
                    'http://localhost:8080'
                ]
            else:  # development
                allowed_origins = [
                    'http://localhost:3000',
                    'http://localhost:8080',
                    'http://localhost:5173',  # Vite dev server
                    'http://127.0.0.1:3000',
                    'http://127.0.0.1:8080'
                ]
    
    # Determine security level
    if security_level is None:
        env = os.getenv('GENOMEVAULT_ENV', 'development').lower()
        clinical_mode = os.getenv('GENOMEVAULT_CLINICAL_MODE', 'false').lower() == 'true'
        
        if clinical_mode:
            security_level = CORSSecurityLevel.CLINICAL
        elif env == 'production':
            security_level = CORSSecurityLevel.STRICT
        elif env == 'staging':
            security_level = CORSSecurityLevel.STANDARD
        else:
            security_level = CORSSecurityLevel.PERMISSIVE
    
    def cors_middleware_factory(app):
        return CORSSecurityMiddleware(
            app=app,
            allowed_origins=allowed_origins,
            security_level=security_level,
            allow_credentials=True,
            max_age=86400 if security_level != CORSSecurityLevel.CLINICAL else 3600
        )
    
    return cors_middleware_factory


# Utility functions for CORS validation
def validate_origin_security(origin: str, security_level: CORSSecurityLevel) -> tuple[bool, str]:
    """Validate an origin against security requirements.
    
    Args:
        origin: Origin to validate
        security_level: Security level to check against
        
    Returns:
        Tuple of (is_secure: bool, reason: str)
    """
    try:
        parsed = urlparse(origin)
    except Exception:
        return False, "Invalid origin format"
    
    # Check HTTPS requirement
    if security_level in [CORSSecurityLevel.STRICT, CORSSecurityLevel.CLINICAL]:
        if parsed.scheme != 'https':
            # Allow localhost for development
            if parsed.hostname not in ['localhost', '127.0.0.1', '::1']:
                return False, "HTTPS required for this security level"
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'\.ngrok\.io$',      # Development tunnels
        r'\.localtunnel\.me$',
        r'\.herokuapp\.com$', # Public hosting (might be temporary)
    ]
    
    if security_level == CORSSecurityLevel.CLINICAL:
        for pattern in suspicious_patterns:
            if re.search(pattern, parsed.hostname or ''):
                return False, f"Suspicious domain pattern: {pattern}"
    
    return True, "Origin passes security checks"


def get_cors_policy_info(security_level: CORSSecurityLevel) -> Dict[str, any]:
    """Get information about CORS policy for a security level.
    
    Args:
        security_level: Security level to describe
        
    Returns:
        Dictionary with policy information
    """
    policies = {
        CORSSecurityLevel.PERMISSIVE: {
            "description": "Permissive CORS policy for development",
            "https_required": False,
            "allowed_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
            "credentials_allowed": True,
            "max_age": 86400,
            "security_headers": ["basic"]
        },
        CORSSecurityLevel.STANDARD: {
            "description": "Standard CORS policy for production",
            "https_required": False,
            "allowed_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
            "credentials_allowed": True,
            "max_age": 86400,
            "security_headers": ["standard"]
        },
        CORSSecurityLevel.STRICT: {
            "description": "Strict CORS policy for high security",
            "https_required": True,
            "allowed_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "credentials_allowed": True,
            "max_age": 3600,
            "security_headers": ["strict", "frame-denial", "content-type-options"]
        },
        CORSSecurityLevel.CLINICAL: {
            "description": "Maximum security CORS policy for PHI data",
            "https_required": True,
            "allowed_methods": ["GET", "POST"],
            "credentials_allowed": True,
            "max_age": 1800,
            "security_headers": ["strict", "frame-denial", "content-type-options", "cross-domain-policies"]
        }
    }
    
    return policies.get(security_level, {})