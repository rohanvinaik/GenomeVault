"""
Security Headers Middleware for GenomeVault.

Implements comprehensive security headers to protect against various
web-based attacks including XSS, CSRF, clickjacking, and content sniffing.
Provides configurable security profiles for different deployment environments
with special considerations for clinical data protection.
"""

import os
from typing import Dict, Optional, List
from enum import Enum

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


class SecurityProfile(str, Enum):
    """Security profiles for different deployment environments."""
    
    DEVELOPMENT = "development"    # Relaxed security for development
    STAGING = "staging"           # Standard security for testing
    PRODUCTION = "production"     # High security for production
    CLINICAL = "clinical"         # Maximum security for PHI data


class CSPDirective(str, Enum):
    """Content Security Policy directive types."""
    
    DEFAULT_SRC = "default-src"
    SCRIPT_SRC = "script-src"
    STYLE_SRC = "style-src"
    IMG_SRC = "img-src"
    FONT_SRC = "font-src"
    CONNECT_SRC = "connect-src"
    MEDIA_SRC = "media-src"
    OBJECT_SRC = "object-src"
    CHILD_SRC = "child-src"
    FRAME_SRC = "frame-src"
    WORKER_SRC = "worker-src"
    MANIFEST_SRC = "manifest-src"
    BASE_URI = "base-uri"
    FORM_ACTION = "form-action"
    FRAME_ANCESTORS = "frame-ancestors"


class SecurityHeadersConfig:
    """Configuration for security headers."""
    
    def __init__(self, profile: SecurityProfile = SecurityProfile.PRODUCTION):
        """Initialize security headers configuration.
        
        Args:
            profile: Security profile to use
        """
        self.profile = profile
        self._init_config()
    
    def _init_config(self):
        """Initialize configuration based on security profile."""
        
        # Base configuration
        self.headers = {
            # Prevent MIME type sniffing
            'X-Content-Type-Options': 'nosniff',
            
            # Control referrer information
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            
            # Prevent browser from caching sensitive pages
            'Cache-Control': 'no-cache, no-store, must-revalidate, private',
            'Pragma': 'no-cache',
            'Expires': '0',
            
            # Remove server information
            'Server': 'GenomeVault'
        }
        
        # Profile-specific configurations
        if self.profile == SecurityProfile.DEVELOPMENT:
            self._configure_development()
        elif self.profile == SecurityProfile.STAGING:
            self._configure_staging()
        elif self.profile == SecurityProfile.PRODUCTION:
            self._configure_production()
        elif self.profile == SecurityProfile.CLINICAL:
            self._configure_clinical()
    
    def _configure_development(self):
        """Configure headers for development environment."""
        self.headers.update({
            'X-Frame-Options': 'SAMEORIGIN',  # Allow framing for dev tools
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'  # Still use HSTS
        })
        
        # Relaxed CSP for development
        csp = self._build_csp({
            CSPDirective.DEFAULT_SRC: ["'self'"],
            CSPDirective.SCRIPT_SRC: ["'self'", "'unsafe-inline'", "'unsafe-eval'", "localhost:*"],
            CSPDirective.STYLE_SRC: ["'self'", "'unsafe-inline'", "localhost:*"],
            CSPDirective.IMG_SRC: ["'self'", "data:", "localhost:*"],
            CSPDirective.CONNECT_SRC: ["'self'", "localhost:*", "ws://localhost:*", "wss://localhost:*"],
            CSPDirective.FONT_SRC: ["'self'", "data:"],
            CSPDirective.FRAME_ANCESTORS: ["'self'"]
        })
        self.headers['Content-Security-Policy'] = csp
    
    def _configure_staging(self):
        """Configure headers for staging environment."""
        self.headers.update({
            'X-Frame-Options': 'SAMEORIGIN',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
        })
        
        # Standard CSP for staging
        csp = self._build_csp({
            CSPDirective.DEFAULT_SRC: ["'self'"],
            CSPDirective.SCRIPT_SRC: ["'self'", "'unsafe-inline'"],  # Some inline scripts for staging
            CSPDirective.STYLE_SRC: ["'self'", "'unsafe-inline'"],
            CSPDirective.IMG_SRC: ["'self'", "data:", "https:"],
            CSPDirective.CONNECT_SRC: ["'self'", "https:", "wss:"],
            CSPDirective.FONT_SRC: ["'self'", "data:"],
            CSPDirective.FRAME_ANCESTORS: ["'none'"],
            CSPDirective.BASE_URI: ["'self'"],
            CSPDirective.FORM_ACTION: ["'self'"]
        })
        self.headers['Content-Security-Policy'] = csp
    
    def _configure_production(self):
        """Configure headers for production environment."""
        self.headers.update({
            'X-Frame-Options': 'DENY',  # Prevent all framing
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=63072000; includeSubDomains; preload',  # 2 years
            'X-Permitted-Cross-Domain-Policies': 'none'
        })
        
        # Strict CSP for production
        csp = self._build_csp({
            CSPDirective.DEFAULT_SRC: ["'none'"],  # Deny by default
            CSPDirective.SCRIPT_SRC: ["'self'"],   # Only same-origin scripts
            CSPDirective.STYLE_SRC: ["'self'"],    # Only same-origin styles
            CSPDirective.IMG_SRC: ["'self'", "data:"],
            CSPDirective.CONNECT_SRC: ["'self'"],
            CSPDirective.FONT_SRC: ["'self'"],
            CSPDirective.MEDIA_SRC: ["'none'"],
            CSPDirective.OBJECT_SRC: ["'none'"],
            CSPDirective.CHILD_SRC: ["'none'"],
            CSPDirective.FRAME_SRC: ["'none'"],
            CSPDirective.WORKER_SRC: ["'self'"],
            CSPDirective.MANIFEST_SRC: ["'self'"],
            CSPDirective.FRAME_ANCESTORS: ["'none'"],
            CSPDirective.BASE_URI: ["'self'"],
            CSPDirective.FORM_ACTION: ["'self'"]
        })
        self.headers['Content-Security-Policy'] = csp
        
        # Add report-only CSP for monitoring
        csp_report = csp + "; report-uri /api/security/csp-report"
        self.headers['Content-Security-Policy-Report-Only'] = csp_report
    
    def _configure_clinical(self):
        """Configure headers for clinical environment (maximum security)."""
        self.headers.update({
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=63072000; includeSubDomains; preload',
            'X-Permitted-Cross-Domain-Policies': 'none',
            'Cross-Origin-Resource-Policy': 'cross-origin',
            'Cross-Origin-Embedder-Policy': 'require-corp',
            'Cross-Origin-Opener-Policy': 'same-origin',
            'X-DNS-Prefetch-Control': 'off',  # Disable DNS prefetching
            'Feature-Policy': self._get_feature_policy(),
            'Permissions-Policy': self._get_permissions_policy()
        })
        
        # Maximum security CSP for clinical
        csp = self._build_csp({
            CSPDirective.DEFAULT_SRC: ["'none'"],
            CSPDirective.SCRIPT_SRC: ["'self'"],
            CSPDirective.STYLE_SRC: ["'self'"],
            CSPDirective.IMG_SRC: ["'self'"],
            CSPDirective.CONNECT_SRC: ["'self'"],
            CSPDirective.FONT_SRC: ["'self'"],
            CSPDirective.MEDIA_SRC: ["'none'"],
            CSPDirective.OBJECT_SRC: ["'none'"],
            CSPDirective.CHILD_SRC: ["'none'"],
            CSPDirective.FRAME_SRC: ["'none'"],
            CSPDirective.WORKER_SRC: ["'none'"],  # No workers in clinical mode
            CSPDirective.MANIFEST_SRC: ["'self'"],
            CSPDirective.FRAME_ANCESTORS: ["'none'"],
            CSPDirective.BASE_URI: ["'none'"],    # No base URI changes
            CSPDirective.FORM_ACTION: ["'self'"]
        })
        self.headers['Content-Security-Policy'] = csp
        
        # Clinical environments should have strict cache control
        self.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private, max-age=0'
    
    def _build_csp(self, directives: Dict[CSPDirective, List[str]]) -> str:
        """Build Content Security Policy header value."""
        csp_parts = []
        for directive, sources in directives.items():
            sources_str = ' '.join(sources)
            csp_parts.append(f"{directive.value} {sources_str}")
        return '; '.join(csp_parts)
    
    def _get_feature_policy(self) -> str:
        """Get Feature Policy header value."""
        # Disable potentially sensitive features
        policies = [
            "geolocation 'none'",
            "microphone 'none'",
            "camera 'none'",
            "magnetometer 'none'",
            "gyroscope 'none'",
            "speaker 'none'",
            "vibrate 'none'",
            "fullscreen 'self'",
            "payment 'none'"
        ]
        return ', '.join(policies)
    
    def _get_permissions_policy(self) -> str:
        """Get Permissions Policy header value."""
        # Modern replacement for Feature Policy
        policies = [
            "geolocation=()",
            "microphone=()",
            "camera=()",
            "magnetometer=()",
            "gyroscope=()",
            "speaker=()",
            "vibrate=()",
            "fullscreen=(self)",
            "payment=()"
        ]
        return ', '.join(policies)
    
    def get_headers(self) -> Dict[str, str]:
        """Get all configured security headers."""
        return self.headers.copy()
    
    def update_csp_for_endpoint(self, endpoint: str) -> Dict[str, str]:
        """Update CSP based on specific endpoint requirements."""
        headers = self.headers.copy()
        
        # API endpoints might need different CSP
        if endpoint.startswith('/api/'):
            # Remove CSP for API endpoints as they return JSON
            headers.pop('Content-Security-Policy', None)
            headers.pop('Content-Security-Policy-Report-Only', None)
        
        # Documentation endpoints might need relaxed CSP
        elif endpoint in ['/docs', '/redoc', '/openapi.json']:
            if self.profile != SecurityProfile.CLINICAL:
                # Allow inline styles for documentation
                csp = self._build_csp({
                    CSPDirective.DEFAULT_SRC: ["'self'"],
                    CSPDirective.SCRIPT_SRC: ["'self'", "'unsafe-inline'"],
                    CSPDirective.STYLE_SRC: ["'self'", "'unsafe-inline'"],
                    CSPDirective.IMG_SRC: ["'self'", "data:"],
                    CSPDirective.CONNECT_SRC: ["'self'"],
                    CSPDirective.FONT_SRC: ["'self'", "data:"]
                })
                headers['Content-Security-Policy'] = csp
        
        return headers


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""
    
    def __init__(self, 
                 app,
                 profile: SecurityProfile = SecurityProfile.PRODUCTION,
                 custom_headers: Optional[Dict[str, str]] = None,
                 exclude_paths: Optional[List[str]] = None):
        """Initialize security headers middleware.
        
        Args:
            app: FastAPI application
            profile: Security profile to use
            custom_headers: Additional custom headers
            exclude_paths: Paths to exclude from security headers
        """
        super().__init__(app)
        
        self.config = SecurityHeadersConfig(profile)
        self.custom_headers = custom_headers or {}
        self.exclude_paths = set(exclude_paths or [])
        
        # Paths that should always have minimal headers
        self.minimal_header_paths = {'/health', '/ping', '/metrics'}
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Add security headers to responses."""
        
        # Process the request
        response = await call_next(request)
        
        # Skip security headers for excluded paths
        if self._should_skip_headers(request.url.path):
            return response
        
        # Get headers for this endpoint
        if request.url.path in self.minimal_header_paths:
            headers = self._get_minimal_headers()
        else:
            headers = self.config.update_csp_for_endpoint(request.url.path)
        
        # Add custom headers
        headers.update(self.custom_headers)
        
        # Apply headers to response
        for name, value in headers.items():
            response.headers[name] = value
        
        # Add special headers for error responses
        if response.status_code >= 400:
            self._add_error_response_headers(response)
        
        return response
    
    def _should_skip_headers(self, path: str) -> bool:
        """Check if security headers should be skipped for this path."""
        return any(excluded_path in path for excluded_path in self.exclude_paths)
    
    def _get_minimal_headers(self) -> Dict[str, str]:
        """Get minimal security headers for health check endpoints."""
        return {
            'X-Content-Type-Options': 'nosniff',
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Server': 'GenomeVault'
        }
    
    def _add_error_response_headers(self, response: Response):
        """Add additional security headers for error responses."""
        # Ensure error responses are not cached
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private'
        
        # Add security headers to prevent information leakage
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-Content-Type-Options'] = 'nosniff'


def create_security_headers_middleware(
    profile: Optional[SecurityProfile] = None,
    custom_headers: Optional[Dict[str, str]] = None
) -> callable:
    """Create security headers middleware with environment-based configuration.
    
    Args:
        profile: Override security profile
        custom_headers: Additional custom headers
        
    Returns:
        Security headers middleware factory function
    """
    
    # Determine profile from environment if not specified
    if profile is None:
        env = os.getenv('GENOMEVAULT_ENV', 'development').lower()
        clinical_mode = os.getenv('GENOMEVAULT_CLINICAL_MODE', 'false').lower() == 'true'
        
        if clinical_mode:
            profile = SecurityProfile.CLINICAL
        elif env == 'production':
            profile = SecurityProfile.PRODUCTION
        elif env == 'staging':
            profile = SecurityProfile.STAGING
        else:
            profile = SecurityProfile.DEVELOPMENT
    
    # Get custom headers from environment
    if custom_headers is None:
        custom_headers = {}
        
        # Add organization-specific headers if configured
        org_name = os.getenv('GENOMEVAULT_ORG_NAME')
        if org_name:
            custom_headers['X-Organization'] = org_name
        
        # Add compliance headers if in clinical mode
        if profile == SecurityProfile.CLINICAL:
            custom_headers.update({
                'X-HIPAA-Compliant': 'true',
                'X-PHI-Protected': 'true',
                'X-Security-Level': 'clinical'
            })
    
    def middleware_factory(app):
        return SecurityHeadersMiddleware(
            app=app,
            profile=profile,
            custom_headers=custom_headers
        )
    
    return middleware_factory


# CSP Report endpoint handler
async def handle_csp_report(request: Request) -> Response:
    """Handle Content Security Policy violation reports."""
    try:
        report_data = await request.json()
        
        # Log CSP violation (without exposing sensitive data)
        violation = report_data.get('csp-report', {})
        print(f"CSP Violation: {violation.get('violated-directive', 'unknown')} "
              f"blocked-uri: {violation.get('blocked-uri', 'unknown')}")
        
        # In production, you might want to send this to a security monitoring system
        
    except Exception as e:
        print(f"Error processing CSP report: {e}")
    
    # Always return 204 for CSP reports
    return Response(status_code=204)


# Utility functions for security headers
def get_security_profile_info(profile: SecurityProfile) -> Dict[str, any]:
    """Get information about a security profile.
    
    Args:
        profile: Security profile to describe
        
    Returns:
        Dictionary with profile information
    """
    profiles = {
        SecurityProfile.DEVELOPMENT: {
            "description": "Relaxed security for development",
            "frame_options": "SAMEORIGIN",
            "hsts_max_age": 31536000,  # 1 year
            "csp_strictness": "relaxed",
            "features_disabled": ["geolocation", "camera", "microphone"]
        },
        SecurityProfile.STAGING: {
            "description": "Standard security for testing",
            "frame_options": "SAMEORIGIN",
            "hsts_max_age": 31536000,
            "csp_strictness": "standard",
            "features_disabled": ["geolocation", "camera", "microphone", "payment"]
        },
        SecurityProfile.PRODUCTION: {
            "description": "High security for production",
            "frame_options": "DENY",
            "hsts_max_age": 63072000,  # 2 years
            "csp_strictness": "strict",
            "features_disabled": ["geolocation", "camera", "microphone", "payment", "vibrate"]
        },
        SecurityProfile.CLINICAL: {
            "description": "Maximum security for PHI data",
            "frame_options": "DENY", 
            "hsts_max_age": 63072000,
            "csp_strictness": "maximum",
            "features_disabled": ["geolocation", "camera", "microphone", "payment", "vibrate", "fullscreen"]
        }
    }
    
    return profiles.get(profile, {})


def validate_security_headers(headers: Dict[str, str], profile: SecurityProfile) -> List[str]:
    """Validate that required security headers are present.
    
    Args:
        headers: Response headers to validate
        profile: Expected security profile
        
    Returns:
        List of missing or incorrect headers
    """
    issues = []
    
    required_headers = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY' if profile in [SecurityProfile.PRODUCTION, SecurityProfile.CLINICAL] else 'SAMEORIGIN',
        'Referrer-Policy': 'strict-origin-when-cross-origin'
    }
    
    if profile in [SecurityProfile.PRODUCTION, SecurityProfile.CLINICAL]:
        required_headers['Strict-Transport-Security'] = 'max-age=63072000; includeSubDomains; preload'
    
    for header, expected_value in required_headers.items():
        actual_value = headers.get(header)
        if not actual_value:
            issues.append(f"Missing required header: {header}")
        elif expected_value not in actual_value:
            issues.append(f"Incorrect header value for {header}: expected '{expected_value}', got '{actual_value}'")
    
    return issues