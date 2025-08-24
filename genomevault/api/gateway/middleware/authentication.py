"""
Authentication middleware for GenomeVault API Gateway.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import HTTPException, Request, Response, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware

from genomevault.observability.logging import get_logger

logger = get_logger(__name__)

# Security scheme for dependency injection
security_scheme = HTTPBearer(auto_error=False)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for API requests.
    
    Supports multiple authentication methods:
    - API Key authentication (X-API-Key header)
    - OAuth2 Bearer token authentication
    - JWT token authentication
    """
    
    def __init__(self, app, api_key_header: str = "X-API-Key"):
        """
        Initialize authentication middleware.
        
        Args:
            app: FastAPI application instance
            api_key_header: Header name for API key authentication
        """
        super().__init__(app)
        self.api_key_header = api_key_header
        
        # Paths that don't require authentication
        self.public_paths = {
            "/docs",
            "/redoc", 
            "/openapi.json",
            "/health",
            "/health/liveness",
            "/health/readiness",
            "/"
        }
        
        # Paths that require different authentication levels
        self.clinical_paths = {"/clinical", "/specialized/audit"}
        self.admin_paths = {"/admin", "/system"}
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process authentication for incoming requests.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or route handler
            
        Returns:
            HTTP response
        """
        # Add request ID for tracing
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Skip authentication for public paths
        if self._is_public_path(request.url.path):
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        
        # Attempt authentication
        auth_result = await self._authenticate_request(request)
        
        if not auth_result["authenticated"]:
            logger.warning(
                f"Authentication failed for {request.url.path}",
                extra={
                    "request_id": request_id,
                    "ip_address": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent"),
                    "error": auth_result["error"]
                }
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "type": "AuthenticationError",
                    "code": "GV_UNAUTHORIZED", 
                    "message": auth_result["error"],
                    "request_id": request_id
                },
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Store authentication context in request state
        request.state.user_id = auth_result["user_id"]
        request.state.auth_method = auth_result["method"]
        request.state.permissions = auth_result["permissions"]
        request.state.rate_limit_tier = auth_result["rate_limit_tier"]
        
        # Check authorization for restricted paths
        if not await self._check_authorization(request, auth_result):
            logger.warning(
                f"Authorization failed for {request.url.path}",
                extra={
                    "request_id": request_id,
                    "user_id": auth_result["user_id"],
                    "required_permissions": self._get_required_permissions(request.url.path)
                }
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "type": "AuthorizationError",
                    "code": "GV_FORBIDDEN",
                    "message": "Insufficient permissions for this resource",
                    "request_id": request_id
                }
            )
        
        # Log successful authentication
        logger.info(
            f"Authenticated request to {request.url.path}",
            extra={
                "request_id": request_id,
                "user_id": auth_result["user_id"],
                "auth_method": auth_result["method"]
            }
        )
        
        # Process request
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response
    
    def _is_public_path(self, path: str) -> bool:
        """Check if path is public and doesn't require authentication."""
        return any(path.startswith(public_path) for public_path in self.public_paths)
    
    async def _authenticate_request(self, request: Request) -> dict:
        """
        Authenticate the incoming request using available methods.
        
        Args:
            request: HTTP request
            
        Returns:
            Authentication result dictionary
        """
        # Try API key authentication first
        api_key = request.headers.get(self.api_key_header)
        if api_key:
            return await self._authenticate_api_key(api_key, request)
        
        # Try OAuth2/JWT Bearer token authentication
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ", 1)[1]
            return await self._authenticate_bearer_token(token, request)
        
        return {
            "authenticated": False,
            "error": "No valid authentication credentials provided"
        }
    
    async def _authenticate_api_key(self, api_key: str, request: Request) -> dict:
        """
        Authenticate using API key.
        
        Args:
            api_key: API key from header
            request: HTTP request
            
        Returns:
            Authentication result
        """
        try:
            # Validate API key format
            if not api_key.startswith("gv_") or len(api_key) < 20:
                return {
                    "authenticated": False,
                    "error": "Invalid API key format"
                }
            
            # TODO: Replace with actual API key validation
            # This would typically query a database or cache
            # For now, we'll use a simple validation for demo purposes
            user_id = await self._validate_api_key(api_key)
            
            if not user_id:
                return {
                    "authenticated": False,
                    "error": "Invalid or expired API key"
                }
            
            # Get user permissions and rate limit tier
            permissions, rate_limit_tier = await self._get_user_context(user_id)
            
            return {
                "authenticated": True,
                "user_id": user_id,
                "method": "api_key",
                "permissions": permissions,
                "rate_limit_tier": rate_limit_tier
            }
            
        except Exception as e:
            logger.error(f"API key authentication error: {e}")
            return {
                "authenticated": False,
                "error": "Authentication service unavailable"
            }
    
    async def _authenticate_bearer_token(self, token: str, request: Request) -> dict:
        """
        Authenticate using OAuth2/JWT Bearer token.
        
        Args:
            token: Bearer token
            request: HTTP request
            
        Returns:
            Authentication result
        """
        try:
            # TODO: Replace with actual JWT token validation
            # This would typically validate JWT signature and decode claims
            user_context = await self._validate_bearer_token(token)
            
            if not user_context:
                return {
                    "authenticated": False,
                    "error": "Invalid or expired token"
                }
            
            return {
                "authenticated": True,
                "user_id": user_context["user_id"],
                "method": "bearer_token",
                "permissions": user_context["permissions"],
                "rate_limit_tier": user_context["rate_limit_tier"]
            }
            
        except Exception as e:
            logger.error(f"Bearer token authentication error: {e}")
            return {
                "authenticated": False,
                "error": "Token validation failed"
            }
    
    async def _validate_api_key(self, api_key: str) -> Optional[str]:
        """
        Validate API key and return user ID.
        
        Args:
            api_key: API key to validate
            
        Returns:
            User ID if valid, None otherwise
        """
        # TODO: Implement actual API key validation
        # This would query your authentication service/database
        
        # For demo purposes, accept keys that start with 'gv_demo_'
        if api_key.startswith("gv_demo_"):
            return f"user_{api_key[8:16]}"
        
        return None
    
    async def _validate_bearer_token(self, token: str) -> Optional[dict]:
        """
        Validate Bearer token and return user context.
        
        Args:
            token: Bearer token to validate
            
        Returns:
            User context if valid, None otherwise
        """
        # TODO: Implement actual JWT token validation
        # This would decode and verify JWT tokens
        
        # For demo purposes, accept tokens that start with 'demo_'
        if token.startswith("demo_"):
            return {
                "user_id": f"user_{token[5:13]}",
                "permissions": ["genomic:read", "pir:query"],
                "rate_limit_tier": "standard"
            }
        
        return None
    
    async def _get_user_context(self, user_id: str) -> tuple[list[str], str]:
        """
        Get user permissions and rate limit tier.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (permissions, rate_limit_tier)
        """
        # TODO: Implement actual user context retrieval
        # This would query user permissions and subscription tier
        
        # For demo purposes, return basic permissions
        return ["genomic:read", "pir:query"], "standard"
    
    async def _check_authorization(self, request: Request, auth_result: dict) -> bool:
        """
        Check if user is authorized to access the requested resource.
        
        Args:
            request: HTTP request
            auth_result: Authentication result
            
        Returns:
            True if authorized, False otherwise
        """
        user_permissions = auth_result.get("permissions", [])
        path = request.url.path
        
        # Check clinical path permissions
        if any(path.startswith(clinical_path) for clinical_path in self.clinical_paths):
            return "clinical:analyze" in user_permissions
        
        # Check admin path permissions
        if any(path.startswith(admin_path) for admin_path in self.admin_paths):
            return "admin:manage" in user_permissions
        
        # For other paths, basic authentication is sufficient
        return True
    
    def _get_required_permissions(self, path: str) -> list[str]:
        """
        Get required permissions for a given path.
        
        Args:
            path: Request path
            
        Returns:
            List of required permissions
        """
        if any(path.startswith(clinical_path) for clinical_path in self.clinical_paths):
            return ["clinical:analyze"]
        
        if any(path.startswith(admin_path) for admin_path in self.admin_paths):
            return ["admin:manage"]
        
        if path.startswith("/vectors"):
            return ["genomic:read"]
        
        if path.startswith("/queries"):
            return ["pir:query"]
        
        if path.startswith("/proofs"):
            return ["zk:prove"]
        
        return []