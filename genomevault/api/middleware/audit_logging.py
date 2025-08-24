"""
Audit Logging Middleware for GenomeVault.

Provides comprehensive audit logging for security events, API access,
and sensitive operations while strictly protecting PHI data. Implements
tamper-resistant logging with cryptographic signatures and structured
JSON output for security monitoring systems.
"""

import json
import time
import hashlib
import hmac
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Set
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


class AuditEventType(str, Enum):
    """Types of events that should be audited."""
    
    # Authentication events
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    AUTH_TOKEN_CREATED = "auth_token_created"
    AUTH_TOKEN_REVOKED = "auth_token_revoked"
    
    # Authorization events
    AUTHZ_ACCESS_GRANTED = "authz_access_granted"
    AUTHZ_ACCESS_DENIED = "authz_access_denied"
    AUTHZ_SCOPE_ESCALATION = "authz_scope_escalation"
    
    # Data access events
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    
    # Clinical data events (PHI-related)
    PHI_ACCESS = "phi_access"
    PHI_MODIFICATION = "phi_modification"
    PHI_EXPORT_ATTEMPTED = "phi_export_attempted"
    PHI_BREACH_DETECTED = "phi_breach_detected"
    
    # Genomic analysis events
    GENOMIC_ANALYSIS_START = "genomic_analysis_start"
    GENOMIC_ANALYSIS_COMPLETE = "genomic_analysis_complete"
    HDC_ENCODING = "hdc_encoding"
    ZK_PROOF_GENERATION = "zk_proof_generation"
    PIR_QUERY = "pir_query"
    
    # Security events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    INPUT_VALIDATION_FAILURE = "input_validation_failure"
    SECURITY_POLICY_VIOLATION = "security_policy_violation"
    
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIGURATION_CHANGE = "configuration_change"
    ERROR_OCCURRED = "error_occurred"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    
    LOW = "low"          # Normal operations
    MEDIUM = "medium"    # Important events
    HIGH = "high"        # Security-relevant events
    CRITICAL = "critical"  # Security incidents, PHI breaches


@dataclass
class AuditEvent:
    """Structure for audit log entries."""
    
    # Core event information
    event_id: str
    timestamp: str
    event_type: AuditEventType
    severity: AuditSeverity
    
    # Actor information (who)
    actor_id: Optional[str]  # API key ID, user ID (hashed)
    actor_type: Optional[str]  # "api_key", "user", "system"
    session_id: Optional[str]  # Session identifier (hashed)
    
    # Request context (what/where)
    resource_type: Optional[str]  # "genomic_data", "clinical_data", "system_config"
    resource_id: Optional[str]   # Resource identifier (hashed if sensitive)
    action: Optional[str]        # HTTP method or operation name
    endpoint: Optional[str]      # API endpoint accessed
    
    # Request metadata (how/when)
    request_id: Optional[str]    # Request correlation ID
    client_info: Optional[Dict[str, Any]]  # Sanitized client information
    request_size: Optional[int]
    response_code: Optional[int]
    response_size: Optional[int]
    duration_ms: Optional[float]
    
    # Additional context
    details: Optional[Dict[str, Any]]  # Additional event-specific data
    risk_indicators: Optional[List[str]]  # Security risk indicators
    
    # Integrity protection
    signature: Optional[str]  # HMAC signature for tamper detection
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Remove None values for cleaner logs
        return {k: v for k, v in data.items() if v is not None}


class PHIDataProtector:
    """Protects PHI data in audit logs."""
    
    # Fields that commonly contain PHI
    PHI_FIELDS = {
        'patient_id', 'sample_id', 'subject_id', 'participant_id',
        'medical_record_number', 'mrn', 'ssn', 'social_security_number',
        'name', 'first_name', 'last_name', 'email', 'phone', 'address',
        'date_of_birth', 'dob', 'birth_date', 'diagnosis', 'condition',
        'genomic_data', 'sequence_data', 'variant_data', 'clinical_notes'
    }
    
    # Patterns that might indicate PHI
    PHI_PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email pattern
        r'\b\d{3}-\d{3}-\d{4}\b',  # Phone number pattern
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # Date pattern
    ]
    
    @classmethod
    def hash_sensitive_value(cls, value: str, salt: str = "genomevault_audit") -> str:
        """Create a consistent hash of sensitive values for audit purposes."""
        if not value:
            return "null"
        
        # Use HMAC for consistent hashing
        return hmac.new(
            salt.encode(), 
            str(value).encode(), 
            hashlib.sha256
        ).hexdigest()[:16]  # Truncate for readability
    
    @classmethod
    def sanitize_field_name(cls, field_name: str) -> bool:
        """Check if a field name indicates PHI."""
        field_lower = field_name.lower()
        return any(phi_field in field_lower for phi_field in cls.PHI_FIELDS)
    
    @classmethod
    def sanitize_data_for_audit(cls, data: Any, context: str = "") -> Any:
        """Sanitize data to remove PHI before audit logging."""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if cls.sanitize_field_name(key):
                    # Hash PHI field values
                    sanitized[key] = cls.hash_sensitive_value(str(value))
                else:
                    # Recursively sanitize nested structures
                    sanitized[key] = cls.sanitize_data_for_audit(value, f"{context}.{key}")
            return sanitized
        
        elif isinstance(data, list):
            return [cls.sanitize_data_for_audit(item, f"{context}[{i}]") for i, item in enumerate(data)]
        
        elif isinstance(data, str):
            # Check for patterns that might be PHI
            for pattern in cls.PHI_PATTERNS:
                import re
                if re.search(pattern, data):
                    return cls.hash_sensitive_value(data)
            
            # If string is very long, it might contain sensitive data
            if len(data) > 1000:
                return f"<large_string:{len(data)}_chars:{cls.hash_sensitive_value(data)}>"
            
            return data
        
        else:
            return data


class AuditLogger:
    """Main audit logging engine."""
    
    def __init__(self, 
                 log_file_path: Optional[str] = None,
                 signing_key: Optional[str] = None,
                 enable_console_output: bool = True,
                 max_log_size_mb: int = 100):
        """Initialize audit logger.
        
        Args:
            log_file_path: Path to audit log file
            signing_key: Key for HMAC signing of audit entries
            enable_console_output: Whether to also log to console
            max_log_size_mb: Maximum size of log file before rotation
        """
        self.log_file_path = log_file_path
        self.signing_key = signing_key or "default_audit_key"  # Should be from secure config
        self.enable_console_output = enable_console_output
        self.max_log_size_mb = max_log_size_mb
        self.phi_protector = PHIDataProtector()
        
        # Initialize log file if specified
        if self.log_file_path:
            self._ensure_log_file_exists()
    
    def _ensure_log_file_exists(self):
        """Ensure audit log file and directory exist."""
        if self.log_file_path:
            log_path = Path(self.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file if it doesn't exist
            if not log_path.exists():
                log_path.touch(mode=0o600)  # Restricted permissions
    
    def _sign_event(self, event_data: Dict[str, Any]) -> str:
        """Create HMAC signature for audit event."""
        # Create canonical representation for signing
        canonical_data = json.dumps(event_data, sort_keys=True, separators=(',', ':'))
        
        return hmac.new(
            self.signing_key.encode(),
            canonical_data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def log_event(self, 
                  event_type: AuditEventType,
                  severity: AuditSeverity = AuditSeverity.MEDIUM,
                  actor_id: Optional[str] = None,
                  actor_type: Optional[str] = None,
                  resource_type: Optional[str] = None,
                  resource_id: Optional[str] = None,
                  action: Optional[str] = None,
                  endpoint: Optional[str] = None,
                  request_id: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None,
                  risk_indicators: Optional[List[str]] = None,
                  **kwargs) -> str:
        """Log an audit event.
        
        Returns:
            Event ID for correlation
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Sanitize details to remove PHI
        sanitized_details = None
        if details:
            sanitized_details = self.phi_protector.sanitize_data_for_audit(details)
        
        # Create audit event
        event = AuditEvent(
            event_id=event_id,
            timestamp=timestamp,
            event_type=event_type,
            severity=severity,
            actor_id=actor_id,
            actor_type=actor_type,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            endpoint=endpoint,
            request_id=request_id,
            details=sanitized_details,
            risk_indicators=risk_indicators,
            **kwargs
        )
        
        # Convert to dict and sign
        event_data = event.to_dict()
        event_data['signature'] = self._sign_event(event_data)
        
        # Write to log file
        if self.log_file_path:
            self._write_to_file(event_data)
        
        # Write to console if enabled
        if self.enable_console_output:
            self._write_to_console(event_data)
        
        return event_id
    
    def _write_to_file(self, event_data: Dict[str, Any]):
        """Write audit event to file."""
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event_data, separators=(',', ':')) + '\n')
                f.flush()  # Ensure immediate write
        except Exception as e:
            # Failsafe: if file logging fails, at least log to console
            print(f"Audit log file write failed: {e}")
            if not self.enable_console_output:
                self._write_to_console(event_data)
    
    def _write_to_console(self, event_data: Dict[str, Any]):
        """Write audit event to console."""
        print(f"AUDIT: {json.dumps(event_data, separators=(',', ':'))}")
    
    def log_request(self,
                    request: Request,
                    response: Optional[Response] = None,
                    duration_ms: Optional[float] = None,
                    event_type: AuditEventType = AuditEventType.DATA_READ):
        """Log a complete request/response cycle."""
        
        # Extract actor information
        actor_id = None
        actor_type = None
        api_key_info = getattr(request.state, 'api_key_info', None)
        if api_key_info:
            actor_id = self.phi_protector.hash_sensitive_value(api_key_info.key_id)
            actor_type = "api_key"
        
        # Extract client information (sanitized)
        client_info = {
            'user_agent_hash': self.phi_protector.hash_sensitive_value(
                request.headers.get('user-agent', 'unknown')
            ),
            'method': request.method,
            'protocol': request.url.scheme
        }
        
        # Determine resource type from endpoint
        resource_type = self._determine_resource_type(request.url.path)
        
        # Get request details
        request_id = getattr(request.state, 'request_id', None)
        
        # Log the event
        self.log_event(
            event_type=event_type,
            severity=self._determine_severity(request.url.path, response),
            actor_id=actor_id,
            actor_type=actor_type,
            resource_type=resource_type,
            action=request.method,
            endpoint=request.url.path,
            request_id=request_id,
            client_info=client_info,
            response_code=response.status_code if response else None,
            duration_ms=duration_ms,
            details={
                'query_params_count': len(request.query_params),
                'has_body': 'content-length' in request.headers
            }
        )
    
    def _determine_resource_type(self, path: str) -> str:
        """Determine resource type from endpoint path."""
        path_lower = path.lower()
        
        if any(pattern in path_lower for pattern in ['/clinical/', '/patient/', '/phi/']):
            return "clinical_data"
        elif any(pattern in path_lower for pattern in ['/genomic/', '/variant/', '/sequence/']):
            return "genomic_data"
        elif any(pattern in path_lower for pattern in ['/hdc/', '/zk/', '/pir/']):
            return "compute_service"
        elif any(pattern in path_lower for pattern in ['/admin/', '/config/', '/keys/']):
            return "system_config"
        else:
            return "api_resource"
    
    def _determine_severity(self, path: str, response: Optional[Response]) -> AuditSeverity:
        """Determine event severity based on endpoint and response."""
        path_lower = path.lower()
        
        # Critical for PHI access
        if any(pattern in path_lower for pattern in ['/clinical/', '/patient/', '/phi/']):
            return AuditSeverity.CRITICAL
        
        # High for sensitive operations
        if any(pattern in path_lower for pattern in ['/admin/', '/keys/', '/users/']):
            return AuditSeverity.HIGH
        
        # High for failed requests (potential attacks)
        if response and response.status_code >= 400:
            return AuditSeverity.HIGH
        
        # Medium for compute operations
        if any(pattern in path_lower for pattern in ['/hdc/', '/zk/', '/pir/']):
            return AuditSeverity.MEDIUM
        
        return AuditSeverity.LOW


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware for automatic audit logging of all requests."""
    
    def __init__(self, app, audit_logger: Optional[AuditLogger] = None):
        """Initialize audit middleware.
        
        Args:
            app: FastAPI application
            audit_logger: Custom audit logger instance
        """
        super().__init__(app)
        
        # Initialize default audit logger if not provided
        if audit_logger is None:
            import os
            log_file = os.getenv('GENOMEVAULT_AUDIT_LOG_FILE', '/var/log/genomevault/audit.log')
            signing_key = os.getenv('GENOMEVAULT_AUDIT_SIGNING_KEY', 'default_audit_key')
            
            audit_logger = AuditLogger(
                log_file_path=log_file,
                signing_key=signing_key,
                enable_console_output=True
            )
        
        self.audit_logger = audit_logger
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Audit all requests and responses."""
        start_time = time.time()
        
        # Skip auditing for certain endpoints to reduce log volume
        if self._should_skip_audit(request):
            return await call_next(request)
        
        response = None
        try:
            # Process the request
            response = await call_next(request)
            
            # Determine event type based on method and response
            event_type = self._get_event_type(request, response)
            
        except Exception as e:
            # Log failed requests
            self.audit_logger.log_event(
                event_type=AuditEventType.ERROR_OCCURRED,
                severity=AuditSeverity.HIGH,
                endpoint=request.url.path,
                action=request.method,
                details={'error_type': type(e).__name__},
                risk_indicators=['unhandled_exception']
            )
            raise
        
        finally:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log the complete request
            if response:
                self.audit_logger.log_request(
                    request=request,
                    response=response,
                    duration_ms=duration_ms,
                    event_type=self._get_event_type(request, response)
                )
        
        return response
    
    def _should_skip_audit(self, request: Request) -> bool:
        """Check if this request should be skipped from audit logging."""
        path = request.url.path.lower()
        
        # Skip health checks and metrics to reduce log volume
        skip_paths = ['/health', '/ping', '/metrics', '/ready']
        return any(skip_path in path for skip_path in skip_paths)
    
    def _get_event_type(self, request: Request, response: Response) -> AuditEventType:
        """Determine audit event type based on request/response."""
        path = request.url.path.lower()
        method = request.method.upper()
        
        # Authentication events
        if response.status_code == 401:
            return AuditEventType.AUTH_FAILURE
        elif response.status_code == 403:
            return AuditEventType.AUTHZ_ACCESS_DENIED
        elif response.status_code == 429:
            return AuditEventType.RATE_LIMIT_EXCEEDED
        
        # PHI access events
        if any(pattern in path for pattern in ['/clinical/', '/patient/', '/phi/']):
            return AuditEventType.PHI_ACCESS
        
        # Data access events
        if method in ['GET', 'HEAD']:
            return AuditEventType.DATA_READ
        elif method in ['POST', 'PUT', 'PATCH']:
            return AuditEventType.DATA_WRITE
        elif method == 'DELETE':
            return AuditEventType.DATA_DELETE
        
        return AuditEventType.DATA_READ


# Global audit logger instance
_global_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _global_audit_logger
    
    if _global_audit_logger is None:
        import os
        log_file = os.getenv('GENOMEVAULT_AUDIT_LOG_FILE')
        signing_key = os.getenv('GENOMEVAULT_AUDIT_SIGNING_KEY', 'default_key')
        
        _global_audit_logger = AuditLogger(
            log_file_path=log_file,
            signing_key=signing_key,
            enable_console_output=True
        )
    
    return _global_audit_logger


# Convenience functions for common audit events
def audit_authentication(success: bool, actor_id: str, details: Optional[Dict] = None):
    """Audit authentication attempt."""
    logger = get_audit_logger()
    event_type = AuditEventType.AUTH_SUCCESS if success else AuditEventType.AUTH_FAILURE
    severity = AuditSeverity.MEDIUM if success else AuditSeverity.HIGH
    
    logger.log_event(
        event_type=event_type,
        severity=severity,
        actor_id=actor_id,
        actor_type="api_key",
        details=details,
        risk_indicators=[] if success else ["auth_failure"]
    )


def audit_phi_access(actor_id: str, resource_id: str, action: str, details: Optional[Dict] = None):
    """Audit PHI data access (highest severity)."""
    logger = get_audit_logger()
    
    logger.log_event(
        event_type=AuditEventType.PHI_ACCESS,
        severity=AuditSeverity.CRITICAL,
        actor_id=actor_id,
        resource_type="clinical_data",
        resource_id=resource_id,
        action=action,
        details=details
    )


def audit_genomic_analysis(analysis_type: str, actor_id: str, details: Optional[Dict] = None):
    """Audit genomic analysis operations."""
    logger = get_audit_logger()
    
    logger.log_event(
        event_type=AuditEventType.GENOMIC_ANALYSIS_START,
        severity=AuditSeverity.MEDIUM,
        actor_id=actor_id,
        resource_type="genomic_data",
        action=analysis_type,
        details=details
    )