"""
Structured logging system for GenomeVault.

Provides JSON structured logging with correlation IDs, privacy-aware field filtering,
and contextual information for genomic operations. Integrates with OpenTelemetry
for distributed tracing correlation.
"""

import json
import logging
import os
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Any, Optional, Union
from datetime import datetime
import threading

try:
    from opentelemetry import trace
    from opentelemetry.trace import format_trace_id, format_span_id
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# Context variables for correlation IDs and metadata
request_id_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_context: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
operation_context: ContextVar[Optional[str]] = ContextVar('operation', default=None)
privacy_context: ContextVar[Optional[str]] = ContextVar('privacy_level', default=None)

# Thread-local storage for fallback when contextvars aren't available
_local = threading.local()


class PrivacyAwareFilter:
    """Filter that redacts or removes sensitive fields from log records."""
    
    # Fields that should be completely removed
    SENSITIVE_FIELDS = {
        'password', 'token', 'secret', 'key', 'auth', 'credential',
        'ssn', 'social_security', 'patient_id', 'sample_id', 'genomic_data',
        'variant_data', 'sequence_data', 'clinical_data', 'phi_data'
    }
    
    # Fields that should be partially redacted (show first/last chars)
    REDACT_FIELDS = {
        'email', 'phone', 'address', 'name', 'user_id', 'client_id'
    }
    
    @classmethod
    def filter_sensitive_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive data from log records."""
        if not isinstance(data, dict):
            return data
            
        filtered = {}
        for key, value in data.items():
            key_lower = key.lower()
            
            # Remove sensitive fields completely
            if any(sensitive in key_lower for sensitive in cls.SENSITIVE_FIELDS):
                filtered[key] = "[REDACTED]"
            # Partially redact certain fields
            elif any(redact in key_lower for redact in cls.REDACT_FIELDS):
                if isinstance(value, str) and len(value) > 4:
                    filtered[key] = f"{value[:2]}***{value[-2:]}"
                else:
                    filtered[key] = "[REDACTED]"
            # Recursively filter nested dictionaries
            elif isinstance(value, dict):
                filtered[key] = cls.filter_sensitive_data(value)
            else:
                filtered[key] = value
                
        return filtered


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def __init__(self, include_trace_info: bool = True):
        """Initialize formatter.
        
        Args:
            include_trace_info: Whether to include OpenTelemetry trace information
        """
        super().__init__()
        self.include_trace_info = include_trace_info and OTEL_AVAILABLE
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base log structure
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation IDs from context
        request_id = request_id_context.get()
        if not request_id:
            # Fallback to thread-local storage
            request_id = getattr(_local, 'request_id', None)
        if request_id:
            log_entry["request_id"] = request_id
            
        user_id = user_id_context.get()
        if not user_id:
            user_id = getattr(_local, 'user_id', None)
        if user_id:
            log_entry["user_id"] = user_id
            
        operation = operation_context.get()
        if not operation:
            operation = getattr(_local, 'operation', None)
        if operation:
            log_entry["operation"] = operation
            
        privacy_level = privacy_context.get()
        if not privacy_level:
            privacy_level = getattr(_local, 'privacy_level', None)
        if privacy_level:
            log_entry["privacy_level"] = privacy_level
        
        # Add OpenTelemetry trace information
        if self.include_trace_info:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                span_context = current_span.get_span_context()
                if span_context.is_valid:
                    log_entry["trace_id"] = format_trace_id(span_context.trace_id)
                    log_entry["span_id"] = format_span_id(span_context.span_id)
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields from the log record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated', 
                          'thread', 'threadName', 'processName', 'process', 'getMessage'}:
                extra_fields[key] = value
        
        if extra_fields:
            # Filter sensitive data from extra fields
            filtered_extra = PrivacyAwareFilter.filter_sensitive_data(extra_fields)
            log_entry["extra"] = filtered_extra
        
        return json.dumps(log_entry, default=str, separators=(',', ':'))


class GenomeVaultLogger:
    """Enhanced logger for GenomeVault with structured logging capabilities."""
    
    def __init__(self, name: str, level: str = "INFO"):
        """Initialize logger.
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self._setup_logger(level)
    
    def _setup_logger(self, level: str):
        """Set up the logger with structured formatting."""
        if not self.logger.handlers:
            self.logger.setLevel(getattr(logging, level.upper()))
            
            # Create handler
            handler = logging.StreamHandler(sys.stdout)
            
            # Use structured formatter if JSON logging is enabled
            use_json = os.getenv("GENOMEVAULT_JSON_LOGGING", "true").lower() == "true"
            if use_json:
                formatter = StructuredFormatter()
            else:
                formatter = logging.Formatter(
                    fmt="%(asctime)s %(levelname)s [%(request_id)s] %(name)s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S%z"
                )
            
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False
    
    def set_context(self, 
                   request_id: Optional[str] = None,
                   user_id: Optional[str] = None,
                   operation: Optional[str] = None,
                   privacy_level: Optional[str] = None):
        """Set logging context for correlation."""
        if request_id:
            request_id_context.set(request_id)
            _local.request_id = request_id
        if user_id:
            user_id_context.set(user_id)
            _local.user_id = user_id
        if operation:
            operation_context.set(operation)
            _local.operation = operation
        if privacy_level:
            privacy_context.set(privacy_level)
            _local.privacy_level = privacy_level
    
    @contextmanager
    def context(self, 
                request_id: Optional[str] = None,
                user_id: Optional[str] = None,
                operation: Optional[str] = None,
                privacy_level: Optional[str] = None):
        """Context manager for temporary logging context."""
        # Store current context
        old_request_id = request_id_context.get()
        old_user_id = user_id_context.get()
        old_operation = operation_context.get()
        old_privacy_level = privacy_context.get()
        
        try:
            # Set new context
            if request_id:
                request_id_context.set(request_id)
            if user_id:
                user_id_context.set(user_id)
            if operation:
                operation_context.set(operation)
            if privacy_level:
                privacy_context.set(privacy_level)
            
            yield
            
        finally:
            # Restore old context
            if old_request_id:
                request_id_context.set(old_request_id)
            if old_user_id:
                user_id_context.set(old_user_id)
            if old_operation:
                operation_context.set(old_operation)
            if old_privacy_level:
                privacy_context.set(old_privacy_level)
    
    def info(self, message: str, **kwargs):
        """Log info message with extra context."""
        self.logger.info(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with extra context."""
        self.logger.error(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with extra context."""
        self.logger.warning(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with extra context."""
        self.logger.debug(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with extra context."""
        self.logger.critical(message, extra=kwargs)
    
    def log_hdc_operation(self, operation: str, dimension: int, duration: float, **kwargs):
        """Log HDC operation with standardized fields."""
        self.info(f"HDC {operation} completed", 
                 hdc_operation=operation,
                 hdc_dimension=dimension,
                 duration_seconds=duration,
                 **kwargs)
    
    def log_zk_operation(self, operation: str, circuit_type: str, duration: float, **kwargs):
        """Log ZK proof operation with standardized fields."""
        self.info(f"ZK {operation} completed",
                 zk_operation=operation,
                 zk_circuit_type=circuit_type,
                 duration_seconds=duration,
                 **kwargs)
    
    def log_pir_operation(self, operation: str, database_size: int, duration: float, **kwargs):
        """Log PIR operation with standardized fields."""
        self.info(f"PIR {operation} completed",
                 pir_operation=operation,
                 pir_database_size=database_size,
                 duration_seconds=duration,
                 **kwargs)
    
    def log_api_request(self, method: str, path: str, status_code: int, duration: float, **kwargs):
        """Log API request with standardized fields."""
        self.info(f"{method} {path} {status_code}",
                 http_method=method,
                 http_path=path,
                 http_status_code=status_code,
                 duration_seconds=duration,
                 **kwargs)


def get_structured_logger(name: str) -> GenomeVaultLogger:
    """Get a structured logger instance."""
    level = os.getenv("GENOMEVAULT_LOG_LEVEL", "INFO").upper()
    return GenomeVaultLogger(name, level)


def configure_structured_logging():
    """Configure structured logging for the entire application."""
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove default handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add structured handler
    handler = logging.StreamHandler(sys.stdout)
    formatter = StructuredFormatter()
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Configure specific loggers to use structured logging
    for logger_name in ['genomevault', 'uvicorn', 'fastapi']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            logger.addHandler(handler)
        logger.propagate = False


# Convenience functions for setting global context
def set_request_context(request_id: str, user_id: Optional[str] = None):
    """Set request context globally."""
    request_id_context.set(request_id)
    _local.request_id = request_id
    if user_id:
        user_id_context.set(user_id)
        _local.user_id = user_id


def generate_request_id() -> str:
    """Generate a new request ID."""
    return str(uuid.uuid4())


def get_request_id() -> Optional[str]:
    """Get current request ID from context."""
    request_id = request_id_context.get()
    if not request_id:
        request_id = getattr(_local, 'request_id', None)
    return request_id