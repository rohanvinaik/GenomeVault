"""
GenomeVault Logging System

Provides standardized logging with privacy-aware features, structured logging,
and integration with monitoring systems.
"""

import logging
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from enum import Enum
import hashlib
import re
from contextlib import contextmanager
from functools import wraps
import threading
import queue


class PrivacyLevel(Enum):
    """Privacy levels for log redaction"""
    PUBLIC = 0      # No redaction needed
    INTERNAL = 1    # Redact in production logs
    SENSITIVE = 2   # Always redact, hash if needed
    SECRET = 3      # Never log, even hashed


class LogEvent(Enum):
    """Standardized log events for monitoring"""
    # System events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    
    # Processing events
    PROCESSING_START = "processing.start"
    PROCESSING_COMPLETE = "processing.complete"
    PROCESSING_ERROR = "processing.error"
    
    # Privacy events
    PRIVACY_QUERY = "privacy.query"
    PRIVACY_VIOLATION = "privacy.violation"
    PRIVACY_BUDGET_EXCEEDED = "privacy.budget_exceeded"
    
    # Security events
    SECURITY_AUTH_SUCCESS = "security.auth.success"
    SECURITY_AUTH_FAILURE = "security.auth.failure"
    SECURITY_AUDIT = "security.audit"
    
    # Blockchain events
    BLOCKCHAIN_TRANSACTION = "blockchain.transaction"
    BLOCKCHAIN_BLOCK = "blockchain.block"
    BLOCKCHAIN_CONSENSUS = "blockchain.consensus"
    
    # Network events
    NETWORK_PIR_QUERY = "network.pir.query"
    NETWORK_PIR_RESPONSE = "network.pir.response"
    NETWORK_CONNECTION = "network.connection"


class PrivacyFilter(logging.Filter):
    """Filter to redact sensitive information from logs"""
    
    # Patterns to redact
    PATTERNS = [
        (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[CREDIT_CARD]'),  # Credit cards
        (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),                # SSN
        (r'\b[A-Z]{2}\d{8}\b', '[ID]'),                     # ID numbers
        (r'chr\d+:\d+-\d+', '[GENOMIC_REGION]'),            # Genomic regions
        (r'rs\d+', '[SNP_ID]'),                             # SNP IDs
        (r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}', '[EMAIL]'),  # Emails
        (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP]'),          # IP addresses
    ]
    
    def __init__(self, privacy_mode: bool = True):
        super().__init__()
        self.privacy_mode = privacy_mode
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and redact log records"""
        if self.privacy_mode and hasattr(record, 'msg'):
            record.msg = self._redact_message(str(record.msg))
            if hasattr(record, 'args') and record.args:
                record.args = tuple(self._redact_message(str(arg)) for arg in record.args)
        return True
    
    def _redact_message(self, message: str) -> str:
        """Redact sensitive patterns from message"""
        for pattern, replacement in self.PATTERNS:
            message = re.sub(pattern, replacement, message)
        return message


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename',
                          'funcName', 'levelname', 'levelno', 'lineno',
                          'module', 'msecs', 'pathname', 'process',
                          'processName', 'relativeCreated', 'thread',
                          'threadName', 'exc_info', 'exc_text', 'stack_info']:
                log_data[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class MetricsLogger:
    """Logger for metrics and performance data"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._metrics_queue = queue.Queue()
        self._metrics_thread = threading.Thread(target=self._process_metrics, daemon=True)
        self._metrics_thread.start()
    
    def log_metric(self, name: str, value: Union[int, float], 
                   tags: Optional[Dict[str, str]] = None):
        """Log a metric value"""
        self._metrics_queue.put({
            'timestamp': datetime.utcnow(),
            'name': name,
            'value': value,
            'tags': tags or {}
        })
    
    def _process_metrics(self):
        """Process metrics queue"""
        batch = []
        while True:
            try:
                metric = self._metrics_queue.get(timeout=1)
                batch.append(metric)
                
                # Batch metrics for efficiency
                if len(batch) >= 100 or self._metrics_queue.empty():
                    self._send_metrics_batch(batch)
                    batch = []
            except queue.Empty:
                if batch:
                    self._send_metrics_batch(batch)
                    batch = []
    
    def _send_metrics_batch(self, batch: List[Dict[str, Any]]):
        """Send batch of metrics"""
        self.logger.info("metrics.batch", extra={
            'metrics': batch,
            'event': LogEvent.SYSTEM_METRIC.value
        })


class AuditLogger:
    """Specialized logger for audit trails"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_access(self, user_id: str, resource: str, action: str, 
                   result: str, metadata: Optional[Dict[str, Any]] = None):
        """Log data access for audit trail"""
        self.logger.info("audit.access", extra={
            'event': LogEvent.SECURITY_AUDIT.value,
            'user_id': self._hash_if_needed(user_id),
            'resource': resource,
            'action': action,
            'result': result,
            'metadata': metadata or {},
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def log_computation(self, user_id: str, operation: str, 
                       privacy_budget_used: float, metadata: Optional[Dict[str, Any]] = None):
        """Log privacy-sensitive computation"""
        self.logger.info("audit.computation", extra={
            'event': LogEvent.PRIVACY_QUERY.value,
            'user_id': self._hash_if_needed(user_id),
            'operation': operation,
            'privacy_budget_used': privacy_budget_used,
            'metadata': metadata or {},
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def _hash_if_needed(self, value: str) -> str:
        """Hash sensitive values for audit logs"""
        # In production, always hash user IDs
        return hashlib.sha256(value.encode()).hexdigest()[:16]


class GenomeVaultLogger:
    """Main logger class for GenomeVault"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GenomeVault logger
        
        Args:
            name: Logger name
            config: Logger configuration
        """
        self.logger = logging.getLogger(name)
        self.config = config or {}
        
        # Set up logging
        self._setup_handlers()
        self._setup_filters()
        
        # Initialize specialized loggers
        self.metrics = MetricsLogger(self.logger)
        self.audit = AuditLogger(self.logger)
    
    def _setup_handlers(self):
        """Set up log handlers"""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(
            getattr(logging, self.config.get('console_level', 'INFO'))
        )
        
        # File handler
        log_dir = Path(self.config.get('log_dir', Path.home() / '.genomevault' / 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"{self.logger.name.replace('.', '_')}.log"
        )
        file_handler.setLevel(
            getattr(logging, self.config.get('file_level', 'DEBUG'))
        )
        
        # Set formatters
        if self.config.get('structured_logs', True):
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Set overall level
        self.logger.setLevel(
            getattr(logging, self.config.get('level', 'INFO'))
        )
    
    def _setup_filters(self):
        """Set up log filters"""
        # Add privacy filter
        privacy_filter = PrivacyFilter(
            privacy_mode=self.config.get('privacy_mode', True)
        )
        for handler in self.logger.handlers:
            handler.addFilter(privacy_filter)
    
    @contextmanager
    def operation_context(self, operation: str, **kwargs):
        """Context manager for operation logging"""
        start_time = datetime.utcnow()
        operation_id = hashlib.md5(
            f"{operation}{start_time}".encode()
        ).hexdigest()[:8]
        
        self.logger.info(f"Operation started: {operation}", extra={
            'operation': operation,
            'operation_id': operation_id,
            'event': LogEvent.PROCESSING_START.value,
            **kwargs
        })
        
        try:
            yield operation_id
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"Operation completed: {operation}", extra={
                'operation': operation,
                'operation_id': operation_id,
                'duration_seconds': duration,
                'event': LogEvent.PROCESSING_COMPLETE.value,
                'status': 'success',
                **kwargs
            })
            
            # Log metrics
            self.metrics.log_metric(
                f"operation.duration.{operation}",
                duration,
                tags={'status': 'success'}
            )
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"Operation failed: {operation}", extra={
                'operation': operation,
                'operation_id': operation_id,
                'duration_seconds': duration,
                'event': LogEvent.PROCESSING_ERROR.value,
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__,
                **kwargs
            }, exc_info=True)
            
            # Log metrics
            self.metrics.log_metric(
                f"operation.duration.{operation}",
                duration,
                tags={'status': 'error', 'error_type': type(e).__name__}
            )
            
            raise
    
    def log_event(self, event: LogEvent, message: str, **kwargs):
        """Log a standardized event"""
        self.logger.info(message, extra={
            'event': event.value,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        })
    
    def __getattr__(self, name):
        """Delegate to underlying logger"""
        return getattr(self.logger, name)


# Global logger registry
_loggers: Dict[str, GenomeVaultLogger] = {}


def get_logger(name: str, config: Optional[Dict[str, Any]] = None) -> GenomeVaultLogger:
    """Get or create a GenomeVault logger"""
    if name not in _loggers:
        _loggers[name] = GenomeVaultLogger(name, config)
    return _loggers[name]


def configure_logging(config: Dict[str, Any]):
    """Configure global logging settings"""
    # Set root logger level
    logging.root.setLevel(
        getattr(logging, config.get('root_level', 'WARNING'))
    )
    
    # Configure library loggers
    for lib_name, lib_level in config.get('library_levels', {}).items():
        logging.getLogger(lib_name).setLevel(
            getattr(logging, lib_level)
        )


def log_operation(operation: str):
    """Decorator for automatic operation logging"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            with logger.operation_context(operation, function=func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Convenience function for privacy-aware logging
def log_genomic_operation(logger: GenomeVaultLogger, operation: str, 
                         user_id: str, privacy_budget: float = 0.0,
                         **metadata):
    """Log a genomic operation with privacy considerations"""
    logger.audit.log_computation(
        user_id=user_id,
        operation=operation,
        privacy_budget_used=privacy_budget,
        metadata=metadata
    )
