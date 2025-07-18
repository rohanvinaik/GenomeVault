"""
Standardized logging setup for GenomeVault.
Provides structured logging with privacy-aware filtering.
"""
import logging
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
from functools import wraps
import hashlib


class PrivacyFilter(logging.Filter):
    """Filter to prevent logging of sensitive genomic data."""
    
    SENSITIVE_PATTERNS = [
        'variant', 'genotype', 'allele', 'mutation',
        'chr', 'position', 'sequence', 'read',
        'phenotype', 'diagnosis', 'patient'
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter out log records containing sensitive information.
        
        Args:
            record: Log record to filter
            
        Returns:
            True if record should be logged, False otherwise
        """
        # Check message for sensitive patterns
        message = str(record.getMessage()).lower()
        
        # Allow structured logs with privacy flags
        if hasattr(record, 'privacy_safe') and record.privacy_safe:
            return True
        
        # Check for sensitive patterns
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern in message and not hasattr(record, 'privacy_reviewed'):
                # Log a sanitized version instead
                record.msg = f"[REDACTED: Contains {pattern} data]"
                return True
        
        return True


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 
                          'funcName', 'levelname', 'levelno', 'lineno', 
                          'module', 'msecs', 'message', 'pathname', 'process',
                          'processName', 'relativeCreated', 'thread', 'threadName']:
                log_data[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class AuditLogger:
    """Specialized logger for audit trail with cryptographic verification."""
    
    def __init__(self, log_path: Path):
        """
        Initialize audit logger.
        
        Args:
            log_path: Path to audit log file
        """
        self.log_path = log_path
        self.logger = logging.getLogger('genomevault.audit')
        
        # Create file handler with rotation
        handler = logging.handlers.RotatingFileHandler(
            self.log_path,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10
        )
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Track log chain
        self.previous_hash = self._get_last_hash()
    
    def _get_last_hash(self) -> str:
        """Get hash of last log entry for chain verification."""
        if not self.log_path.exists():
            return "0" * 64  # Genesis hash
        
        # Read last line and extract hash
        with open(self.log_path, 'rb') as f:
            # Seek to end and read backwards to find last complete line
            f.seek(0, 2)  # End of file
            file_size = f.tell()
            
            if file_size == 0:
                return "0" * 64
            
            # Read last 4KB (should contain last line)
            read_size = min(file_size, 4096)
            f.seek(file_size - read_size)
            data = f.read()
            
            # Find last complete line
            lines = data.splitlines()
            if lines:
                try:
                    last_entry = json.loads(lines[-1])
                    return last_entry.get('chain_hash', "0" * 64)
                except:
                    return "0" * 64
        
        return "0" * 64
    
    def log_event(self, event_type: str, actor: str, action: str, 
                  resource: Optional[str] = None, result: str = "success",
                  metadata: Optional[Dict[str, Any]] = None):
        """
        Log an audit event with cryptographic chaining.
        
        Args:
            event_type: Type of event (e.g., 'data_access', 'proof_generation')
            actor: Identity performing the action
            action: Action being performed
            resource: Resource being acted upon
            result: Result of the action
            metadata: Additional metadata
        """
        # Create audit entry
        entry = {
            'event_type': event_type,
            'actor': actor,
            'action': action,
            'resource': resource,
            'result': result,
            'metadata': metadata or {},
            'previous_hash': self.previous_hash
        }
        
        # Calculate entry hash
        entry_str = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry['chain_hash'] = entry_hash
        
        # Log entry
        self.logger.info(entry['action'], extra=entry)
        
        # Update chain
        self.previous_hash = entry_hash


class PerformanceLogger:
    """Logger for performance metrics and profiling."""
    
    def __init__(self):
        """Initialize performance logger."""
        self.logger = logging.getLogger('genomevault.performance')
        self.logger.setLevel(logging.INFO)
    
    def log_operation(self, operation: str):
        """
        Decorator to log operation performance.
        
        Args:
            operation: Name of the operation
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                import time
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    self.logger.info(
                        f"Operation completed",
                        extra={
                            'operation': operation,
                            'duration_seconds': duration,
                            'status': 'success',
                            'privacy_safe': True
                        }
                    )
                    
                    return result
                
                except Exception as e:
                    duration = time.time() - start_time
                    
                    self.logger.error(
                        f"Operation failed",
                        extra={
                            'operation': operation,
                            'duration_seconds': duration,
                            'status': 'failure',
                            'error': str(e),
                            'privacy_safe': True
                        },
                        exc_info=True
                    )
                    
                    raise
            
            return wrapper
        return decorator


def setup_logging(
    log_level: Union[str, int] = logging.INFO,
    log_format: str = "structured",
    enable_console: bool = True,
    enable_file: bool = True,
    log_dir: Optional[Path] = None,
    enable_privacy_filter: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for GenomeVault.
    
    Args:
        log_level: Logging level
        log_format: Format type ('structured' or 'simple')
        enable_console: Enable console output
        enable_file: Enable file output
        log_dir: Directory for log files
        enable_privacy_filter: Enable privacy filtering
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('genomevault')
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Set up formatters
    if log_format == "structured":
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Add privacy filter
    if enable_privacy_filter:
        privacy_filter = PrivacyFilter()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        if enable_privacy_filter:
            console_handler.addFilter(privacy_filter)
        logger.addHandler(console_handler)
    
    # File handler
    if enable_file:
        if log_dir is None:
            log_dir = Path.home() / '.genomevault' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'genomevault.log',
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        if enable_privacy_filter:
            file_handler.addFilter(privacy_filter)
        logger.addHandler(file_handler)
    
    # Log startup
    logger.info(
        "GenomeVault logging initialized",
        extra={
            'log_level': logging.getLevelName(log_level),
            'log_format': log_format,
            'privacy_filter': enable_privacy_filter,
            'privacy_safe': True
        }
    )
    
    return logger


# Global logger instance
logger = setup_logging()

# Specialized loggers
audit_logger = AuditLogger(Path.home() / '.genomevault' / 'logs' / 'audit.log')
performance_logger = PerformanceLogger()


# Convenience functions
def log_data_access(actor: str, resource: str, action: str = "read", 
                   success: bool = True, metadata: Optional[Dict] = None):
    """Log data access event."""
    audit_logger.log_event(
        event_type="data_access",
        actor=actor,
        action=action,
        resource=resource,
        result="success" if success else "failure",
        metadata=metadata
    )


def log_proof_generation(actor: str, circuit_type: str, proof_id: str,
                        success: bool = True, metadata: Optional[Dict] = None):
    """Log proof generation event."""
    audit_logger.log_event(
        event_type="proof_generation",
        actor=actor,
        action=f"generate_{circuit_type}_proof",
        resource=proof_id,
        result="success" if success else "failure",
        metadata=metadata
    )


def log_consent_update(actor: str, consent_type: str, granted: bool,
                      metadata: Optional[Dict] = None):
    """Log consent update event."""
    audit_logger.log_event(
        event_type="consent_update",
        actor=actor,
        action=f"{'grant' if granted else 'revoke'}_{consent_type}_consent",
        result="success",
        metadata=metadata
    )


# Example usage
if __name__ == "__main__":
    # Test logging
    logger.info("System startup", extra={'privacy_safe': True})
    logger.warning("This contains variant data that should be filtered")
    
    # Test audit logging
    log_data_access(
        actor="user123",
        resource="genome_profile_456",
        action="read",
        metadata={'purpose': 'analysis'}
    )
    
    # Test performance logging
    @performance_logger.log_operation("test_operation")
    def example_operation():
        import time
        time.sleep(0.1)
        return "completed"
    
    result = example_operation()
    print(f"Operation result: {result}")
