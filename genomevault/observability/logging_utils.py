"""
Structured logging utilities with PHI leak detection.

Implements JSON structured logging with request ID correlation,
PHI detection and redaction, and audit trail generation.
"""

import json
import logging
import re
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, List, Set, Callable
from uuid import uuid4
import hashlib

from pythonjsonlogger import jsonlogger


class LogLevel(Enum):
    """Log levels with numeric values."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class PHICategory(Enum):
    """Categories of Protected Health Information."""
    NAME = "name"
    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    MRN = "medical_record_number"
    DOB = "date_of_birth"
    ADDRESS = "address"
    GENOMIC = "genomic_data"
    DIAGNOSIS = "diagnosis"
    MEDICATION = "medication"


@dataclass
class PHIDetection:
    """PHI detection result."""
    category: PHICategory
    pattern: str
    confidence: float
    position: tuple  # (start, end)
    redacted_value: str


class PHILeakageDetector:
    """
    Detector for PHI leakage in logs.
    
    Scans log messages for potential PHI and redacts sensitive information.
    """
    
    def __init__(self, sensitivity: str = "high"):
        """
        Initialize PHI detector.
        
        Args:
            sensitivity: Detection sensitivity (low, medium, high)
        """
        self.sensitivity = sensitivity
        self.patterns = self._load_patterns()
        self.whitelist: Set[str] = set()
        self.detection_stats = {
            'total_scanned': 0,
            'phi_detected': 0,
            'redactions': 0,
            'false_positives': 0
        }
        
    def _load_patterns(self) -> Dict[PHICategory, List[re.Pattern]]:
        """Load PHI detection patterns."""
        patterns = {
            PHICategory.SSN: [
                re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
                re.compile(r'\b\d{9}\b')
            ],
            PHICategory.EMAIL: [
                re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
            ],
            PHICategory.PHONE: [
                re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
                re.compile(r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b')
            ],
            PHICategory.MRN: [
                re.compile(r'\b(MRN|mrn)[:\s]*[A-Z0-9]{6,12}\b'),
                re.compile(r'\b[A-Z]{2}\d{6,10}\b')  # Common MRN format
            ],
            PHICategory.DOB: [
                re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
                re.compile(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b')
            ],
            PHICategory.GENOMIC: [
                re.compile(r'\b(rs|chr)\d+[:\s]*\d+\b'),  # SNP/position
                re.compile(r'\b[ACGT]{10,}\b')  # DNA sequence
            ],
            PHICategory.NAME: [
                # More complex - check against name databases
                re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b')  # Simple name pattern
            ]
        }
        
        # Adjust patterns based on sensitivity
        if self.sensitivity == "low":
            # Remove less certain patterns
            patterns[PHICategory.NAME] = []
        elif self.sensitivity == "high":
            # Add more aggressive patterns
            patterns[PHICategory.ADDRESS] = [
                re.compile(r'\b\d+\s+[A-Za-z\s]+\s+(Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr)\b', re.I)
            ]
        
        return patterns
    
    def detect(self, text: str) -> List[PHIDetection]:
        """
        Detect PHI in text.
        
        Args:
            text: Text to scan
            
        Returns:
            List of PHI detections
        """
        detections = []
        self.detection_stats['total_scanned'] += 1
        
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Check whitelist
                    if match.group() in self.whitelist:
                        continue
                    
                    # Calculate confidence based on pattern and context
                    confidence = self._calculate_confidence(
                        category, match.group(), text, match.span()
                    )
                    
                    if confidence > 0.5:  # Threshold for detection
                        detection = PHIDetection(
                            category=category,
                            pattern=pattern.pattern,
                            confidence=confidence,
                            position=match.span(),
                            redacted_value=self._redact_value(category, match.group())
                        )
                        detections.append(detection)
                        self.detection_stats['phi_detected'] += 1
        
        return detections
    
    def redact(self, text: str, detections: Optional[List[PHIDetection]] = None) -> str:
        """
        Redact PHI from text.
        
        Args:
            text: Text to redact
            detections: Pre-computed detections (optional)
            
        Returns:
            Redacted text
        """
        if detections is None:
            detections = self.detect(text)
        
        if not detections:
            return text
        
        # Sort by position (reverse to maintain indices)
        detections.sort(key=lambda d: d.position[0], reverse=True)
        
        redacted = text
        for detection in detections:
            start, end = detection.position
            redacted = redacted[:start] + detection.redacted_value + redacted[end:]
            self.detection_stats['redactions'] += 1
        
        return redacted
    
    def _calculate_confidence(self, 
                            category: PHICategory, 
                            value: str, 
                            context: str,
                            position: tuple) -> float:
        """
        Calculate confidence score for PHI detection.
        
        Args:
            category: PHI category
            value: Detected value
            context: Full text context
            position: Position in text
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.7  # Base confidence
        
        # Adjust based on category
        if category == PHICategory.SSN:
            # High confidence for SSN pattern
            confidence = 0.95
        elif category == PHICategory.EMAIL:
            # Check if it's a system email
            if any(domain in value for domain in ['localhost', 'example.com', 'test.com']):
                confidence = 0.1
            else:
                confidence = 0.9
        elif category == PHICategory.NAME:
            # Lower confidence for names without context
            if any(keyword in context.lower() for keyword in ['patient', 'user', 'person']):
                confidence = 0.8
            else:
                confidence = 0.4
        elif category == PHICategory.GENOMIC:
            # Check for genomic context
            if any(keyword in context.lower() for keyword in ['variant', 'snp', 'mutation', 'sequence']):
                confidence = 0.9
            else:
                confidence = 0.5
        
        return confidence
    
    def _redact_value(self, category: PHICategory, value: str) -> str:
        """
        Create redacted version of value.
        
        Args:
            category: PHI category
            value: Original value
            
        Returns:
            Redacted value
        """
        # Generate deterministic hash for consistency
        hash_suffix = hashlib.sha256(value.encode()).hexdigest()[:6]
        
        if category == PHICategory.SSN:
            return f"[SSN-{hash_suffix}]"
        elif category == PHICategory.EMAIL:
            parts = value.split('@')
            if len(parts) == 2:
                return f"[EMAIL-{parts[0][0]}***@{parts[1][0]}***]"
            return f"[EMAIL-{hash_suffix}]"
        elif category == PHICategory.PHONE:
            return f"[PHONE-***-***-{value[-4:]}]"
        elif category == PHICategory.MRN:
            return f"[MRN-{hash_suffix}]"
        elif category == PHICategory.NAME:
            return f"[NAME-{hash_suffix}]"
        elif category == PHICategory.GENOMIC:
            return f"[GENOMIC-{len(value)}bp-{hash_suffix}]"
        else:
            return f"[REDACTED-{category.value}-{hash_suffix}]"
    
    def add_to_whitelist(self, values: List[str]) -> None:
        """Add values to whitelist."""
        self.whitelist.update(values)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return self.detection_stats.copy()


class StructuredLogger:
    """
    Structured JSON logger with request correlation and PHI detection.
    """
    
    def __init__(self, 
                 name: str,
                 level: str = "INFO",
                 enable_phi_detection: bool = True,
                 output_file: Optional[str] = None):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            level: Log level
            enable_phi_detection: Enable PHI leak detection
            output_file: Optional log file path
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Configure JSON formatter
        formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            enable_phi_detection=enable_phi_detection
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if output_file:
            file_handler = logging.FileHandler(output_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # PHI detector
        self.phi_detector = PHILeakageDetector() if enable_phi_detection else None
        
        # Request context
        self.request_context = {}
        
    @contextmanager
    def request_context_manager(self, 
                               request_id: Optional[str] = None,
                               user_id: Optional[str] = None,
                               **kwargs):
        """
        Context manager for request correlation.
        
        Args:
            request_id: Request ID (generated if not provided)
            user_id: User ID
            **kwargs: Additional context fields
        """
        if request_id is None:
            request_id = str(uuid4())
        
        old_context = self.request_context.copy()
        
        self.request_context = {
            'request_id': request_id,
            'user_id': user_id,
            'start_time': time.time(),
            **kwargs
        }
        
        try:
            self.info(f"Request started", extra={'event': 'request_start'})
            yield request_id
        except Exception as e:
            self.error(f"Request failed: {str(e)}", 
                      extra={'event': 'request_error', 'error': str(e)})
            raise
        finally:
            duration = time.time() - self.request_context['start_time']
            self.info(f"Request completed", 
                     extra={'event': 'request_end', 'duration_ms': duration * 1000})
            self.request_context = old_context
    
    def _prepare_log_data(self, msg: str, extra: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Prepare log data with context and PHI detection.
        
        Args:
            msg: Log message
            extra: Extra fields
            
        Returns:
            Prepared log data
        """
        data = {
            'message': msg,
            'timestamp': datetime.utcnow().isoformat(),
            **self.request_context
        }
        
        if extra:
            data.update(extra)
        
        # PHI detection and redaction
        if self.phi_detector:
            # Check message
            detections = self.phi_detector.detect(msg)
            if detections:
                data['phi_detected'] = True
                data['phi_categories'] = list(set(d.category.value for d in detections))
                data['message'] = self.phi_detector.redact(msg, detections)
            
            # Check extra fields
            if extra:
                for key, value in extra.items():
                    if isinstance(value, str):
                        value_detections = self.phi_detector.detect(value)
                        if value_detections:
                            data[key] = self.phi_detector.redact(value, value_detections)
        
        return data
    
    def debug(self, msg: str, extra: Optional[Dict] = None) -> None:
        """Log debug message."""
        data = self._prepare_log_data(msg, extra)
        self.logger.debug(msg, extra=data)
    
    def info(self, msg: str, extra: Optional[Dict] = None) -> None:
        """Log info message."""
        data = self._prepare_log_data(msg, extra)
        self.logger.info(msg, extra=data)
    
    def warning(self, msg: str, extra: Optional[Dict] = None) -> None:
        """Log warning message."""
        data = self._prepare_log_data(msg, extra)
        self.logger.warning(msg, extra=data)
    
    def error(self, msg: str, extra: Optional[Dict] = None, exc_info: bool = False) -> None:
        """Log error message."""
        data = self._prepare_log_data(msg, extra)
        
        if exc_info:
            data['traceback'] = traceback.format_exc()
        
        self.logger.error(msg, extra=data, exc_info=exc_info)
    
    def critical(self, msg: str, extra: Optional[Dict] = None) -> None:
        """Log critical message."""
        data = self._prepare_log_data(msg, extra)
        self.logger.critical(msg, extra=data)
    
    def audit(self, 
             action: str,
             resource: str,
             outcome: str,
             details: Optional[Dict] = None) -> None:
        """
        Log audit event.
        
        Args:
            action: Action performed
            resource: Resource accessed
            outcome: Outcome (success/failure)
            details: Additional details
        """
        audit_data = {
            'audit': True,
            'action': action,
            'resource': resource,
            'outcome': outcome,
            'details': details or {}
        }
        
        self.info(f"Audit: {action} on {resource} - {outcome}", extra=audit_data)
    
    def metric(self,
              name: str,
              value: float,
              unit: str,
              tags: Optional[Dict] = None) -> None:
        """
        Log metric event.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            tags: Metric tags
        """
        metric_data = {
            'metric': True,
            'metric_name': name,
            'metric_value': value,
            'metric_unit': unit,
            'metric_tags': tags or {}
        }
        
        self.info(f"Metric: {name}={value}{unit}", extra=metric_data)


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter with PHI detection.
    """
    
    def __init__(self, *args, enable_phi_detection: bool = True, **kwargs):
        """Initialize formatter."""
        super().__init__(*args, **kwargs)
        self.enable_phi_detection = enable_phi_detection
        self.phi_detector = PHILeakageDetector() if enable_phi_detection else None
    
    def add_fields(self, log_record: Dict, record: logging.LogRecord, message_dict: Dict):
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add standard fields
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        
        # Add extra fields from record
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        
        # PHI detection on final output
        if self.phi_detector and self.enable_phi_detection:
            # Scan the entire log record
            log_str = json.dumps(log_record)
            detections = self.phi_detector.detect(log_str)
            
            if detections:
                log_record['phi_detection_warning'] = True
                log_record['phi_categories_detected'] = list(set(d.category.value for d in detections))
                
                # Redact sensitive fields
                for key, value in log_record.items():
                    if isinstance(value, str):
                        field_detections = self.phi_detector.detect(value)
                        if field_detections:
                            log_record[key] = self.phi_detector.redact(value, field_detections)


# Global logger factory
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str, 
              level: Optional[str] = None,
              enable_phi_detection: bool = True) -> StructuredLogger:
    """
    Get or create a structured logger.
    
    Args:
        name: Logger name
        level: Log level (uses env var if not specified)
        enable_phi_detection: Enable PHI detection
        
    Returns:
        Structured logger instance
    """
    if name not in _loggers:
        import os
        
        # Determine log level
        if level is None:
            level = os.getenv('GENOMEVAULT_LOG_LEVEL', 'INFO')
        
        # Check for component-specific level
        component = name.split('.')[0] if '.' in name else name
        env_var = f'GENOMEVAULT_{component.upper()}_LOG_LEVEL'
        component_level = os.getenv(env_var)
        if component_level:
            level = component_level
        
        # Create logger
        log_dir = os.getenv('GENOMEVAULT_LOG_DIR', 'logs')
        log_file = None
        
        if log_dir and os.getenv('GENOMEVAULT_LOG_TO_FILE', 'false').lower() == 'true':
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f'{name}.log')
        
        _loggers[name] = StructuredLogger(
            name=name,
            level=level,
            enable_phi_detection=enable_phi_detection,
            output_file=log_file
        )
    
    return _loggers[name]


# Convenience function for module-level logger
def get_module_logger(enable_phi_detection: bool = True) -> StructuredLogger:
    """
    Get logger for calling module.
    
    Args:
        enable_phi_detection: Enable PHI detection
        
    Returns:
        Module logger
    """
    import inspect
    
    frame = inspect.currentframe()
    if frame and frame.f_back:
        module = inspect.getmodule(frame.f_back)
        if module:
            return get_logger(module.__name__, enable_phi_detection=enable_phi_detection)
    
    return get_logger('unknown')


# Export convenience
__all__ = [
    'StructuredLogger',
    'PHILeakageDetector',
    'PHICategory',
    'PHIDetection',
    'get_logger',
    'get_module_logger',
    'LogLevel'
]