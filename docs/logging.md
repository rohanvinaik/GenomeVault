# GenomeVault Logging System

This document describes the comprehensive logging system implemented for GenomeVault, including configuration, usage, and best practices.

## Features

- **Centralized Configuration**: Single point of control for all logging settings
- **Environment Variables**: Configure logging through environment variables
- **Log Rotation**: Automatic log file rotation to prevent disk space issues
- **Multiple Handlers**: Console, file, and specialized handlers
- **Component-Specific Levels**: Different log levels for different components
- **Performance Logging**: Dedicated performance metrics logging
- **JSON Support**: Structured logging for production environments
- **Context Managers**: Convenient logging with automatic timing
- **Decorators**: Automatic function execution logging

## Quick Start

### Basic Usage

```python
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("Application started")
logger.error("Something went wrong", exc_info=True)
```

### Environment Configuration

Set log level via environment variable:
```bash
export GENOMEVAULT_LOG_LEVEL=DEBUG
export GENOMEVAULT_LOG_DIR=/var/log/genomevault
```

### Component-Specific Logging

```bash
export GENOMEVAULT_API_LOG_LEVEL=INFO
export GENOMEVAULT_ZK_PROOFS_LOG_LEVEL=WARNING
export GENOMEVAULT_HYPERVECTOR_LOG_LEVEL=DEBUG
```

## Configuration

### Environment Variables

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `GENOMEVAULT_LOG_LEVEL` | Global log level | INFO | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `GENOMEVAULT_LOG_DIR` | Log directory | logs | Any valid directory path |
| `GENOMEVAULT_ENV` | Environment type | development | development, production, testing, staging |

#### Component-Specific Variables

- `GENOMEVAULT_HYPERVECTOR_LOG_LEVEL`
- `GENOMEVAULT_ZK_PROOFS_LOG_LEVEL`
- `GENOMEVAULT_PIR_LOG_LEVEL`
- `GENOMEVAULT_FEDERATED_LOG_LEVEL`
- `GENOMEVAULT_BLOCKCHAIN_LOG_LEVEL`
- `GENOMEVAULT_API_LOG_LEVEL`

### Programmatic Configuration

```python
from genomevault.utils.logging import configure_logging

configure_logging(
    level=logging.INFO,
    enable_file_logging=True,
    enable_json_logging=True,
    max_bytes=100 * 1024 * 1024,  # 100MB
    backup_count=5
)
```

### Environment-Specific Configuration

```python
from genomevault.config.logging_config import configure_for_environment

# Automatically configure based on GENOMEVAULT_ENV
configure_for_environment()

# Or explicitly set environment
configure_for_environment("production")
```

## Log Files

The logging system creates several log files:

### Main Application Log (`genomevault.log`)
- Contains all application logs at configured level
- Rotates when reaching max size (default 10MB)
- Keeps backup files (default 5)

### Error Log (`genomevault_errors.log`)
- Contains only ERROR and CRITICAL messages
- Uses detailed format with function names and line numbers
- Useful for monitoring and alerting

### Performance Log (`genomevault_performance.log`)
- Contains performance metrics in JSON format
- Separate from main application log
- Used for monitoring and optimization

## Usage Patterns

### Basic Logging

```python
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)

logger.debug("Detailed debugging information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred", exc_info=True)
logger.critical("Critical system error")
```

### Performance Logging

```python
from genomevault.utils.logging import log_performance

# Manual performance logging
start_time = time.perf_counter()
# ... do work ...
duration = time.perf_counter() - start_time
log_performance("operation_name", duration, user_id=123, size="large")
```

### Decorator-Based Logging

```python
from genomevault.utils.logging import log_operation

@log_operation
def encode_hypervector(data):
    """This function will be automatically logged."""
    # ... implementation ...
    return encoded_data
```

### Context Manager Logging

```python
from genomevault.utils.logging import ContextLogger, get_logger

logger = get_logger(__name__)

with ContextLogger(logger, "processing_batch", batch_size=100, user_id=123):
    # ... do work ...
    # Automatically logs start time, completion time, and duration
    pass
```

### Convenience Functions

```python
from genomevault.utils.logging import info, error, debug

debug("Debug message", "genomevault.component")
info("Info message")
error("Error occurred")
```

## Environment Configurations

### Development Environment
- **Log Level**: DEBUG
- **File Logging**: Enabled
- **Console Logging**: Enabled
- **JSON Format**: Disabled
- **Max File Size**: 50MB
- **Backup Count**: 3

### Production Environment
- **Log Level**: INFO
- **File Logging**: Enabled
- **Console Logging**: Enabled
- **JSON Format**: Enabled
- **Max File Size**: 100MB
- **Backup Count**: 10

### Testing Environment
- **Log Level**: WARNING
- **File Logging**: Disabled
- **Console Logging**: Enabled
- **JSON Format**: Disabled

### Container Environments

For Docker/Kubernetes deployments:

```python
from genomevault.config.logging_config import configure_for_docker

# Optimized for containerized environments
configure_for_docker()
```

Features:
- No file logging (logs to stdout/stderr)
- JSON format for log aggregation
- Minimal overhead

## Best Practices

### 1. Use Appropriate Log Levels

```python
logger.debug("Variable x = %s", x)           # Development debugging
logger.info("User %s logged in", user_id)   # Important events
logger.warning("Rate limit exceeded")        # Potential issues
logger.error("Database connection failed")  # Errors that affect functionality
logger.critical("System out of memory")     # System-threatening issues
```

### 2. Include Context Information

```python
logger.info("Processing started", extra={
    "user_id": user.id,
    "batch_size": len(items),
    "component": "hypervector_encoder"
})
```

### 3. Use Structured Logging for Production

```python
import json

logger.info(json.dumps({
    "event": "user_action",
    "user_id": user_id,
    "action": "encode_genome",
    "timestamp": datetime.utcnow().isoformat(),
    "metadata": {"genome_size": len(data)}
}))
```

### 4. Don't Log Sensitive Information

```python
# DON'T do this
logger.info("User credentials: %s", password)

# DO this instead
logger.info("Authentication attempt for user: %s", username)
```

### 5. Use Performance Logging for Optimization

```python
@log_operation
def expensive_computation(data):
    """Automatically logs execution time for performance monitoring."""
    return process_data(data)
```

## Monitoring and Alerting

### Log File Locations

- Development: `logs/` directory
- Production: Configured via `GENOMEVAULT_LOG_DIR`
- Docker: stdout/stderr (no files)

### Error Monitoring

Monitor `genomevault_errors.log` for:
- ERROR and CRITICAL messages
- Stack traces and exception information
- System failures

### Performance Monitoring

Parse `genomevault_performance.log` for:
- Operation execution times
- Performance trends
- Resource usage patterns

### Example Monitoring Script

```python
import json
from pathlib import Path

def monitor_performance():
    perf_log = Path("logs/genomevault_performance.log")
    
    with open(perf_log) as f:
        for line in f:
            if "PERF" in line:
                # Extract JSON data
                json_part = line.split("PERF | ")[1]
                metrics = json.loads(json_part)
                
                # Check for slow operations
                if metrics["duration_ms"] > 1000:
                    print(f"Slow operation: {metrics}")
```

## Integration with External Systems

### ELK Stack (Elasticsearch, Logstash, Kibana)

Configure JSON logging for easier parsing:

```python
configure_logging(enable_json_logging=True)
```

### Prometheus/Grafana

Use performance logs for metrics:

```python
# Custom metrics collection
from prometheus_client import Histogram

operation_duration = Histogram('genomevault_operation_duration_seconds',
                              'Time spent on operations',
                              ['operation_name'])

@log_operation
def monitored_function():
    with operation_duration.labels('encode_hypervector').time():
        # ... function implementation ...
        pass
```

### Sentry Error Tracking

```python
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

sentry_logging = LoggingIntegration(
    level=logging.INFO,
    event_level=logging.ERROR
)

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[sentry_logging]
)
```

## Troubleshooting

### Common Issues

1. **Log files not created**
   - Check directory permissions
   - Verify `GENOMEVAULT_LOG_DIR` exists
   - Ensure sufficient disk space

2. **Missing log messages**
   - Check log level configuration
   - Verify logger name hierarchy
   - Check handler configuration

3. **Performance impact**
   - Use appropriate log levels in production
   - Consider async logging for high-throughput systems
   - Monitor disk space usage

### Debug Configuration

```python
from genomevault.utils.logging import configure_logging
import logging

# Enable debug logging temporarily
configure_logging(
    level=logging.DEBUG,
    force_reconfigure=True
)
```

## Migration from Old Logging

If migrating from simple print statements or basic logging:

1. Replace `print()` with appropriate log calls
2. Add logger initialization
3. Configure log levels appropriately
4. Add performance logging for critical operations

### Automatic Migration Script

The project includes an automatic migration script that converts print statements to logging calls. See `scripts/remove_debug_prints.py` for details.

## Testing Logging Configuration

Run the comprehensive test suite:

```bash
python scripts/test_logging_config.py
```

This validates:
- Basic logging functionality
- Environment configurations
- File creation and rotation
- Performance logging
- Context managers and decorators