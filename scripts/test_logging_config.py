#!/usr/bin/env python3

"""Test script to validate the new logging configuration."""

import os
import sys
import time
from pathlib import Path

# Add genomevault to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomevault.utils.logging import (
    configure_logging,
    get_logger,
    log_performance,
    log_operation,
    ContextLogger,
)
from genomevault.config.logging_config import (
    configure_for_environment,
    configure_for_docker,
    auto_configure,
)


def test_basic_logging():
    """Test basic logging functionality."""
    print("=" * 60)
    print("=" * 60)

    # Test basic logger creation
    logger = get_logger("test.basic")

    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")

    pass  # Debug print removed


def test_component_loggers():
    """Test component-specific loggers."""
    print("=" * 60)

    components = [
        "genomevault.hypervector",
        "genomevault.zk_proofs",
        "genomevault.pir",
        "genomevault.federated",
        "genomevault.blockchain",
        "genomevault.api",
    ]

    for component in components:
        logger = get_logger(component)
        logger.info(f"Testing {component} logger")

    pass  # Debug print removed


def test_performance_logging():
    """Test performance logging functionality."""
    print("=" * 60)

    # Test direct performance logging
    log_performance("test_operation", 0.123, param1="value1", param2=42)

    # Test performance decorator
    @log_operation
    def sample_operation():
        """Sample operation for testing."""
        time.sleep(0.1)
        return "completed"

    result = sample_operation()
    print(f"Operation result: {result}")

    pass  # Debug print removed


def test_context_logging():
    """Test context manager logging."""
    print("=" * 60)

    logger = get_logger("test.context")

    # Test successful operation
    with ContextLogger(logger, "successful_operation", user_id=123, action="test"):
        time.sleep(0.05)
        logger.info("Doing work inside context")

    # Test failed operation
    try:
        with ContextLogger(logger, "failed_operation", user_id=456):
            time.sleep(0.02)
            raise ValueError("Simulated error")
    except ValueError:
        pass  # Expected

    pass  # Debug print removed


def test_environment_configurations():
    """Test different environment configurations."""
    print("=" * 60)

    environments = ["development", "production", "testing", "staging"]

    for env in environments:
        pass  # Debug print removed

        # Set environment variable
        old_env = os.environ.get("GENOMEVAULT_ENV")
        os.environ["GENOMEVAULT_ENV"] = env

        try:
            configure_for_environment(env)
            logger = get_logger(f"test.{env}")
            logger.info(f"Testing {env} environment configuration")
            print(f"✓ {env} environment configured successfully")
        except Exception as e:
            print(f"✗ {env} environment failed: {e}")
        finally:
            # Restore original environment
            if old_env:
                os.environ["GENOMEVAULT_ENV"] = old_env
            elif "GENOMEVAULT_ENV" in os.environ:
                del os.environ["GENOMEVAULT_ENV"]


def test_log_files():
    """Test that log files are created and rotated properly."""
    print("=" * 60)

    # Configure logging with file output
    log_dir = Path("test_logs")
    configure_logging(
        log_dir=log_dir,
        enable_file_logging=True,
        max_bytes=1024,  # Small size for testing rotation
        backup_count=2,
    )

    logger = get_logger("test.files")

    # Generate some log messages
    for i in range(100):
        logger.info(f"Test message {i} - " + "x" * 50)  # Make messages larger

    # Check if log files were created
    expected_files = [
        "genomevault.log",
        "genomevault_errors.log",
        "genomevault_performance.log",
    ]

    for filename in expected_files:
        file_path = log_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✓ {filename} created ({size} bytes)")
        else:
            print(f"✗ {filename} not found")

    print(f"✓ Log files created in {log_dir}")


def test_environment_variables():
    """Test environment variable configuration."""
    print("=" * 60)

    # Test log level from environment
    original_level = os.environ.get("GENOMEVAULT_LOG_LEVEL")

    try:
        os.environ["GENOMEVAULT_LOG_LEVEL"] = "DEBUG"
        configure_logging(force_reconfigure=True)
        logger = get_logger("test.env_vars")
        logger.debug("This DEBUG message should be visible")
        print("✓ Environment variable log level working")

        # Test component-specific levels
        os.environ["GENOMEVAULT_API_LOG_LEVEL"] = "ERROR"
        configure_logging(force_reconfigure=True)
        api_logger = get_logger("genomevault.api")
        api_logger.warning("This warning should NOT be visible for API logger")
        api_logger.error("This error SHOULD be visible for API logger")
        print("✓ Component-specific log levels working")

    finally:
        # Restore original environment
        if original_level:
            os.environ["GENOMEVAULT_LOG_LEVEL"] = original_level
        else:
            os.environ.pop("GENOMEVAULT_LOG_LEVEL", None)
        os.environ.pop("GENOMEVAULT_API_LOG_LEVEL", None)


def test_json_logging():
    """Test JSON logging format."""
    print("=" * 60)

    # Configure with JSON logging
    configure_logging(
        enable_json_logging=True,
        enable_file_logging=True,
        log_dir=Path("test_logs_json"),
        force_reconfigure=True,
    )

    logger = get_logger("test.json")
    logger.info("This message should be in JSON format in the log file")

    print("✓ JSON logging configuration applied")
    pass  # Debug print removed


def cleanup_test_logs():
    """Clean up test log directories."""
    import shutil

    test_dirs = ["test_logs", "test_logs_json"]
    for dir_name in test_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"✓ Cleaned up {dir_name}")


def main():
    """Run all logging tests."""
    print("=" * 60)
    print(f"Python version: {sys.version}")
    pass  # Debug print removed

    try:
        test_basic_logging()
        test_component_loggers()
        test_performance_logging()
        test_context_logging()
        test_environment_configurations()
        test_log_files()
        test_environment_variables()
        test_json_logging()

        print("=" * 60)
        print("2. Set appropriate environment variables in your .env file")
        print("3. Use the new logging system throughout your application")

    except Exception as e:
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        cleanup_test_logs()


if __name__ == "__main__":
    main()
