from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


#!/usr/bin/env python3
"""
Example file to demonstrate autofix changes
"""

# Example 1: Print statements that should be converted to logging
logger.debug("Starting the application")
filename = "test.txt"  # Define variable before use
logger.debug(f"Processing file: {filename}")


# Example 2: Bare except clause
def risky_operation():
    """Example risky operation."""
    raise ValueError("Example error")


try:
    risky_operation()
except Exception:
    logger.exception("Unhandled exception")
    logger.debug("Something went wrong")
    raise


# Example 3: Broad exception handling
def another_operation():
    """Example operation."""
    pass


try:
    another_operation()
except Exception:
    pass


# Example 4: Unused parameters
def process_data(self, data, unused_param, another_unused):
    """Process some data"""
    return data * 2


# Example 5: Star import
from os import *


# Example 6: Top-level function call
def initialize_something():
    """Initialize example."""
    pass


initialize_something()


class MyClass:
    def method_with_issues(self, param1, param2, unused_param):
        """Method with various issues"""
        try:
            result = param1 + param2
            logger.debug(f"Result is: {result}")
            return result
        except Exception:
            logger.exception("Unhandled exception")
            logger.debug("Failed to calculate")
            raise


# Example usage at module level
if __name__ == "__main__":
    obj = MyClass()
    obj.method_with_issues(1, 2, "not used")
