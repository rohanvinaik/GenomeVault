"""
Utilities for checking optional dependencies and providing helpful error messages.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class OptionalDependencyError(ImportError):
    """Error raised when an optional dependency is missing."""

    def __init__(self, package: str, feature: str, install_extra: str | None = None):
        if install_extra:
            message = (
                f"The '{package}' package is required for {feature}. "
                f"Install with: pip install genomevault[{install_extra}]"
            )
        else:
            message = (
                f"The '{package}' package is required for {feature}. "
                f"Install with: pip install {package}"
            )
        super().__init__(message)
        self.package = package
        self.feature = feature
        self.install_extra = install_extra


def require_package(package: str, feature: str, install_extra: str | None = None) -> Callable:
    """
    Decorator to require an optional package for a function or method.

    Args:
        package: Name of required package
        feature: Description of what the package enables
        install_extra: Optional extra name for pip install

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_package_available(package):
                raise OptionalDependencyError(package, feature, install_extra)
            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


def is_package_available(package: str) -> bool:
    """
    Check if a package is available for import.

    Args:
        package: Package name to check

    Returns:
        True if package is available, False otherwise
    """
    return importlib.util.find_spec(package) is not None


def try_import(package: str, feature: str, install_extra: str | None = None) -> Any:
    """
    Try to import a package, raising a helpful error if it's not available.

    Args:
        package: Package name to import
        feature: Description of what the package enables
        install_extra: Optional extra name for pip install

    Returns:
        Imported package module

    Raises:
        OptionalDependencyError: If package is not available
    """
    try:
        return importlib.import_module(package)
    except ImportError:
        raise OptionalDependencyError(package, feature, install_extra)


# Pre-check common packages
TORCH_AVAILABLE = is_package_available("torch")
SKLEARN_AVAILABLE = is_package_available("sklearn")
CUPY_AVAILABLE = is_package_available("cupy")
PANDAS_AVAILABLE = is_package_available("pandas")
NUMPY_AVAILABLE = is_package_available("numpy")


def require_ml_features(func: Callable) -> Callable:
    """Decorator to require ML packages (torch, sklearn, numpy)."""
    return require_package("torch", "ML features", "ml")(
        require_package("sklearn", "ML features", "ml")(
            require_package("numpy", "ML features", "ml")(func)
        )
    )


def require_gpu_features(func: Callable) -> Callable:
    """Decorator to require GPU packages (cupy)."""
    return require_package("cupy", "GPU acceleration", "gpu")(func)


def require_zk_features(func: Callable) -> Callable:
    """Decorator to require ZK packages (pysnark)."""
    return require_package("pysnark", "zero-knowledge proofs", "zk")(func)


def get_available_features() -> dict[str, bool]:
    """
    Get a dictionary of available optional features.

    Returns:
        Dictionary mapping feature names to availability
    """
    return {
        "ml": TORCH_AVAILABLE and SKLEARN_AVAILABLE and NUMPY_AVAILABLE,
        "gpu": CUPY_AVAILABLE,
        "zk": is_package_available("pysnark"),
        "nanopore": is_package_available("ont_fast5_api"),
        "pandas": PANDAS_AVAILABLE,
    }


def print_feature_status():
    """Print the status of all optional features."""
    features = get_available_features()

    logger.info("GenomeVault Optional Features:")
    logger.info("=" * 35)

    for feature, available in features.items():
        status = "✓ Available" if available else "✗ Not installed"
        logger.info(f"{feature:15} {status}")

    unavailable = [f for f, a in features.items() if not a]
    if unavailable:
        logger.info("\nTo install missing features:")
        for feature in unavailable:
            logger.info(f"  pip install genomevault[{feature}]")
        logger.info("  pip install genomevault[full]  # Install all features")
