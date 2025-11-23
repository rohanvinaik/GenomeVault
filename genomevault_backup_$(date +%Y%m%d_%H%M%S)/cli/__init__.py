"""Module for cli functionality."""

# Import the main entry point from the cli module
try:
    from ..cli import main, app
except ImportError:
    # Fallback to importing from main
    from .main import main
    app = None

# Import legacy functions for backward compatibility
try:
    from .training_proof_cli import main as training_main
    from .main import encode, sim, index_build, search, prove, verify
except ImportError:
    training_main = None
    encode = sim = index_build = search = prove = verify = None

__all__ = [
    "app",
    "main",
    "encode",
    "index_build",
    "prove",
    "search",
    "sim",
    "verify",
    "training_main",
]
