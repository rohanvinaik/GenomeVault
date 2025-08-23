"""Module for cli functionality."""

from .training_proof_cli import main as training_main
from .main import encode, sim, index_build, search, prove, verify, main

__all__ = [
    "encode",
    "index_build",
    "main",
    "prove",
    "search",
    "sim",
    "verify",
    "training_main",
]
