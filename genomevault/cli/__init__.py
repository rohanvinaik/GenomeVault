"""Module for cli functionality."""

from .training_proof_cli import main
from .main import encode, sim, index_build, search, prove, verify, main

__all__ = [
    "encode",
    "index_build",
    "main",
    "main",
    "prove",
    "search",
    "sim",
    "verify",
]
