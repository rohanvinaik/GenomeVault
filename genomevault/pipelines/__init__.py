"""Module for pipelines functionality."""

from .profile import profile_dataframe
from .e2e_pipeline import run_e2e

__all__ = [
    "profile_dataframe",
    "run_e2e",
]
