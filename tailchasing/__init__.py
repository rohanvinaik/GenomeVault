"""
Tailchasing: Performance optimization analysis with chromatin-inspired visualizations.

This module provides tools for analyzing and visualizing performance bottlenecks
using concepts inspired by chromatin organization and Hi-C contact matrices.
"""

from .calibrate import CalibrationTool, ThrashEvent
from .cli import main as cli_main
from .config import DistanceWeights, PolymerConfig, get_config, save_config
from .core.reporting import HiCHeatmapGenerator, PolymerMetricsReport

__all__ = [
    "HiCHeatmapGenerator",
    "PolymerMetricsReport",
    "PolymerConfig",
    "DistanceWeights",
    "CalibrationTool",
    "ThrashEvent",
    "get_config",
    "save_config",
    "cli_main",
]
