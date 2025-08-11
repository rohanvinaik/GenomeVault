"""
Experimental advanced PIR (Private Information Retrieval) protocols.

EXPERIMENTAL: These PIR implementations are research prototypes.
They have not been audited for production use.
"""

import warnings

warnings.warn(
    "Advanced PIR protocols are experimental and not audited for production use. "
    "Security properties have not been formally verified.",
    FutureWarning,
    stacklevel=2,
)

# Original imports preserved for compatibility
try:
    from .graph_pir import GraphBasedPIR, PangenomeGraph
except ImportError:
    GraphBasedPIR = None
    PangenomeGraph = None

from .it_pir import InformationTheoreticPIR, PIRQuery, PIRResponse
from .robust_it_pir import RobustITPIR

__all__ = [
    "GraphBasedPIR",
    "InformationTheoreticPIR",
    "PIRQuery",
    "PIRResponse",
    "PangenomeGraph",
    "RobustITPIR",
]
