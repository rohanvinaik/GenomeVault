"""API endpoints and routers for routers."""

from .vectors import encode_vector, perform_operation, calculate_similarity
from .clinical import clinical_eval
from .pir import pir_query
from .config import (
    EncodingConfigBase,
    EncodingConfigCreate,
    EncodingConfigUpdate,
    EncodingConfigResponse,
    UserPreferencesBase,
    UserPreferencesPatch,
)
from .encode import EncodeIn, EncodeOut, do_encode
from .health import health
from .ledger import append_entry, verify_chain, list_entries
from .federated import aggregate
from .topology import NodeInfo, TopologyRequest, TopologyResponse, NETWORK_NODES
from .healthz import (
    HealthStatus,
    DetailedHealthStatus,
    check_database,
    check_cache,
    check_filesystem,
)
from .proofs import create_proof, verify_proof
from .governance import (
    consent_grant,
    consent_revoke,
    consent_check,
    dsar_export,
    dsar_erase,
    ropa,
)

__all__ = [
    "DetailedHealthStatus",
    "EncodeIn",
    "EncodeOut",
    "EncodingConfigBase",
    "EncodingConfigCreate",
    "EncodingConfigResponse",
    "EncodingConfigUpdate",
    "HealthStatus",
    "NETWORK_NODES",
    "NodeInfo",
    "TopologyRequest",
    "TopologyResponse",
    "UserPreferencesBase",
    "UserPreferencesPatch",
    "aggregate",
    "append_entry",
    "calculate_similarity",
    "check_cache",
    "check_database",
    "check_filesystem",
    "clinical_eval",
    "consent_check",
    "consent_grant",
    "consent_revoke",
    "create_proof",
    "do_encode",
    "dsar_erase",
    "dsar_export",
    "encode_vector",
    "health",
    "list_entries",
    "perform_operation",
    "pir_query",
    "ropa",
    "verify_chain",
    "verify_proof",
]
