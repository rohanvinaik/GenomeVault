"""
Version tracking for GenomeVault components.
Centralized version constants used throughout the system.
"""

# Protocol versions
PIR_PROTOCOL_VERSION = "PIR-IT-1.0"
PIR_PROTOCOL_REVISION = "2025.01.24"

# Circuit versions
ZK_CIRCUIT_VERSION = "v2.1.0"
HDC_ENCODER_VERSION = "v1.3.0"
VERIFIER_CONTRACT_VERSION = "v1.2.0"

# Component versions
PIR_CLIENT_VERSION = "1.0.0"
PIR_SERVER_VERSION = "1.0.0"
QUERY_BUILDER_VERSION = "1.0.0"
SHARD_MANAGER_VERSION = "1.0.0"

# Seeds (truncated for security in code)
MASTER_SEED_PREFIX = "0x7a3b9c5d"
HDC_SEED = "HDC-2025-01"
ZK_SEED = "ZK-SNARK-2025-01"

# Version compatibility
MIN_COMPATIBLE_PIR_VERSION = "PIR-IT-0.9"
MIN_COMPATIBLE_CLIENT_VERSION = "0.9.0"


def get_version_info():
    """Get complete version information."""
    return {
        "pir_protocol": {
            "version": PIR_PROTOCOL_VERSION,
            "revision": PIR_PROTOCOL_REVISION,
            "min_compatible": MIN_COMPATIBLE_PIR_VERSION,
        },
        "circuits": {
            "zk": ZK_CIRCUIT_VERSION,
            "hdc": HDC_ENCODER_VERSION,
            "verifier": VERIFIER_CONTRACT_VERSION,
        },
        "components": {
            "pir_client": PIR_CLIENT_VERSION,
            "pir_server": PIR_SERVER_VERSION,
            "query_builder": QUERY_BUILDER_VERSION,
            "shard_manager": SHARD_MANAGER_VERSION,
        },
    }


def check_compatibility(client_version: str, server_version: str) -> bool:
    """
    Check if client and server versions are compatible.

    Args:
        client_version: Client version string
        server_version: Server version string

    Returns:
        True if compatible, False otherwise
    """
    # Simple version check - in production would be more sophisticated
    client_major = int(client_version.split(".")[0])
    server_major = int(server_version.split(".")[0])

    return client_major == server_major
