import pytest
import shutil


def require_toolchain():
    """Require toolchain.
    Returns:
        Result of the operation."""
    if not (shutil.which("circom") and shutil.which("snarkjs") and shutil.which("node")):
        pytest.skip("ZK toolchain (circom/snarkjs/node) not available")
