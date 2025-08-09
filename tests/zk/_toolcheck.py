import shutil


import pytest


def require_toolchain():
    if not (shutil.which("circom") and shutil.which("snarkjs") and shutil.which("node")):
        pytest.skip("ZK toolchain (circom/snarkjs/node) not available")
