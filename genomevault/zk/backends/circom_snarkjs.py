from __future__ import annotations

"""Circom Snarkjs module."""
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CircuitPaths:
    """Data container for circuitpaths information."""

    root: Path  # e.g., genomevault/zk/circuits/sum64
    circom: Path  # sum64.circom
    build: Path  # build directory inside root
    r1cs: Path  # build/sum64.r1cs
    wasm: Path  # build/sum64_js/sum64.wasm
    zkey: Path  # build/sum64_final.zkey
    vkey: Path  # build/verification_key.json

    @staticmethod
    def for_sum64(repo_root: Path) -> CircuitPaths:
        """For sum64.

        Args:
            repo_root: Repo root.

        Returns:
            CircuitPaths instance.
        """
        root = repo_root / "genomevault" / "zk" / "circuits" / "sum64"
        return CircuitPaths(
            root=root,
            circom=root / "sum64.circom",
            build=root / "build",
            r1cs=root / "build" / "sum64.r1cs",
            wasm=root / "build" / "sum64_js" / "sum64.wasm",
            zkey=root / "build" / "sum64_final.zkey",
            vkey=root / "build" / "verification_key.json",
        )


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def toolchain_available() -> bool:
    """Toolchain available.

    Returns:
        Boolean result.
    """
    return bool(_which("circom") and _which("snarkjs") and _which("node"))


def run(cmd: list[str], cwd: Path) -> None:
    """Run.

    Args:
        cmd: Cmd.
        cwd: Cwd.
    """
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_built(paths: CircuitPaths) -> None:
    """Builds artifacts if missing. Requires circom + snarkjs toolchain."""
    if not toolchain_available():
        raise RuntimeError("ZK toolchain not available: require circom, snarkjs, node")

    paths.build.mkdir(parents=True, exist_ok=True)

    # Compile circom → r1cs, wasm
    if not paths.r1cs.exists() or not paths.wasm.exists():
        run(
            [
                "circom",
                str(paths.circom),
                "--r1cs",
                "--wasm",
                "--output",
                str(paths.build),
            ],
            cwd=paths.root,
        )

    # Powers of tau & zkey (small 12 power for tests)
    pot0 = paths.build / "pot12_0000.ptau"
    potF = paths.build / "pot12_final.ptau"
    if not potF.exists():
        run(
            ["snarkjs", "powersoftau", "new", "bn128", "12", str(pot0), "-v"],
            cwd=paths.root,
        )
        run(
            [
                "snarkjs",
                "powersoftau",
                "contribute",
                str(pot0),
                str(potF),
                "--name",
                "genesis",
                "-v",
            ],
            cwd=paths.root,
        )

    if not paths.zkey.exists():
        zkey0 = paths.build / "sum64_0000.zkey"
        run(
            ["snarkjs", "groth16", "setup", str(paths.r1cs), str(potF), str(zkey0)],
            cwd=paths.root,
        )
        run(
            [
                "snarkjs",
                "zkey",
                "export",
                "verificationkey",
                str(zkey0),
                str(paths.vkey),
            ],
            cwd=paths.root,
        )
        # Optionally contribute and mark final
        paths.zkey.write_bytes(zkey0.read_bytes())

    if not paths.vkey.exists():
        run(
            [
                "snarkjs",
                "zkey",
                "export",
                "verificationkey",
                str(paths.zkey),
                str(paths.vkey),
            ],
            cwd=paths.root,
        )


def prove(paths: CircuitPaths, a: int, b: int, c_public: int) -> dict:
    """Generate a Groth16 proof for private a,b such that a + b == c_public."""
    ensure_built(paths)

    input_json = {"a": int(a), "b": int(b), "c": int(c_public)}
    inp = paths.build / "input.json"
    wtns = paths.build / "witness.wtns"
    proof_json = paths.build / "proof.json"
    public_json = paths.build / "public.json"

    inp.write_text(json.dumps(input_json), encoding="utf-8")

    # Witness generation
    wasm_dir = paths.wasm.parent  # build/sum64_js
    gen_witness = wasm_dir / "generate_witness.js"
    run(["node", str(gen_witness), str(paths.wasm), str(inp), str(wtns)], cwd=paths.root)

    # Proof
    run(
        [
            "snarkjs",
            "groth16",
            "prove",
            str(paths.zkey),
            str(wtns),
            str(proof_json),
            str(public_json),
        ],
        cwd=paths.root,
    )

    return {
        "proof": json.loads(proof_json.read_text(encoding="utf-8")),
        "public": json.loads(public_json.read_text(encoding="utf-8")),
    }


def verify(paths: CircuitPaths, proof: dict, public: dict) -> bool:
    """Verify.

    Args:
        paths: File or directory paths.
        proof: Zero-knowledge proof.
        public: Public.

    Returns:
        Boolean result.

    Raises:
        RuntimeError: When operation fails.
    """
    ensure_built(paths)
    tmp_proof = paths.build / "tmp_proof.json"
    tmp_pub = paths.build / "tmp_public.json"
    tmp_proof.write_text(json.dumps(proof), encoding="utf-8")
    tmp_pub.write_text(json.dumps(public), encoding="utf-8")
    try:
        run(
            [
                "snarkjs",
                "groth16",
                "verify",
                str(paths.vkey),
                str(tmp_pub),
                str(tmp_proof),
            ],
            cwd=paths.root,
        )
        return True
    except subprocess.CalledProcessError:
        logger.exception("Unhandled exception")
        return False
        raise RuntimeError("Unspecified error")
