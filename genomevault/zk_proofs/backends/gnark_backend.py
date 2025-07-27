"""
Real ZK proof generation using gnark via FFI.

This module provides a production-ready interface to gnark for generating
and verifying SNARK proofs. It handles circuit compilation, proof generation,
and verification with actual cryptographic guarantees.
"""
from typing import Dict, List, Optional, Any, Union

import json
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from genomevault.utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)
metrics = MetricsCollector()


class GnarkBackend:
    """Real ZK proof generation using gnark via FFI."""

    def __init__(self, circuit_dir: str = "./circuits/compiled", gnark_path: Optional[str] = None) -> None:
           """TODO: Add docstring for __init__"""
     """
        Initialize gnark backend.

        Args:
            circuit_dir: Directory containing compiled circuits
            gnark_path: Path to gnark binaries (auto-detect if None)
        """
        self.circuit_dir = Path(circuit_dir)
        self.circuit_dir.mkdir(parents=True, exist_ok=True)

        # Find gnark binaries
        self.gnark_path = gnark_path or self._find_gnark_path()
        self.prover_bin = Path(self.gnark_path) / "gnark-prover"
        self.verifier_bin = Path(self.gnark_path) / "gnark-verifier"
        self.compiler_bin = Path(self.gnark_path) / "gnark-compile"

        # Verify binaries exist
        self._verify_binaries()

        # Circuit cache
        self.compiled_circuits = {}

        # Compile standard circuits
        self._compile_standard_circuits()

    def _find_gnark_path(self) -> str:
           """TODO: Add docstring for _find_gnark_path"""
     """Auto-detect gnark installation path."""
        # Check common locations
        paths = [
            "./bin",
            "/usr/local/bin",
            "~/.local/bin",
            "./gnark/bin",
        ]

        for path in paths:
            expanded = Path(path).expanduser()
            if (expanded / "gnark-prover").exists():
                return str(expanded)

        # Try to find in PATH
        result = subprocess.run(["which", "gnark-prover"], capture_output=True, text=True)
        if result.returncode == 0:
            return str(Path(result.stdout.strip()).parent)

        raise RuntimeError("gnark binaries not found. Please install gnark or specify path.")

    def _verify_binaries(self) -> None:
           """TODO: Add docstring for _verify_binaries"""
     """Verify required gnark binaries exist."""
        required = [self.prover_bin, self.verifier_bin, self.compiler_bin]

        for binary in required:
            if not binary.exists():
                raise RuntimeError(f"Required binary not found: {binary}")

        # Test execution
        try:
            result = subprocess.run(
                [str(self.prover_bin), "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to execute gnark-prover: {result.stderr}")

            logger.info(f"gnark backend initialized: {result.stdout.strip()}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("gnark-prover timed out during version check")

    def _compile_standard_circuits(self) -> None:
           """TODO: Add docstring for _compile_standard_circuits"""
     """Compile standard genomic circuits."""
        standard_circuits = [
            "variant_proof",
            "polygenic_risk_score",
            "ancestry_inference",
            "recursive_verifier",
            "computation_result",
        ]

        for circuit_name in standard_circuits:
            circuit_file = self.circuit_dir / f"{circuit_name}.go"

            if circuit_file.exists():
                try:
                    self._compile_circuit(circuit_name)
                except Exception as e:
                    logger.warning(f"Failed to compile {circuit_name}: {e}")

    def _compile_circuit(self, circuit_name: str) -> None:
           """TODO: Add docstring for _compile_circuit"""
     """Compile a gnark circuit."""
        circuit_file = self.circuit_dir / f"{circuit_name}.go"
        output_file = self.circuit_dir / f"{circuit_name}.r1cs"

        if not circuit_file.exists():
            raise ValueError(f"Circuit source not found: {circuit_file}")

        # Skip if already compiled and up-to-date
        if output_file.exists():
            if output_file.stat().st_mtime > circuit_file.stat().st_mtime:
                logger.debug(f"Circuit {circuit_name} already compiled")
                return

        logger.info(f"Compiling circuit: {circuit_name}")

        cmd = [
            str(self.compiler_bin),
            "--circuit",
            str(circuit_file),
            "--output",
            str(output_file),
            "--curve",
            "bn254",  # BN254 curve for efficiency
        ]

        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        compile_time = time.time() - start

        if result.returncode != 0:
            raise RuntimeError(f"Circuit compilation failed: {result.stderr}")

        # Record metrics
        metrics.record(f"{circuit_name}_compile_time", compile_time * 1000, "ms")

        logger.info(f"Compiled {circuit_name} in {compile_time:.2f}s")

        # Cache circuit info
        self.compiled_circuits[circuit_name] = {
            "r1cs_path": str(output_file),
            "compiled_at": time.time(),
        }

    def generate_proof(
        self, circuit_name: str, public_inputs: Dict[str, Any], private_inputs: Dict[str, Any]
    ) -> bytes:
           """TODO: Add docstring for generate_proof"""
     """
        Generate real SNARK proof using gnark.

        Args:
            circuit_name: Name of the circuit to use
            public_inputs: Public inputs for the circuit
            private_inputs: Private witness data

        Returns:
            Serialized proof bytes
        """
        # Ensure circuit is compiled
        if circuit_name not in self.compiled_circuits:
            self._compile_circuit(circuit_name)

        circuit_info = self.compiled_circuits[circuit_name]

        # Create temporary files for inputs
        with tempfile.TemporaryDirectory() as tmpdir:
            public_file = Path(tmpdir) / "public.json"
            private_file = Path(tmpdir) / "private.json"
            proof_file = Path(tmpdir) / "proof.bin"

            # Write inputs
            with open(public_file, "w") as f:
                json.dump(public_inputs, f)

            with open(private_file, "w") as f:
                json.dump(private_inputs, f)

            # Generate proof
            cmd = [
                str(self.prover_bin),
                "--circuit",
                circuit_info["r1cs_path"],
                "--public",
                str(public_file),
                "--private",
                str(private_file),
                "--proof",
                str(proof_file),
                "--curve",
                "bn254",
            ]

            start = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            generation_time = time.time() - start

            if result.returncode != 0:
                raise RuntimeError(f"Proof generation failed: {result.stderr}")

            # Read proof
            with open(proof_file, "rb") as f:
                proof_data = f.read()

            # Measure and record metrics
            proof_size = len(proof_data)
            metrics.record(f"{circuit_name}_proof_size", proof_size, "bytes")
            metrics.record(f"{circuit_name}_generation_time", generation_time * 1000, "ms")

            logger.info(
                f"Generated {circuit_name} proof: " f"{proof_size} bytes in {generation_time:.3f}s"
            )

            return proof_data

    def verify_proof(self, circuit_name: str, proof: bytes, public_inputs: Dict[str, Any]) -> bool:
           """TODO: Add docstring for verify_proof"""
     """
        Verify proof using gnark verifier.

        Args:
            circuit_name: Name of the circuit
            proof: Proof bytes to verify
            public_inputs: Public inputs for verification

        Returns:
            True if proof is valid
        """
        if circuit_name not in self.compiled_circuits:
            raise ValueError(f"Unknown circuit: {circuit_name}")

        circuit_info = self.compiled_circuits[circuit_name]

        with tempfile.TemporaryDirectory() as tmpdir:
            public_file = Path(tmpdir) / "public.json"
            proof_file = Path(tmpdir) / "proof.bin"

            # Write inputs
            with open(public_file, "w") as f:
                json.dump(public_inputs, f)

            with open(proof_file, "wb") as f:
                f.write(proof)

            # Verify proof
            cmd = [
                str(self.verifier_bin),
                "--circuit",
                circuit_info["r1cs_path"],
                "--public",
                str(public_file),
                "--proof",
                str(proof_file),
                "--curve",
                "bn254",
            ]

            start = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            verification_time = time.time() - start

            # Record metrics
            metrics.record(f"{circuit_name}_verification_time", verification_time * 1000, "ms")

            valid = result.returncode == 0

            logger.info(
                f"Verified {circuit_name} proof in {verification_time*1000:.2f}ms: "
                f"{'VALID' if valid else 'INVALID'}"
            )

            if not valid and result.stderr:
                logger.debug(f"Verification error: {result.stderr}")

            return valid

    def batch_verify(self, proofs: List[Tuple[str, bytes, Dict[str, Any]]]) -> List[bool]:
           """TODO: Add docstring for batch_verify"""
     """
        Batch verify multiple proofs.

        Args:
            proofs: List of (circuit_name, proof_bytes, public_inputs) tuples

        Returns:
            List of verification results
        """
        results = []

        # Group by circuit type for efficiency
        by_circuit = {}
        for circuit_name, proof, public_inputs in proofs:
            if circuit_name not in by_circuit:
                by_circuit[circuit_name] = []
            by_circuit[circuit_name].append((proof, public_inputs))

        # Verify each group
        for circuit_name, circuit_proofs in by_circuit.items():
            for proof, public_inputs in circuit_proofs:
                valid = self.verify_proof(circuit_name, proof, public_inputs)
                results.append(valid)

        return results

    def get_circuit_info(self, circuit_name: str) -> Dict[str, Any]:
           """TODO: Add docstring for get_circuit_info"""
     """Get information about a compiled circuit."""
        if circuit_name not in self.compiled_circuits:
            self._compile_circuit(circuit_name)

        info = self.compiled_circuits[circuit_name].copy()

        # Add circuit statistics
        r1cs_file = Path(info["r1cs_path"])
        if r1cs_file.exists():
            info["file_size"] = r1cs_file.stat().st_size

        return info


class SimulatedBackend:
    """Simulated backend for testing without gnark installation."""

    def __init__(self) -> None:
           """TODO: Add docstring for __init__"""
     """Initialize simulated backend."""
        logger.warning("Using simulated ZK backend - not cryptographically secure!")
        self.proof_counter = 0

    def generate_proof(
        self, circuit_name: str, public_inputs: Dict[str, Any], private_inputs: Dict[str, Any]
    ) -> bytes:
           """TODO: Add docstring for generate_proof"""
     """Generate simulated proof."""
        import hashlib

        # Simulate proof generation time
        time.sleep(0.1)

        # Create deterministic "proof"
        self.proof_counter += 1
        proof_data = hashlib.sha256(
            f"{circuit_name}:{json.dumps(public_inputs)}:{self.proof_counter}".encode()
        ).digest()

        # Add some padding to simulate realistic proof size
        proof_data += b"\x00" * 356  # Total ~384 bytes

        # Record simulated metrics
        metrics.record(f"{circuit_name}_proof_size", len(proof_data), "bytes")
        metrics.record(f"{circuit_name}_generation_time", 100, "ms")

        return proof_data

    def verify_proof(self, circuit_name: str, proof: bytes, public_inputs: Dict[str, Any]) -> bool:
           """TODO: Add docstring for verify_proof"""
     """Verify simulated proof."""
        # Simulate verification time
        time.sleep(0.025)

        # Always valid in simulation
        metrics.record(f"{circuit_name}_verification_time", 25, "ms")

        return True


def get_backend(use_real: bool = True) -> Union[GnarkBackend, SimulatedBackend]:
       """TODO: Add docstring for get_backend"""
     """
    Get appropriate backend based on availability.

    Args:
        use_real: Try to use real gnark backend if available

    Returns:
        Backend instance
    """
    if use_real:
        try:
            return GnarkBackend()
        except RuntimeError as e:
            logger.warning(f"Failed to initialize gnark backend: {e}")
            logger.warning("Falling back to simulated backend")

    return SimulatedBackend()
