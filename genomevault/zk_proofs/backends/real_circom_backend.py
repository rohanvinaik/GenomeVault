"""Real Circom backend integration for production ZK proofs."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple
import time

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class RealCircomBackend:
    """Production-ready Circom backend for cryptographically secure proofs."""

    def __init__(self, circuits_dir: str = "zk_circuits"):
        """
        Initialize Circom backend.

        Args:
            circuits_dir: Directory containing compiled circuits
        """
        self.circuits_dir = Path(circuits_dir)
        self.build_dir = self.circuits_dir / "build"
        self.available = self._check_availability()

        if self.available:
            logger.info("Real Circom backend initialized successfully")
        else:
            logger.warning("Circom backend not available - missing dependencies")

    def _check_availability(self) -> bool:
        """Check if Circom toolchain is available."""
        try:
            # Check snarkjs
            result = subprocess.run(
                ["snarkjs", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return False

            # Check for compiled circuits
            if not self.build_dir.exists():
                logger.warning(f"Circuits directory not found: {self.build_dir}")
                return False

            # Check for at least one compiled circuit
            r1cs_files = list(self.build_dir.glob("*.r1cs"))
            if not r1cs_files:
                logger.warning("No compiled circuits found")
                return False

            return True

        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def generate_proof(self, circuit_name: str, inputs: Dict[str, Any]) -> Tuple[Dict, list, float]:
        """
        Generate proof using native Circom.

        Args:
            circuit_name: Name of the circuit
            inputs: Circuit inputs

        Returns:
            Tuple of (proof, public_signals, generation_time)
        """
        if not self.available:
            raise RuntimeError("Circom backend not available")

        # Map circuit names
        circuit_map = {
            "variant_presence": "variant_presence",
            "polygenic_risk_score": "prs_calculation",
            "diabetes_risk_alert": "diabetes_risk",
            "pharmacogenomic": "prs_calculation",  # Reuse PRS for now
            "ancestry_composition": "prs_calculation",  # Reuse PRS for now
        }

        actual_circuit = circuit_map.get(circuit_name, circuit_name)

        # Prepare paths
        wasm_path = self.build_dir / f"{actual_circuit}_js" / f"{actual_circuit}.wasm"
        zkey_path = self.build_dir / f"{actual_circuit}_final.zkey"

        if not wasm_path.exists():
            raise FileNotFoundError(f"WASM file not found: {wasm_path}")
        if not zkey_path.exists():
            raise FileNotFoundError(f"zkey file not found: {zkey_path}")

        # Prepare inputs based on circuit
        prepared_inputs = self._prepare_inputs(actual_circuit, inputs)

        # Write input to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(prepared_inputs, f)
            input_file = f.name

        # Generate temporary output files
        proof_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
        public_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name

        try:
            # Generate proof
            start_time = time.time()

            result = subprocess.run(
                [
                    "snarkjs",
                    "groth16",
                    "fullprove",
                    input_file,
                    str(wasm_path),
                    str(zkey_path),
                    proof_file,
                    public_file,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            generation_time = time.time() - start_time

            if result.returncode != 0:
                raise RuntimeError(f"Proof generation failed: {result.stderr}")

            # Read proof and public signals
            with open(proof_file) as f:
                proof = json.load(f)
            with open(public_file) as f:
                public_signals = json.load(f)

            logger.info(f"Generated real proof for {circuit_name} in {generation_time:.2f}s")

            return proof, public_signals, generation_time

        finally:
            # Cleanup temp files
            for file in [input_file, proof_file, public_file]:
                Path(file).unlink(missing_ok=True)

    def verify_proof(self, circuit_name: str, proof: Dict, public_signals: list) -> bool:
        """
        Verify a proof.

        Args:
            circuit_name: Name of the circuit
            proof: Proof to verify
            public_signals: Public signals

        Returns:
            True if proof is valid
        """
        if not self.available:
            raise RuntimeError("Circom backend not available")

        # Map circuit names
        circuit_map = {
            "variant_presence": "variant_presence",
            "polygenic_risk_score": "prs_calculation",
            "diabetes_risk_alert": "diabetes_risk",
        }

        actual_circuit = circuit_map.get(circuit_name, circuit_name)

        # Get verification key
        vkey_path = self.build_dir / f"{actual_circuit}_verification_key.json"

        if not vkey_path.exists():
            raise FileNotFoundError(f"Verification key not found: {vkey_path}")

        # Write proof and signals to temp files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(proof, f)
            proof_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(public_signals, f)
            public_file = f.name

        try:
            # Verify proof
            result = subprocess.run(
                ["snarkjs", "groth16", "verify", str(vkey_path), public_file, proof_file],
                capture_output=True,
                text=True,
                timeout=10,
            )

            valid = "OK" in result.stdout or result.returncode == 0

            logger.info(f"Proof verification for {circuit_name}: {'valid' if valid else 'invalid'}")

            return valid

        finally:
            # Cleanup
            Path(proof_file).unlink(missing_ok=True)
            Path(public_file).unlink(missing_ok=True)

    def _prepare_inputs(self, circuit_name: str, raw_inputs: Dict) -> Dict:
        """
        Prepare inputs for specific circuit.

        Args:
            circuit_name: Name of the circuit
            raw_inputs: Raw input data

        Returns:
            Prepared inputs matching circuit requirements
        """
        if circuit_name == "variant_presence":
            # Prepare variant presence inputs
            variants = raw_inputs.get("variants", [])
            query = raw_inputs.get("query", [0, 0, 0])

            # Convert variant data to numeric format
            numeric_variants = []
            for v in variants:
                if isinstance(v, dict):
                    # Convert chr to number (1-22, X=23, Y=24)
                    chr_num = self._chr_to_num(v.get("chr", "1"))
                    pos = v.get("pos", 0)
                    alt = self._base_to_num(v.get("alt", "A"))
                    numeric_variants.append([chr_num, pos, alt])
                else:
                    numeric_variants.append(v)

            # Pad to circuit size (100)
            while len(numeric_variants) < 100:
                numeric_variants.append([0, 0, 0])

            # Convert query
            if isinstance(query, dict):
                query = [
                    self._chr_to_num(query.get("chr", "1")),
                    query.get("pos", 0),
                    self._base_to_num(query.get("alt", "A")),
                ]

            return {"variants": numeric_variants[:100], "query": query}

        elif circuit_name == "prs_calculation":
            # Prepare PRS inputs
            genotypes = raw_inputs.get("genotypes", [])
            weights = raw_inputs.get("weights", [])

            # Pad to circuit size (500)
            while len(genotypes) < 500:
                genotypes.append(0)
            while len(weights) < 500:
                weights.append(0)

            return {"genotypes": genotypes[:500], "weights": weights[:500]}

        elif circuit_name == "diabetes_risk":
            # Prepare diabetes risk inputs
            risk_factors = raw_inputs.get("risk_factors", [])
            thresholds = raw_inputs.get("thresholds", [30, 60, 80])

            # Ensure we have 10 risk factors
            while len(risk_factors) < 10:
                risk_factors.append(0)

            return {"risk_factors": risk_factors[:10], "thresholds": thresholds[:3]}

        else:
            # Return as-is for unknown circuits
            return raw_inputs

    def _chr_to_num(self, chr_str: str) -> int:
        """Convert chromosome string to number."""
        if isinstance(chr_str, int):
            return chr_str

        chr_str = str(chr_str).replace("chr", "").upper()
        if chr_str == "X":
            return 23
        elif chr_str == "Y":
            return 24
        elif chr_str == "M" or chr_str == "MT":
            return 25
        else:
            try:
                return int(chr_str)
            except ValueError:
                return 0

    def _base_to_num(self, base: str) -> int:
        """Convert nucleotide base to number."""
        if isinstance(base, int):
            return base

        base_map = {"A": 1, "T": 2, "G": 3, "C": 4}
        return base_map.get(str(base).upper(), 0)

    def get_circuit_info(self, circuit_name: str) -> Dict[str, Any]:
        """
        Get information about a circuit.

        Args:
            circuit_name: Name of the circuit

        Returns:
            Circuit information
        """
        circuit_map = {
            "variant_presence": "variant_presence",
            "polygenic_risk_score": "prs_calculation",
            "diabetes_risk_alert": "diabetes_risk",
        }

        actual_circuit = circuit_map.get(circuit_name, circuit_name)

        r1cs_path = self.build_dir / f"{actual_circuit}.r1cs"
        zkey_path = self.build_dir / f"{actual_circuit}_final.zkey"

        info = {
            "name": circuit_name,
            "actual_circuit": actual_circuit,
            "available": r1cs_path.exists() and zkey_path.exists(),
            "r1cs_size": r1cs_path.stat().st_size if r1cs_path.exists() else 0,
            "zkey_size": zkey_path.stat().st_size if zkey_path.exists() else 0,
        }

        # Get constraint count if possible
        if r1cs_path.exists():
            try:
                result = subprocess.run(
                    ["snarkjs", "r1cs", "info", str(r1cs_path)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                # Parse constraint count from output
                for line in result.stdout.split("\n"):
                    if "Constraints:" in line:
                        info["constraints"] = int(line.split(":")[-1].strip())
                        break
            except:
                pass

        return info
