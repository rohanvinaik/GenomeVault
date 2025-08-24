"""
Circom/SnarkJS backend for actual zero-knowledge proof generation.
"""

from __future__ import annotations

import json
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
import hashlib
import tempfile

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CircomCircuit:
    """Circom circuit configuration."""
    
    name: str
    circuit_path: Path
    build_dir: Path
    public_signals: list[str]
    private_signals: list[str]
    
    @property
    def r1cs_path(self) -> Path:
        return self.build_dir / f"{self.name}.r1cs"
    
    @property
    def wasm_path(self) -> Path:
        return self.build_dir / f"{self.name}_js" / f"{self.name}.wasm"
    
    @property
    def zkey_path(self) -> Path:
        return self.build_dir / f"{self.name}_final.zkey"
    
    @property
    def vkey_path(self) -> Path:
        return self.build_dir / "verification_key.json"


class CircomBackend:
    """Backend for generating real ZK proofs using Circom and SnarkJS."""
    
    def __init__(self):
        """Initialize the Circom backend."""
        self.repo_root = Path(__file__).parent.parent.parent
        self.circuits_dir = self.repo_root / "zk" / "circuits"
        self.circuits: Dict[str, CircomCircuit] = {}
        self._initialize_circuits()
        
    def _initialize_circuits(self):
        """Initialize available circuits."""
        # Variant presence circuit
        self.circuits["variant_presence"] = CircomCircuit(
            name="variant_presence",
            circuit_path=self.circuits_dir / "variant_presence" / "variant_presence.circom",
            build_dir=self.circuits_dir / "variant_presence" / "build",
            public_signals=["variant_hash", "reference_hash", "commitment_root"],
            private_signals=["chr", "position", "ref_allele", "alt_allele", 
                           "merkle_proof", "merkle_indices", "witness_randomness"]
        )
        
        # Diabetes risk circuit
        self.circuits["diabetes_risk"] = CircomCircuit(
            name="diabetes_risk",
            circuit_path=self.circuits_dir / "diabetes_risk" / "diabetes_risk.circom",
            build_dir=self.circuits_dir / "diabetes_risk" / "build",
            public_signals=["glucose_threshold", "risk_threshold", "result_commitment"],
            private_signals=["glucose_reading", "risk_score", "witness_randomness"]
        )
        
    def check_dependencies(self) -> bool:
        """Check if required tools are available."""
        circom = shutil.which("circom")
        snarkjs = shutil.which("snarkjs")
        node = shutil.which("node")
        
        if not all([circom, snarkjs, node]):
            missing = []
            if not circom:
                missing.append("circom")
            if not snarkjs:
                missing.append("snarkjs")
            if not node:
                missing.append("node")
            logger.warning(f"Missing dependencies: {', '.join(missing)}")
            return False
        return True
        
    def _run_command(self, cmd: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Run a shell command."""
        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {' '.join(cmd)}")
            logger.error(f"Error: {e.stderr}")
            raise
            
    def compile_circuit(self, circuit_name: str) -> bool:
        """Compile a Circom circuit to R1CS and WASM."""
        if circuit_name not in self.circuits:
            logger.error(f"Unknown circuit: {circuit_name}")
            return False
            
        circuit = self.circuits[circuit_name]
        
        # Check if already compiled
        if circuit.r1cs_path.exists() and circuit.wasm_path.exists():
            logger.info(f"Circuit {circuit_name} already compiled")
            return True
            
        # Create build directory
        circuit.build_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if circuit file exists
        if not circuit.circuit_path.exists():
            logger.warning(f"Circuit file not found: {circuit.circuit_path}")
            # Return True to fall back to mock implementation
            return True
            
        try:
            # Compile circuit
            logger.info(f"Compiling circuit {circuit_name}")
            self._run_command([
                "circom",
                str(circuit.circuit_path),
                "--r1cs",
                "--wasm",
                "--output", str(circuit.build_dir)
            ], cwd=circuit.circuit_path.parent)
            
            return True
        except Exception as e:
            logger.error(f"Failed to compile circuit: {e}")
            return False
            
    def setup_trusted_setup(self, circuit_name: str, tau_power: int = 12) -> bool:
        """Perform trusted setup for a circuit."""
        if circuit_name not in self.circuits:
            return False
            
        circuit = self.circuits[circuit_name]
        
        # Check if already set up
        if circuit.zkey_path.exists() and circuit.vkey_path.exists():
            logger.info(f"Trusted setup already complete for {circuit_name}")
            return True
            
        # Check if R1CS exists
        if not circuit.r1cs_path.exists():
            logger.warning(f"R1CS not found for {circuit_name}, using mock setup")
            return True
            
        try:
            # Powers of tau ceremony (simplified for development)
            pot_0 = circuit.build_dir / f"pot{tau_power}_0000.ptau"
            pot_final = circuit.build_dir / f"pot{tau_power}_final.ptau"
            
            if not pot_final.exists():
                # New ceremony
                self._run_command([
                    "snarkjs", "powersoftau", "new", "bn128", str(tau_power), str(pot_0)
                ], cwd=circuit.build_dir)
                
                # Contribute (in production, multiple parties would contribute)
                self._run_command([
                    "snarkjs", "powersoftau", "contribute",
                    str(pot_0), str(pot_final),
                    "--name", "First contribution", "-e", "random_entropy"
                ], cwd=circuit.build_dir)
            
            # Generate zkey
            zkey_0 = circuit.build_dir / f"{circuit.name}_0000.zkey"
            self._run_command([
                "snarkjs", "groth16", "setup",
                str(circuit.r1cs_path), str(pot_final), str(zkey_0)
            ], cwd=circuit.build_dir)
            
            # Export verification key
            self._run_command([
                "snarkjs", "zkey", "export", "verificationkey",
                str(zkey_0), str(circuit.vkey_path)
            ], cwd=circuit.build_dir)
            
            # Mark as final (in production, would have more contributions)
            shutil.copy(zkey_0, circuit.zkey_path)
            
            logger.info(f"Trusted setup complete for {circuit_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to perform trusted setup: {e}")
            return False
            
    def generate_witness(self, circuit_name: str, inputs: Dict[str, Any]) -> Optional[Path]:
        """Generate witness for the circuit."""
        if circuit_name not in self.circuits:
            return None
            
        circuit = self.circuits[circuit_name]
        
        # Check if WASM exists
        if not circuit.wasm_path.exists():
            logger.warning(f"WASM not found for {circuit_name}")
            return None
            
        try:
            # Write input to file
            input_path = circuit.build_dir / "input.json"
            input_path.write_text(json.dumps(inputs))
            
            # Generate witness
            witness_path = circuit.build_dir / "witness.wtns"
            witness_generator = circuit.wasm_path.parent / "generate_witness.js"
            
            if witness_generator.exists():
                self._run_command([
                    "node", str(witness_generator),
                    str(circuit.wasm_path), str(input_path), str(witness_path)
                ], cwd=circuit.build_dir)
                
                return witness_path
            else:
                logger.warning(f"Witness generator not found: {witness_generator}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate witness: {e}")
            return None
            
    def generate_proof(
        self, 
        circuit_name: str, 
        public_inputs: Dict[str, Any],
        private_inputs: Dict[str, Any]
    ) -> Optional[Tuple[Dict, Dict]]:
        """Generate a zero-knowledge proof."""
        
        # Check dependencies
        if not self.check_dependencies():
            logger.info("Circom dependencies not available, falling back to mock proof")
            return None
            
        if circuit_name not in self.circuits:
            logger.error(f"Unknown circuit: {circuit_name}")
            return None
            
        circuit = self.circuits[circuit_name]
        
        # Ensure circuit is compiled and setup
        if not self.compile_circuit(circuit_name):
            logger.info("Circuit compilation failed, using mock proof")
            return None
            
        if not self.setup_trusted_setup(circuit_name):
            logger.info("Trusted setup failed, using mock proof")
            return None
            
        # Prepare inputs
        all_inputs = {}
        all_inputs.update(public_inputs)
        all_inputs.update(private_inputs)
        
        # Convert inputs to proper format
        formatted_inputs = self._format_inputs(circuit_name, all_inputs)
        
        # Generate witness
        witness_path = self.generate_witness(circuit_name, formatted_inputs)
        if not witness_path:
            logger.info("Witness generation failed, using mock proof")
            return None
            
        try:
            # Generate proof
            proof_path = circuit.build_dir / "proof.json"
            public_path = circuit.build_dir / "public.json"
            
            self._run_command([
                "snarkjs", "groth16", "prove",
                str(circuit.zkey_path), str(witness_path),
                str(proof_path), str(public_path)
            ], cwd=circuit.build_dir)
            
            # Read proof and public signals
            proof = json.loads(proof_path.read_text())
            public = json.loads(public_path.read_text())
            
            logger.info(f"Generated real ZK proof for {circuit_name}")
            return proof, public
            
        except Exception as e:
            logger.error(f"Failed to generate proof: {e}")
            return None
            
    def verify_proof(
        self, 
        circuit_name: str,
        proof: Dict,
        public_signals: list
    ) -> bool:
        """Verify a zero-knowledge proof."""
        
        if circuit_name not in self.circuits:
            return False
            
        circuit = self.circuits[circuit_name]
        
        # Check if verification key exists
        if not circuit.vkey_path.exists():
            logger.warning("Verification key not found")
            return False
            
        try:
            # Write proof and public signals to temp files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(proof, f)
                proof_file = Path(f.name)
                
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(public_signals, f)
                public_file = Path(f.name)
                
            # Verify proof
            result = self._run_command([
                "snarkjs", "groth16", "verify",
                str(circuit.vkey_path), str(public_file), str(proof_file)
            ], cwd=circuit.build_dir)
            
            # Clean up temp files
            proof_file.unlink()
            public_file.unlink()
            
            # Check verification result
            return "OK" in result.stdout or "true" in result.stdout.lower()
            
        except Exception as e:
            logger.error(f"Failed to verify proof: {e}")
            return False
            
    def _format_inputs(self, circuit_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Format inputs for the circuit."""
        formatted = {}
        
        if circuit_name == "variant_presence":
            # Convert variant data to field elements
            formatted["variant_hash"] = self._to_field_element(inputs.get("variant_hash", ""))
            formatted["reference_hash"] = self._to_field_element(inputs.get("reference_hash", ""))
            formatted["commitment_root"] = self._to_field_element(inputs.get("commitment_root", ""))
            
            # Private inputs
            variant_data = inputs.get("variant_data", {})
            formatted["chr"] = self._chr_to_int(variant_data.get("chr", "chr1"))
            formatted["position"] = int(variant_data.get("pos", 0))
            formatted["ref_allele"] = self._allele_to_int(variant_data.get("ref", "A"))
            formatted["alt_allele"] = self._allele_to_int(variant_data.get("alt", "G"))
            
            # Merkle proof (simplified)
            merkle_proof = inputs.get("merkle_proof", [])
            formatted["merkle_proof"] = [self._to_field_element(p) for p in merkle_proof[:20]]
            formatted["merkle_proof"].extend([0] * (20 - len(formatted["merkle_proof"])))
            
            formatted["merkle_indices"] = [0] * 20
            formatted["witness_randomness"] = self._to_field_element(
                inputs.get("witness_randomness", "0")
            )
            
        elif circuit_name == "diabetes_risk":
            # Public inputs
            formatted["glucose_threshold"] = int(inputs.get("glucose_threshold", 126))
            formatted["risk_threshold"] = int(inputs.get("risk_threshold", 0.75) * 1000)
            formatted["result_commitment"] = self._to_field_element(
                inputs.get("result_commitment", "")
            )
            
            # Private inputs
            formatted["glucose_reading"] = int(inputs.get("glucose_reading", 140))
            formatted["risk_score"] = int(inputs.get("risk_score", 0.82) * 1000)
            formatted["witness_randomness"] = self._to_field_element(
                inputs.get("witness_randomness", "0")
            )
            
        return formatted
        
    def _to_field_element(self, value: str) -> int:
        """Convert a string to a field element."""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            if value.startswith("0x"):
                return int(value, 16)
            # Hash and convert to field element
            hash_bytes = hashlib.sha256(value.encode()).digest()
            # Take first 31 bytes to fit in field (BN254 has ~254-bit field)
            return int.from_bytes(hash_bytes[:31], 'big')
        return 0
        
    def _chr_to_int(self, chr_str: str) -> int:
        """Convert chromosome string to integer."""
        if chr_str.startswith("chr"):
            chr_str = chr_str[3:]
        if chr_str == "X":
            return 23
        elif chr_str == "Y":
            return 24
        elif chr_str == "M" or chr_str == "MT":
            return 25
        else:
            try:
                return int(chr_str)
            except:
                return 0
                
    def _allele_to_int(self, allele: str) -> int:
        """Convert allele to integer."""
        mapping = {"A": 1, "T": 2, "C": 3, "G": 4}
        return mapping.get(allele.upper(), 0)