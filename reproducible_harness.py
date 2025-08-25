#!/usr/bin/env python3
"""
Reproducible Benchmark Harness for GenomeVault
Ensures deterministic, portable, and verifiable benchmark execution
"""

import os
import sys
import json
import time
import hashlib
import subprocess
import platform
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import random

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Cryptographic imports
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

class ReproducibleHarness:
    """Deterministic and verifiable benchmark harness"""
    
    def __init__(self, seed: int = 42, output_dir: Optional[str] = None):
        self.seed = seed
        self.git_sha = self._get_git_sha()
        self.timestamp = datetime.utcnow().isoformat()
        self.run_id = f"{self.git_sha[:8]}_{self.timestamp.replace(':', '-').replace('.', '_')}"
        
        # Set up output directory
        if output_dir:
            self.output_base = Path(output_dir)
        else:
            self.output_base = Path("results")
        
        self.output_dir = self.output_base / self.git_sha / self.timestamp.replace(':', '-')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cryptographic keys
        self._init_keys()
        
        # Set all random seeds for reproducibility
        self._set_seeds()
        
        # Capture environment
        self.env_snapshot = self._capture_environment()
        
    def _get_git_sha(self) -> str:
        """Get current git SHA"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _init_keys(self):
        """Initialize or load RSA keys for signing"""
        key_dir = Path.home() / ".genomevault" / "keys"
        key_dir.mkdir(parents=True, exist_ok=True)
        
        private_key_path = key_dir / "benchmark_private.pem"
        public_key_path = key_dir / "benchmark_public.pem"
        
        if private_key_path.exists():
            # Load existing keys
            with open(private_key_path, 'rb') as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend()
                )
            with open(public_key_path, 'rb') as f:
                self.public_key = serialization.load_pem_public_key(
                    f.read(),
                    backend=default_backend()
                )
        else:
            # Generate new keys
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            self.public_key = self.private_key.public_key()
            
            # Save keys
            with open(private_key_path, 'wb') as f:
                f.write(self.private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            with open(public_key_path, 'wb') as f:
                f.write(self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
            
            print(f"📝 Generated new benchmark keys at {key_dir}")
    
    def _set_seeds(self):
        """Set all random seeds for reproducibility"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        if TORCH_AVAILABLE:
            torch.manual_seed(self.seed)
            
            # Set CUDA seeds if available
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
                
            # Deterministic operations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        # Set environment variables
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        os.environ['GENOMEVAULT_SEED'] = str(self.seed)
    
    def _capture_environment(self) -> Dict[str, Any]:
        """Capture complete environment snapshot"""
        env = {
            'timestamp': self.timestamp,
            'git_sha': self.git_sha,
            'seed': self.seed,
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
            },
            'environment_variables': {
                k: v for k, v in os.environ.items() 
                if 'GENOMEVAULT' in k or 'PYTHON' in k
            },
            'cuda_available': torch.cuda.is_available() if TORCH_AVAILABLE else False,
            'cuda_version': torch.version.cuda if TORCH_AVAILABLE and torch.cuda.is_available() else None,
        }
        
        # Try to get Metal info on macOS
        if platform.system() == 'Darwin':
            try:
                import mlx.core as mx
                env['metal_available'] = True
                env['metal_device'] = str(mx.default_device())
            except:
                env['metal_available'] = False
        
        return env
    
    def generate_sbom(self) -> Dict[str, Any]:
        """Generate Software Bill of Materials"""
        sbom = {
            'format': 'genomevault-sbom-1.0',
            'timestamp': self.timestamp,
            'git_sha': self.git_sha,
            'dependencies': {}
        }
        
        # Get pip packages
        try:
            result = subprocess.run(
                ['pip', 'freeze'],
                capture_output=True,
                text=True,
                check=True
            )
            for line in result.stdout.strip().split('\n'):
                if '==' in line:
                    pkg, version = line.split('==')
                    sbom['dependencies'][pkg] = version
        except:
            pass
        
        # Add system libraries
        sbom['system_libraries'] = {
            'python': platform.python_version(),
            'numpy': np.__version__,
        }
        
        if TORCH_AVAILABLE:
            sbom['system_libraries']['torch'] = torch.__version__
        
        return sbom
    
    def run_benchmark(self, benchmark_name: str, benchmark_func, **kwargs) -> Dict[str, Any]:
        """Run a benchmark with full reproducibility"""
        print(f"\n{'='*60}")
        print(f"Running: {benchmark_name}")
        print(f"SHA: {self.git_sha[:8]} | Seed: {self.seed}")
        print(f"{'='*60}")
        
        # Reset seeds before each benchmark
        self._set_seeds()
        
        # Create benchmark-specific directory
        bench_dir = self.output_dir / benchmark_name.replace(' ', '_').lower()
        bench_dir.mkdir(exist_ok=True)
        
        # Redirect stdout/stderr to capture logs
        log_file = bench_dir / "execution.log"
        
        start_time = time.time()
        
        try:
            # Run benchmark
            with open(log_file, 'w') as log:
                # Save original stdout/stderr
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                
                # Redirect to log file
                sys.stdout = log
                sys.stderr = log
                
                try:
                    results = benchmark_func(**kwargs)
                finally:
                    # Restore stdout/stderr
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
            
            execution_time = time.time() - start_time
            
            # Package results
            benchmark_result = {
                'name': benchmark_name,
                'status': 'success',
                'execution_time': execution_time,
                'seed': self.seed,
                'timestamp': datetime.utcnow().isoformat(),
                'results': results
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            benchmark_result = {
                'name': benchmark_name,
                'status': 'failed',
                'execution_time': execution_time,
                'seed': self.seed,
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
        
        # Save individual benchmark result
        with open(bench_dir / "result.json", 'w') as f:
            json.dump(benchmark_result, f, indent=2, default=str)
        
        print(f"✅ {benchmark_name}: {benchmark_result['status']} ({execution_time:.2f}s)")
        
        return benchmark_result
    
    def finalize(self, results: List[Dict[str, Any]]):
        """Finalize run with signing and verification"""
        
        # Create master results file
        master_results = {
            'run_id': self.run_id,
            'git_sha': self.git_sha,
            'timestamp': self.timestamp,
            'seed': self.seed,
            'environment': self.env_snapshot,
            'benchmarks': results
        }
        
        # Save results
        results_file = self.output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(master_results, f, indent=2, default=str)
        
        # Save environment snapshot
        with open(self.output_dir / "environment.json", 'w') as f:
            json.dump(self.env_snapshot, f, indent=2)
        
        # Save SBOM
        sbom = self.generate_sbom()
        with open(self.output_dir / "sbom.json", 'w') as f:
            json.dump(sbom, f, indent=2)
        
        # Sign results
        signature = self._sign_results(master_results)
        with open(self.output_dir / "signature.sig", 'wb') as f:
            f.write(signature)
        
        # Save public key for verification
        with open(self.output_dir / "public_key.pem", 'wb') as f:
            f.write(self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))
        
        # Create verification script
        self._create_verification_script()
        
        print(f"\n✅ Results saved to: {self.output_dir}")
        print(f"   - results.json (signed)")
        print(f"   - environment.json")
        print(f"   - sbom.json")
        print(f"   - signature.sig")
        print(f"   - verify.py (verification script)")
    
    def _sign_results(self, results: Dict[str, Any]) -> bytes:
        """Sign results with private key"""
        # Serialize results deterministically
        results_json = json.dumps(results, sort_keys=True, default=str)
        results_bytes = results_json.encode('utf-8')
        
        # Sign
        signature = self.private_key.sign(
            results_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature
    
    def _create_verification_script(self):
        """Create standalone verification script"""
        verify_script = '''#!/usr/bin/env python3
"""Verify benchmark results signature"""

import json
import sys
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

def verify_results(results_dir: Path):
    """Verify signed benchmark results"""
    
    # Load files
    with open(results_dir / "results.json", 'r') as f:
        results = json.load(f)
    
    with open(results_dir / "signature.sig", 'rb') as f:
        signature = f.read()
    
    with open(results_dir / "public_key.pem", 'rb') as f:
        public_key = serialization.load_pem_public_key(
            f.read(),
            backend=default_backend()
        )
    
    # Serialize results deterministically
    results_json = json.dumps(results, sort_keys=True, default=str)
    results_bytes = results_json.encode('utf-8')
    
    # Verify signature
    try:
        public_key.verify(
            signature,
            results_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        print("✅ Signature verified successfully!")
        print(f"   Run ID: {results['run_id']}")
        print(f"   Git SHA: {results['git_sha']}")
        print(f"   Timestamp: {results['timestamp']}")
        return True
    except InvalidSignature:
        print("❌ Invalid signature! Results may have been tampered with.")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify.py <results_directory>")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)
    
    success = verify_results(results_dir)
    sys.exit(0 if success else 1)
'''
        
        with open(self.output_dir / "verify.py", 'w') as f:
            f.write(verify_script)
        
        # Make executable
        os.chmod(self.output_dir / "verify.py", 0o755)


# Benchmark implementations
def benchmark_hdc_encoding(seed: int = 42) -> Dict[str, Any]:
    """Benchmark HDC encoding with deterministic execution"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from genomevault.hypervector_transform.encoding import HypervectorEncoder, HypervectorConfig
    from genomevault.core.constants import OmicsType
    
    np.random.seed(seed)
    
    results = {}
    dimensions = [4096, 8192, 16384]
    
    for dim in dimensions:
        config = HypervectorConfig(dimension=dim)
        encoder = HypervectorEncoder(config=config)
        
        # Generate deterministic test data
        data = np.random.randn(100).astype(np.float32)
        
        # Time encoding
        start = time.time()
        encoded = encoder.encode(data, OmicsType.GENOMIC)
        encoding_time = (time.time() - start) * 1000
        
        # Convert to numpy for stats
        if hasattr(encoded, 'numpy'):
            encoded = encoded.numpy()
        elif hasattr(encoded, 'cpu'):
            encoded = encoded.cpu().numpy()
        
        sparsity = np.mean(encoded == 0)
        
        results[f'{dim}D'] = {
            'encoding_time_ms': encoding_time,
            'sparsity': float(sparsity),
            'dimension': dim
        }
    
    return results


def benchmark_pir_queries(seed: int = 42) -> Dict[str, Any]:
    """Benchmark PIR queries with deterministic execution"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from genomevault.pir.accelerated_pir import AcceleratedPIREngine
    
    np.random.seed(seed)
    
    results = {}
    database_sizes = [100, 1000, 10000]
    
    for size in database_sizes:
        # Create deterministic database
        database = np.random.bytes(size * 100)  # 100 bytes per record
        
        # Initialize PIR
        engine = AcceleratedPIREngine(database, n_servers=3)
        
        # Query middle record
        query_idx = size // 2
        
        start = time.time()
        result = engine.query(query_idx)
        query_time = (time.time() - start) * 1000
        
        results[f'{size}_records'] = {
            'query_time_ms': query_time,
            'database_size': size,
            'record_size': 100
        }
    
    return results


def main():
    """Main execution with Docker support detection"""
    
    # Check if running in Docker
    in_docker = os.path.exists('/.dockerenv') or os.environ.get('GENOMEVAULT_DOCKER') == '1'
    
    # Use appropriate output directory
    if in_docker:
        output_dir = "/genomevault/results"
    else:
        output_dir = None  # Use default
    
    # Initialize harness
    harness = ReproducibleHarness(seed=42, output_dir=output_dir)
    
    # Run benchmarks
    results = []
    
    # HDC Encoding
    result = harness.run_benchmark(
        "HDC Encoding",
        benchmark_hdc_encoding,
        seed=harness.seed
    )
    results.append(result)
    
    # PIR Queries
    result = harness.run_benchmark(
        "PIR Queries", 
        benchmark_pir_queries,
        seed=harness.seed
    )
    results.append(result)
    
    # Finalize and sign
    harness.finalize(results)
    
    print("\n" + "="*60)
    print("REPRODUCIBLE BENCHMARK COMPLETE")
    print("="*60)
    print(f"Results: {harness.output_dir}")
    print(f"Verify: python {harness.output_dir}/verify.py {harness.output_dir}")


if __name__ == "__main__":
    main()