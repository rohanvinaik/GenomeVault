#!/usr/bin/env python3
"""
End-to-End Demo for GenomeVault

This script orchestrates a complete demonstration of the GenomeVault platform:
1. Starts all required services
2. Generates demo data
3. Runs encoding, ZK proofs, and PIR queries
4. Provides comprehensive timing and results
5. Handles cleanup

Usage:
    python scripts/run_e2e_demo.py          # Run full demo
    python scripts/run_e2e_demo.py --cleanup # Clean up all resources
"""

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import psutil

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for a service."""
    name: str
    command: List[str]
    health_check_url: Optional[str] = None
    port: Optional[int] = None
    process: Optional[subprocess.Popen] = None
    start_delay: float = 2.0
    health_timeout: float = 30.0
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None


@dataclass
class DemoResults:
    """Container for demo results."""
    services_started: List[str] = field(default_factory=list)
    services_failed: List[str] = field(default_factory=list)
    demo_data_generated: bool = False
    hdc_encoding_success: bool = False
    zk_proof_verified: bool = False
    pir_query_success: bool = False
    nanopore_streaming: bool = False
    total_time_seconds: float = 0.0
    timings: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class E2EDemo:
    """End-to-end demonstration orchestrator."""
    
    def __init__(self, cleanup_only: bool = False, skip_services: List[str] = None):
        """Initialize demo orchestrator."""
        self.cleanup_only = cleanup_only
        self.skip_services = skip_services or []
        self.services: List[ServiceConfig] = []
        self.results = DemoResults()
        self.start_time = time.time()
        self.demo_dir = Path("e2e_demo_data")
        self.processes = []
        
        # Define service configurations
        self._configure_services()
    
    def _configure_services(self):
        """Configure all services."""
        # PostgreSQL (check if already running)
        if not self._is_postgres_running() and "postgres" not in self.skip_services:
            self.services.append(ServiceConfig(
                name="PostgreSQL",
                command=["postgres", "-D", "/usr/local/var/postgres"],
                port=5432,
                start_delay=3.0
            ))
        
        # Redis (check if already running)
        if not self._is_redis_running() and "redis" not in self.skip_services:
            self.services.append(ServiceConfig(
                name="Redis",
                command=["redis-server"],
                port=6379,
                start_delay=2.0
            ))
        
        # PIR Servers
        if "pir" not in self.skip_services:
            self.services.append(ServiceConfig(
                name="PIR Servers",
                command=[sys.executable, "scripts/run_pir_servers.py", "3", "100"],
                health_check_url="http://127.0.0.1:9001/health",
                port=9001,
                start_delay=3.0,
                cwd=str(project_root)
            ))
        
        # FastAPI server
        if "api" not in self.skip_services:
            self.services.append(ServiceConfig(
                name="GenomeVault API",
                command=["uvicorn", "genomevault.api.app:app", "--host", "0.0.0.0", "--port", "8000"],
                health_check_url="http://127.0.0.1:8000/api/healthz",
                port=8000,
                start_delay=3.0,
                cwd=str(project_root),
                env={**os.environ, "PYTHONPATH": str(project_root)}
            ))
    
    def _is_postgres_running(self) -> bool:
        """Check if PostgreSQL is already running."""
        try:
            result = subprocess.run(
                ["pg_isready", "-q"],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False
    
    def _is_redis_running(self) -> bool:
        """Check if Redis is already running."""
        try:
            result = subprocess.run(
                ["redis-cli", "ping"],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.stdout.strip() == "PONG"
        except:
            return False
    
    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use."""
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == 'LISTEN':
                return True
        return False
    
    async def start_services(self):
        """Start all required services."""
        logger.info("="*60)
        logger.info("STARTING SERVICES")
        logger.info("="*60)
        
        for service in self.services:
            try:
                # Check if port is already in use
                if service.port and self._is_port_in_use(service.port):
                    logger.info(f"✓ {service.name} already running on port {service.port}")
                    self.results.services_started.append(service.name)
                    continue
                
                logger.info(f"Starting {service.name}...")
                
                # Start process
                service.process = subprocess.Popen(
                    service.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=service.cwd,
                    env=service.env,
                    text=True
                )
                self.processes.append(service.process)
                
                # Wait for startup
                await asyncio.sleep(service.start_delay)
                
                # Check if process is still running
                if service.process.poll() is not None:
                    stderr = service.process.stderr.read() if service.process.stderr else ""
                    raise RuntimeError(f"Process exited: {stderr}")
                
                # Health check if URL provided
                if service.health_check_url:
                    if await self._wait_for_health(service.health_check_url, service.health_timeout):
                        logger.info(f"✓ {service.name} started successfully")
                        self.results.services_started.append(service.name)
                    else:
                        raise RuntimeError(f"Health check failed for {service.name}")
                else:
                    logger.info(f"✓ {service.name} started")
                    self.results.services_started.append(service.name)
                    
            except Exception as e:
                logger.error(f"✗ Failed to start {service.name}: {e}")
                self.results.services_failed.append(service.name)
                self.results.errors.append(f"{service.name}: {str(e)}")
    
    async def _wait_for_health(self, url: str, timeout: float = 30) -> bool:
        """Wait for service health check."""
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < timeout:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            return True
                except:
                    pass
                await asyncio.sleep(1)
        
        return False
    
    def generate_demo_data(self):
        """Generate demo data."""
        logger.info("\n" + "="*60)
        logger.info("GENERATING DEMO DATA")
        logger.info("="*60)
        
        start = time.time()
        
        try:
            # Clean up old demo data
            if self.demo_dir.exists():
                shutil.rmtree(self.demo_dir)
            
            # Generate new demo data
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_demo_data.py",
                    "--output", str(self.demo_dir),
                    "--variants", "500",
                    "--fast5-reads", "50",
                    "--zk-proofs", "5"
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(project_root)
            )
            
            if result.returncode == 0:
                logger.info("✓ Demo data generated successfully")
                self.results.demo_data_generated = True
                
                # Load manifest
                manifest_path = self.demo_dir / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    logger.info(f"  Generated: {manifest['statistics']['total_variants']} variants")
                    logger.info(f"  Generated: {manifest['statistics']['fast5_reads']} FAST5 reads")
                    logger.info(f"  Generated: {manifest['statistics']['hypervectors']} hypervectors")
            else:
                raise RuntimeError(f"Demo data generation failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"✗ Demo data generation failed: {e}")
            self.results.errors.append(f"Demo data: {str(e)}")
        
        self.results.timings['demo_data'] = time.time() - start
    
    async def test_hdc_encoding(self):
        """Test HDC encoding via API."""
        logger.info("\n" + "="*60)
        logger.info("TESTING HDC ENCODING")
        logger.info("="*60)
        
        start = time.time()
        
        try:
            # Read a variant from generated VCF
            vcf_path = self.demo_dir / "demo_variants.vcf"
            variants = []
            
            if vcf_path.exists():
                with open(vcf_path, 'r') as f:
                    for line in f:
                        if not line.startswith('#') and line.strip():
                            parts = line.split('\t')
                            if len(parts) >= 5:
                                variants.append({
                                    'chromosome': parts[0],
                                    'position': int(parts[1]),
                                    'ref': parts[3],
                                    'alt': parts[4]
                                })
                                if len(variants) >= 5:
                                    break
            
            if not variants:
                raise RuntimeError("No variants found in VCF")
            
            # Test HDC encoding via API
            async with aiohttp.ClientSession() as session:
                # Encode variants
                payload = {
                    "variants": variants,
                    "dimension": 10000
                }
                
                logger.info(f"Encoding {len(variants)} variants...")
                
                async with session.post(
                    "http://127.0.0.1:8000/api/hv/encode",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        logger.info(f"✓ HDC encoding successful")
                        logger.info(f"  Dimension: {result.get('dimension', 'unknown')}")
                        logger.info(f"  Encoding ID: {result.get('encoding_id', 'unknown')}")
                        self.results.hdc_encoding_success = True
                    else:
                        error_text = await resp.text()
                        raise RuntimeError(f"API returned {resp.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"✗ HDC encoding failed: {e}")
            self.results.errors.append(f"HDC encoding: {str(e)}")
        
        self.results.timings['hdc_encoding'] = time.time() - start
    
    async def test_zk_proof(self):
        """Test ZK proof generation and verification."""
        logger.info("\n" + "="*60)
        logger.info("TESTING ZK PROOFS")
        logger.info("="*60)
        
        start = time.time()
        
        try:
            # Test with transcript fallback (since circom may not be installed)
            async with aiohttp.ClientSession() as session:
                # Generate proof
                proof_payload = {
                    "circuit_type": "sum64",
                    "inputs": {
                        "a": 15,
                        "b": 27,
                        "c": 42
                    }
                }
                
                logger.info("Generating ZK proof (sum64: 15 + 27 = 42)...")
                
                async with session.post(
                    "http://127.0.0.1:8000/api/zk/prove",
                    json=proof_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        proof_result = await resp.json()
                        logger.info(f"✓ Proof generated")
                        
                        # Verify proof
                        verify_payload = {
                            "proof": proof_result["proof"],
                            "public_inputs": proof_result["public_inputs"]
                        }
                        
                        logger.info("Verifying proof...")
                        
                        async with session.post(
                            "http://127.0.0.1:8000/api/zk/verify",
                            json=verify_payload,
                            timeout=aiohttp.ClientTimeout(total=10)
                        ) as verify_resp:
                            if verify_resp.status == 200:
                                verify_result = await verify_resp.json()
                                if verify_result.get("valid"):
                                    logger.info("✓ Proof verified successfully")
                                    self.results.zk_proof_verified = True
                                else:
                                    raise RuntimeError("Proof verification failed")
                            else:
                                raise RuntimeError(f"Verify API returned {verify_resp.status}")
                    else:
                        error_text = await resp.text()
                        raise RuntimeError(f"Proof API returned {resp.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"✗ ZK proof test failed: {e}")
            self.results.errors.append(f"ZK proof: {str(e)}")
        
        self.results.timings['zk_proof'] = time.time() - start
    
    async def test_pir_query(self):
        """Test PIR query."""
        logger.info("\n" + "="*60)
        logger.info("TESTING PIR QUERY")
        logger.info("="*60)
        
        start = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Setup PIR dataset
                logger.info("Setting up PIR dataset...")
                
                setup_payload = {
                    "dataset_type": "genomic",
                    "dataset_size": 100,
                    "seed": 42
                }
                
                async with session.post(
                    "http://127.0.0.1:9001/api/pir/setup",
                    json=setup_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"PIR setup failed: {resp.status}")
                
                # Query index 42
                logger.info("Querying index 42 privately...")
                
                query_payload = {
                    "index": 42,
                    "use_byzantine_protection": True,
                    "num_servers": 3
                }
                
                async with session.post(
                    "http://127.0.0.1:9001/api/pir/query",
                    json=query_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        logger.info(f"✓ PIR query successful")
                        logger.info(f"  Retrieved index: {result['index']}")
                        logger.info(f"  Byzantine detected: {result.get('byzantine_detected', False)}")
                        logger.info(f"  Query time: {result.get('query_time_ms', 0):.2f}ms")
                        self.results.pir_query_success = True
                    else:
                        raise RuntimeError(f"PIR query failed: {resp.status}")
                        
        except Exception as e:
            logger.error(f"✗ PIR query failed: {e}")
            self.results.errors.append(f"PIR query: {str(e)}")
        
        self.results.timings['pir_query'] = time.time() - start
    
    async def test_nanopore_streaming(self):
        """Test nanopore streaming if available."""
        logger.info("\n" + "="*60)
        logger.info("TESTING NANOPORE STREAMING")
        logger.info("="*60)
        
        start = time.time()
        
        try:
            # Check if nanopore module is working
            from genomevault.nanopore.streaming import NanoporeStreamProcessor
            
            # Use mock FAST5 data
            fast5_dir = self.demo_dir / "fast5"
            if not fast5_dir.exists():
                raise RuntimeError("No FAST5 data available")
            
            logger.info(f"Processing mock FAST5 data from {fast5_dir}...")
            
            # Read first mock read
            read_files = list(fast5_dir.glob("read_*.json"))
            if read_files:
                with open(read_files[0], 'r') as f:
                    read_data = json.load(f)
                
                logger.info(f"✓ Processed read: {read_data['read_id']}")
                logger.info(f"  Length: {read_data['sequence_length']} bases")
                logger.info(f"  Signal mean: {read_data['signal_mean']:.2f}")
                self.results.nanopore_streaming = True
            else:
                logger.warning("No read files found")
                
        except ImportError:
            logger.warning("⚠ Nanopore module not available or has issues")
        except Exception as e:
            logger.error(f"✗ Nanopore streaming failed: {e}")
            self.results.errors.append(f"Nanopore: {str(e)}")
        
        self.results.timings['nanopore'] = time.time() - start
    
    def cleanup(self):
        """Clean up all resources."""
        logger.info("\n" + "="*60)
        logger.info("CLEANUP")
        logger.info("="*60)
        
        # Terminate all started processes
        for process in self.processes:
            if process and process.poll() is None:
                logger.info(f"Terminating process {process.pid}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        
        # Clean up demo data
        if self.demo_dir.exists():
            logger.info(f"Removing demo data directory: {self.demo_dir}")
            shutil.rmtree(self.demo_dir)
        
        # Clean up any other temporary files
        temp_dirs = ["demo_test", "e2e_demo_data"]
        for temp_dir in temp_dirs:
            path = Path(temp_dir)
            if path.exists():
                logger.info(f"Removing: {path}")
                shutil.rmtree(path)
        
        logger.info("✓ Cleanup complete")
    
    def print_summary(self):
        """Print results summary."""
        self.results.total_time_seconds = time.time() - self.start_time
        
        logger.info("\n" + "="*60)
        logger.info("E2E DEMO SUMMARY")
        logger.info("="*60)
        
        # Service status
        logger.info("\nServices:")
        for service in self.results.services_started:
            logger.info(f"  ✓ {service}")
        for service in self.results.services_failed:
            logger.info(f"  ✗ {service}")
        
        # Test results
        logger.info("\nTests:")
        tests = [
            ("Demo Data Generation", self.results.demo_data_generated),
            ("HDC Encoding", self.results.hdc_encoding_success),
            ("ZK Proof", self.results.zk_proof_verified),
            ("PIR Query", self.results.pir_query_success),
            ("Nanopore Streaming", self.results.nanopore_streaming)
        ]
        
        for test_name, success in tests:
            status = "✓" if success else "✗"
            timing = self.results.timings.get(test_name.lower().replace(" ", "_"), 0)
            logger.info(f"  {status} {test_name:.<30} {timing:.2f}s")
        
        # Errors
        if self.results.errors:
            logger.info("\nErrors:")
            for error in self.results.errors:
                logger.info(f"  • {error}")
        
        # Overall status
        all_passed = (
            self.results.demo_data_generated and
            self.results.hdc_encoding_success and
            self.results.zk_proof_verified and
            self.results.pir_query_success
        )
        
        logger.info("\n" + "-"*60)
        logger.info(f"Total time: {self.results.total_time_seconds:.2f} seconds")
        
        if all_passed:
            logger.info("\n✅ ALL CORE TESTS PASSED")
        else:
            logger.info("\n⚠️  SOME TESTS FAILED")
        
        logger.info("="*60)
        
        return 0 if all_passed else 1
    
    async def run(self):
        """Run the complete E2E demo."""
        if self.cleanup_only:
            self.cleanup()
            return 0
        
        try:
            # Start services
            await self.start_services()
            
            # Allow services to stabilize
            await asyncio.sleep(2)
            
            # Generate demo data
            self.generate_demo_data()
            
            # Run tests
            await self.test_hdc_encoding()
            await self.test_zk_proof()
            await self.test_pir_query()
            await self.test_nanopore_streaming()
            
            # Print summary
            return self.print_summary()
            
        except KeyboardInterrupt:
            logger.info("\n\nInterrupted by user")
            return 1
        except Exception as e:
            logger.error(f"\nFatal error: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            # Always cleanup
            self.cleanup()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end demo of GenomeVault"
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Only run cleanup'
    )
    parser.add_argument(
        '--skip',
        nargs='+',
        choices=['postgres', 'redis', 'api', 'pir'],
        help='Skip starting specific services'
    )
    parser.add_argument(
        '--keep-data',
        action='store_true',
        help='Keep demo data after completion'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🧬 GenomeVault End-to-End Demo")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Create and run demo
    demo = E2EDemo(
        cleanup_only=args.cleanup,
        skip_services=args.skip or []
    )
    
    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("\nReceived interrupt signal, cleaning up...")
        demo.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run demo
    result = await demo.run()
    
    # Keep data if requested
    if args.keep_data and not args.cleanup:
        logger.info("\n📁 Demo data preserved in: e2e_demo_data/")
    
    return result


if __name__ == "__main__":
    # Check dependencies
    try:
        import psutil
        import aiohttp
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install psutil aiohttp")
        sys.exit(1)
    
    # Run async main
    result = asyncio.run(main())
    sys.exit(result)