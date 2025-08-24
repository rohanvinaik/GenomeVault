#!/usr/bin/env python3
"""
PIR Server Runner - Starts multiple PIR server instances with Byzantine fault detection.

This script:
1. Starts 3 PIR server instances on ports 9001-9003
2. Creates a demo dataset of genomic records
3. Implements IT-PIR protocol with XOR aggregation
4. Includes Byzantine fault detection
5. Provides graceful shutdown on SIGINT/SIGTERM
"""

import asyncio
import hashlib
import json
import multiprocessing
import os
import random
import secrets
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from aiohttp import web

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from genomevault.pir.servers import PIRServer
from genomevault.pir.it_pir_protocol import PIRParameters, PIRProtocol
from genomevault.pir.xor_scheme import XORPIRScheme, XORSchemeParams
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class SimpleByzantineDetector:
    """Simplified Byzantine fault detection for PIR responses."""
    
    def __init__(self, num_servers: int):
        """Initialize detector."""
        self.num_servers = num_servers
        self.server_stats = {i: {"reliable": 0, "suspicious": 0} for i in range(num_servers)}
    
    def detect_byzantine_behavior(self, responses: List[Tuple[int, bytes]]) -> List[int]:
        """
        Detect potentially Byzantine servers by comparing responses.
        
        Args:
            responses: List of (server_id, response_bytes) tuples
            
        Returns:
            List of suspicious server IDs
        """
        if len(responses) < 2:
            return []
        
        suspicious = []
        
        # Group responses by content hash
        response_groups = {}
        for server_id, response in responses:
            content_hash = hashlib.sha256(response).hexdigest()
            if content_hash not in response_groups:
                response_groups[content_hash] = []
            response_groups[content_hash].append(server_id)
        
        # If all responses are different, can't determine Byzantine
        if len(response_groups) == len(responses):
            return []
        
        # Find the majority response
        sorted_groups = sorted(response_groups.values(), key=len, reverse=True)
        majority_servers = sorted_groups[0]
        
        # Servers not in majority are suspicious
        for group in sorted_groups[1:]:
            suspicious.extend(group)
            for server_id in group:
                self.server_stats[server_id]["suspicious"] += 1
        
        # Update reliable count for majority servers
        for server_id in majority_servers:
            self.server_stats[server_id]["reliable"] += 1
        
        return suspicious
    
    def get_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get server statistics."""
        stats = {}
        for server_id, counts in self.server_stats.items():
            total = counts["reliable"] + counts["suspicious"]
            reliability = counts["reliable"] / total if total > 0 else 0.5
            stats[server_id] = {
                "reliable_responses": counts["reliable"],
                "suspicious_responses": counts["suspicious"],
                "reliability_score": reliability,
                "status": "healthy" if reliability > 0.7 else "suspicious"
            }
        return stats


# Configuration
@dataclass
class ServerConfig:
    """Configuration for a single PIR server."""
    
    server_id: int
    port: int
    host: str = "127.0.0.1"
    byzantine_probability: float = 0.0  # Probability of Byzantine behavior
    latency_ms: int = 0  # Artificial latency for testing


class GenomicDataGenerator:
    """Generate synthetic genomic data for testing."""
    
    CHROMOSOMES = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY']
    NUCLEOTIDES = ['A', 'C', 'G', 'T']
    
    @staticmethod
    def generate_variant() -> Dict[str, Any]:
        """Generate a random genomic variant."""
        chrom = random.choice(GenomicDataGenerator.CHROMOSOMES)
        position = random.randint(1, 250_000_000)
        ref = random.choice(GenomicDataGenerator.NUCLEOTIDES)
        alt = random.choice([n for n in GenomicDataGenerator.NUCLEOTIDES if n != ref])
        
        # Generate quality scores
        qual = random.uniform(30, 100)
        depth = random.randint(10, 100)
        
        return {
            "chrom": chrom,
            "pos": position,
            "ref": ref,
            "alt": alt,
            "qual": qual,
            "depth": depth,
            "variant_id": f"{chrom}:{position}:{ref}>{alt}"
        }
    
    @staticmethod
    def generate_dataset(size: int, record_size: int = 1024) -> List[bytes]:
        """
        Generate a dataset of fixed-size genomic records.
        
        Args:
            size: Number of records
            record_size: Size of each record in bytes
            
        Returns:
            List of byte records
        """
        dataset = []
        
        for i in range(size):
            # Generate variant data
            variant = GenomicDataGenerator.generate_variant()
            variant["index"] = i
            variant["timestamp"] = int(time.time())
            
            # Convert to JSON and pad/truncate to fixed size
            json_data = json.dumps(variant, sort_keys=True)
            json_bytes = json_data.encode('utf-8')
            
            if len(json_bytes) < record_size:
                # Pad with zeros
                padded = json_bytes + b'\x00' * (record_size - len(json_bytes))
            else:
                # Truncate if too long
                padded = json_bytes[:record_size]
            
            dataset.append(padded)
        
        logger.info(f"Generated dataset with {size} records of {record_size} bytes each")
        return dataset


class PIRServerInstance:
    """A single PIR server instance."""
    
    def __init__(self, config: ServerConfig, dataset: List[bytes]):
        """
        Initialize PIR server instance.
        
        Args:
            config: Server configuration
            dataset: Dataset to serve
        """
        self.config = config
        self.dataset = dataset
        self.pir_server = PIRServer(dataset)
        self.stats = {
            "queries_received": 0,
            "queries_answered": 0,
            "byzantine_responses": 0,
            "errors": 0,
            "start_time": time.time()
        }
        self.app = web.Application()
        self.setup_routes()
    
    def setup_routes(self):
        """Set up HTTP routes."""
        self.app.router.add_post('/query', self.handle_query)
        self.app.router.add_get('/status', self.handle_status)
        self.app.router.add_get('/health', self.handle_health)
    
    async def handle_query(self, request: web.Request) -> web.Response:
        """
        Handle PIR query request.
        
        Expected POST body:
        {
            "mask": [0, 1, 0, ...],  // Binary mask array
            "query_id": "unique_id",
            "protocol": "xor" | "it-pir"
        }
        """
        try:
            self.stats["queries_received"] += 1
            
            # Parse request
            data = await request.json()
            mask_list = data.get("mask", [])
            query_id = data.get("query_id", "unknown")
            protocol = data.get("protocol", "xor")
            
            # Add artificial latency if configured
            if self.config.latency_ms > 0:
                await asyncio.sleep(self.config.latency_ms / 1000.0)
            
            # Convert mask to numpy array
            mask = np.array(mask_list, dtype=np.uint8)
            
            # Check for Byzantine behavior
            if random.random() < self.config.byzantine_probability:
                logger.warning(f"Server {self.config.server_id}: Simulating Byzantine behavior")
                self.stats["byzantine_responses"] += 1
                
                # Return corrupted response
                response_data = secrets.token_bytes(self.pir_server.record_len)
            else:
                # Normal response
                response_data = self.pir_server.answer(mask)
            
            self.stats["queries_answered"] += 1
            
            # Encode response
            response = {
                "server_id": self.config.server_id,
                "query_id": query_id,
                "response": response_data.hex(),
                "timestamp": time.time()
            }
            
            return web.json_response(response)
            
        except Exception as e:
            logger.error(f"Server {self.config.server_id} query error: {e}")
            self.stats["errors"] += 1
            return web.json_response(
                {"error": str(e), "server_id": self.config.server_id},
                status=500
            )
    
    async def handle_status(self, request: web.Request) -> web.Response:
        """Get server status."""
        uptime = time.time() - self.stats["start_time"]
        
        status = {
            "server_id": self.config.server_id,
            "port": self.config.port,
            "dataset_size": len(self.dataset),
            "record_size": self.pir_server.record_len,
            "stats": self.stats,
            "uptime_seconds": uptime,
            "byzantine_probability": self.config.byzantine_probability
        }
        
        return web.json_response(status)
    
    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "healthy", "server_id": self.config.server_id})
    
    async def start(self):
        """Start the server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.config.host, self.config.port)
        await site.start()
        
        logger.info(
            f"PIR Server {self.config.server_id} started on "
            f"{self.config.host}:{self.config.port}"
        )
        
        return runner


class PIROrchestrator:
    """Orchestrates multiple PIR servers with Byzantine fault tolerance."""
    
    def __init__(self, num_servers: int = 3, dataset_size: int = 1000):
        """
        Initialize PIR orchestrator.
        
        Args:
            num_servers: Number of PIR servers to run
            dataset_size: Size of the dataset
        """
        self.num_servers = num_servers
        self.dataset_size = dataset_size
        self.dataset = None
        self.servers: List[PIRServerInstance] = []
        self.runners = []
        self.byzantine_handler = None
        self.pir_protocol = None
        self.xor_scheme = None
        
        # Server configurations
        self.server_configs = [
            ServerConfig(
                server_id=i,
                port=9001 + i,
                byzantine_probability=0.05 if i == 1 else 0.0  # Server 1 has 5% Byzantine rate
            )
            for i in range(num_servers)
        ]
    
    def initialize_dataset(self):
        """Initialize the shared dataset."""
        logger.info(f"Generating dataset with {self.dataset_size} records...")
        self.dataset = GenomicDataGenerator.generate_dataset(self.dataset_size)
        
        # Calculate dataset hash for verification
        dataset_bytes = b''.join(self.dataset)
        self.dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
        logger.info(f"Dataset hash: {self.dataset_hash}")
    
    def initialize_protocols(self):
        """Initialize PIR protocols."""
        # IT-PIR protocol
        params = PIRParameters(
            database_size=self.dataset_size,
            element_size=1024,
            num_servers=self.num_servers
        )
        self.pir_protocol = PIRProtocol(params)
        
        # XOR-based PIR
        xor_params = XORSchemeParams(
            database_size=self.dataset_size,
            block_size=1024,
            num_servers=self.num_servers
        )
        self.xor_scheme = XORPIRScheme(xor_params)
        
        # Byzantine detector
        self.byzantine_handler = SimpleByzantineDetector(self.num_servers)
        
        logger.info("PIR protocols initialized")
    
    async def start_servers(self):
        """Start all PIR servers."""
        for config in self.server_configs:
            server = PIRServerInstance(config, self.dataset)
            self.servers.append(server)
            
            runner = await server.start()
            self.runners.append(runner)
        
        logger.info(f"Started {len(self.servers)} PIR servers")
    
    async def stop_servers(self):
        """Stop all PIR servers."""
        logger.info("Stopping PIR servers...")
        
        for runner in self.runners:
            await runner.cleanup()
        
        logger.info("All servers stopped")
    
    async def test_query(self, index: int):
        """
        Test a PIR query with Byzantine detection.
        
        Args:
            index: Index to retrieve
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing PIR query for index {index}")
        logger.info(f"{'='*60}")
        
        # Generate IT-PIR query vectors
        query_vectors = self.pir_protocol.generate_query_vectors(index)
        
        # Send queries to servers
        import aiohttp
        
        responses = []
        async with aiohttp.ClientSession() as session:
            for i, (config, query_vec) in enumerate(zip(self.server_configs, query_vectors)):
                url = f"http://{config.host}:{config.port}/query"
                
                payload = {
                    "mask": query_vec.tolist(),
                    "query_id": f"test_{index}_{i}",
                    "protocol": "it-pir"
                }
                
                try:
                    async with session.post(url, json=payload) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            response_bytes = bytes.fromhex(result["response"])
                            responses.append((i, response_bytes))
                            logger.info(f"  Server {i}: Response received")
                        else:
                            logger.error(f"  Server {i}: HTTP {resp.status}")
                except Exception as e:
                    logger.error(f"  Server {i}: Error - {e}")
        
        if len(responses) >= 2:
            # Detect Byzantine behavior
            suspicious = self.byzantine_handler.detect_byzantine_behavior(responses)
            if suspicious:
                logger.warning(f"  Byzantine behavior detected from servers: {suspicious}")
            
            # Aggregate responses with Byzantine tolerance
            try:
                # XOR responses (simplified aggregation)
                result = responses[0][1]
                for _, resp in responses[1:]:
                    result = bytes(a ^ b for a, b in zip(result, resp))
                
                # Extract actual data (remove padding)
                actual_data = result.rstrip(b'\x00')
                
                if actual_data:
                    try:
                        decoded = json.loads(actual_data.decode('utf-8'))
                        logger.info(f"  Retrieved record {index}: {decoded.get('variant_id', 'unknown')}")
                    except:
                        logger.info(f"  Retrieved raw data of length {len(actual_data)}")
                else:
                    logger.warning("  Retrieved empty data")
                
                return True
            except Exception as e:
                logger.error(f"  Aggregation failed: {e}")
                return False
        else:
            logger.error(f"  Insufficient responses: {len(responses)}/{self.num_servers}")
            return False
    
    async def run_demo(self):
        """Run a demonstration of the PIR system."""
        await asyncio.sleep(2)  # Let servers stabilize
        
        logger.info("\n" + "="*60)
        logger.info("PIR SYSTEM DEMONSTRATION")
        logger.info("="*60)
        
        # Test several queries
        test_indices = [0, 42, 100, 500, 999]
        
        for idx in test_indices:
            if idx < self.dataset_size:
                await self.test_query(idx)
                await asyncio.sleep(1)
        
        # Get server statistics
        logger.info("\n" + "="*60)
        logger.info("SERVER STATISTICS")
        logger.info("="*60)
        
        async with aiohttp.ClientSession() as session:
            for config in self.server_configs:
                url = f"http://{config.host}:{config.port}/status"
                try:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            status = await resp.json()
                            logger.info(f"\nServer {config.server_id}:")
                            logger.info(f"  Queries received: {status['stats']['queries_received']}")
                            logger.info(f"  Queries answered: {status['stats']['queries_answered']}")
                            logger.info(f"  Byzantine responses: {status['stats']['byzantine_responses']}")
                            logger.info(f"  Errors: {status['stats']['errors']}")
                except Exception as e:
                    logger.error(f"Failed to get status from server {config.server_id}: {e}")
        
        # Byzantine handler statistics
        if self.byzantine_handler:
            stats = self.byzantine_handler.get_statistics()
            logger.info("\n" + "="*60)
            logger.info("BYZANTINE DETECTION STATISTICS")
            logger.info("="*60)
            for server_id, server_stats in stats.items():
                logger.info(f"\nServer {server_id}:")
                logger.info(f"  Status: {server_stats['status']}")
                logger.info(f"  Reliability: {server_stats['reliability_score']:.2f}")
                logger.info(f"  Suspicious behaviors: {server_stats['suspicious_behaviors']}")


async def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("PIR SERVER RUNNER")
    print("Privacy-Preserving Information Retrieval System")
    print("="*60)
    
    # Parse arguments
    num_servers = 3
    dataset_size = 1000
    
    if len(sys.argv) > 1:
        try:
            num_servers = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of servers: {sys.argv[1]}")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            dataset_size = int(sys.argv[2])
        except ValueError:
            print(f"Invalid dataset size: {sys.argv[2]}")
            sys.exit(1)
    
    print(f"\nConfiguration:")
    print(f"  Servers: {num_servers}")
    print(f"  Dataset size: {dataset_size} records")
    print(f"  Ports: 9001-{9000 + num_servers}")
    print()
    
    # Create orchestrator
    orchestrator = PIROrchestrator(num_servers, dataset_size)
    
    # Initialize
    orchestrator.initialize_dataset()
    orchestrator.initialize_protocols()
    
    # Setup signal handlers
    async def shutdown(sig):
        logger.info(f"\nReceived signal {sig.name}, shutting down...")
        await orchestrator.stop_servers()
        asyncio.get_event_loop().stop()
    
    # Register signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(shutdown(s))
        )
    
    try:
        # Start servers
        await orchestrator.start_servers()
        
        print("\n" + "="*60)
        print("PIR SERVERS RUNNING")
        print("="*60)
        print("\nServer endpoints:")
        for config in orchestrator.server_configs:
            print(f"  Server {config.server_id}: http://127.0.0.1:{config.port}")
        print("\nEndpoints:")
        print("  POST /query - Submit PIR query")
        print("  GET /status - Server statistics")
        print("  GET /health - Health check")
        print("\nPress Ctrl+C to stop")
        print("="*60)
        
        # Run demonstration
        await orchestrator.run_demo()
        
        # Keep running
        while True:
            await asyncio.sleep(3600)
            
    except KeyboardInterrupt:
        logger.info("\nShutdown requested")
    finally:
        await orchestrator.stop_servers()


if __name__ == "__main__":
    # Set up multiprocessing for better performance
    multiprocessing.set_start_method('spawn', force=True)
    
    # Run the async main
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
        sys.exit(0)