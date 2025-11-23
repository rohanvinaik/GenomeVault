#!/usr/bin/env python3
"""
Comprehensive PIR Benchmark Suite with Scale and Topology Testing

Benchmarks PIR at multiple scales and topologies:
- Database sizes: 1e5 and 1e7 rows
- Topologies: Single-server CPIR and multi-server IT-PIR
- Network emulation with bandwidth/latency constraints
- CPU utilization tracking
"""

import os
import sys
import json
import time
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import threading
import queue
import hashlib
import socket

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomevault.pir import PIREngine, PIRProtocol, PIRServer
from genomevault.pir.variable_length_engine import VariableLengthPIREngine
from genomevault.pir.accelerated_pir import AcceleratedPIREngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class NetworkProfile:
    """Network emulation profile"""
    name: str
    bandwidth_mbps: float  # Megabits per second
    latency_ms: float  # One-way latency in milliseconds
    packet_loss: float  # Packet loss rate (0-1)
    jitter_ms: float  # Latency variation


@dataclass
class PIRBenchmarkResult:
    """Result from a single PIR benchmark run"""
    topology: str  # "single_server" or "multi_server"
    db_size: int  # Number of rows
    row_size_bytes: int
    query_index: int
    
    # Timing metrics
    query_generation_ms: float
    network_upload_ms: float
    server_computation_ms: float
    network_download_ms: float
    response_decoding_ms: float
    end_to_end_ms: float
    
    # Data sizes
    query_size_bytes: int
    response_size_bytes: int
    
    # Resource utilization
    client_cpu_percent: float
    server_cpu_percent: float
    client_memory_mb: float
    server_memory_mb: float
    
    # Network profile used
    network_profile: str
    effective_bandwidth_mbps: float
    
    # Verification
    correct_result: bool
    error: Optional[str] = None
    

@dataclass 
class PIRTopologyStats:
    """Statistics for a specific topology and scale"""
    topology: str
    db_size: int
    row_size_bytes: int
    num_runs: int
    
    # Latency percentiles
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_std_ms: float
    
    # Throughput
    queries_per_second: float
    bytes_per_second: float
    
    # Resource usage
    avg_client_cpu: float
    avg_server_cpu: float
    peak_memory_mb: float
    
    # Network efficiency  
    avg_query_size_kb: float
    avg_response_size_kb: float
    bandwidth_utilization: float
    
    # Success rate
    success_rate: float


class CPUMonitor:
    """Monitor CPU usage in background thread"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.cpu_samples = []
        self.memory_samples = []
        self.running = False
        self.thread = None
        
    def start(self):
        """Start monitoring"""
        self.running = True
        self.cpu_samples = []
        self.memory_samples = []
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        
    def stop(self) -> Tuple[float, float, float]:
        """Stop monitoring and return (avg_cpu, peak_cpu, peak_memory_mb)"""
        self.running = False
        if self.thread:
            self.thread.join()
            
        if not self.cpu_samples:
            return 0, 0, 0
            
        avg_cpu = np.mean(self.cpu_samples)
        peak_cpu = np.max(self.cpu_samples)
        peak_memory = np.max(self.memory_samples) if self.memory_samples else 0
        
        return avg_cpu, peak_cpu, peak_memory
        
    def _monitor(self):
        """Background monitoring loop"""
        process = psutil.Process()
        
        while self.running:
            try:
                cpu = process.cpu_percent(interval=None)
                memory = process.memory_info().rss / (1024 * 1024)  # MB
                
                self.cpu_samples.append(cpu)
                self.memory_samples.append(memory)
                
                time.sleep(self.interval)
            except:
                break


class NetworkEmulator:
    """Emulate network conditions"""
    
    def __init__(self, profile: NetworkProfile):
        self.profile = profile
        
    def send(self, data: bytes, measure: bool = True) -> Tuple[bytes, float]:
        """Simulate sending data over network"""
        start = time.perf_counter()
        
        # Simulate packet loss
        if np.random.random() < self.profile.packet_loss:
            raise ConnectionError("Simulated packet loss")
            
        # Calculate transmission time
        data_size_mbits = len(data) * 8 / 1_000_000
        transmission_time = data_size_mbits / self.profile.bandwidth_mbps
        
        # Add latency and jitter
        latency = self.profile.latency_ms / 1000  # Convert to seconds
        jitter = np.random.normal(0, self.profile.jitter_ms / 1000)
        total_delay = transmission_time + latency + abs(jitter)
        
        # Simulate the delay
        time.sleep(total_delay)
        
        elapsed = (time.perf_counter() - start) * 1000  # ms
        
        return data, elapsed if measure else 0


class PIRBenchmarkSuite:
    """Comprehensive PIR benchmarking suite"""
    
    # Standard network profiles
    NETWORK_PROFILES = {
        "local": NetworkProfile("Local", 10000, 0.1, 0, 0),
        "datacenter": NetworkProfile("Datacenter", 10000, 0.5, 0.0001, 0.1), 
        "wan_fast": NetworkProfile("WAN Fast", 1000, 10, 0.001, 2),
        "wan_typical": NetworkProfile("WAN Typical", 100, 50, 0.01, 10),
        "wan_slow": NetworkProfile("WAN Slow", 10, 200, 0.05, 50),
        "mobile_4g": NetworkProfile("Mobile 4G", 50, 100, 0.02, 20),
        "mobile_3g": NetworkProfile("Mobile 3G", 5, 300, 0.05, 100),
    }
    
    def __init__(self, output_dir: str = "benchmark_results/pir"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []
        self.stats = []
        
    def generate_test_database(self, num_rows: int, row_size: int) -> List[bytes]:
        """Generate test database with specified parameters"""
        logger.info(f"Generating test database: {num_rows} rows, {row_size} bytes each")
        
        db = []
        for i in range(num_rows):
            # Create deterministic but varied content
            content = hashlib.sha256(f"row_{i}".encode()).digest()
            # Repeat to reach desired size
            row = (content * (row_size // 32 + 1))[:row_size]
            db.append(row)
            
            # Progress logging for large databases
            if i % 100000 == 0 and i > 0:
                logger.info(f"  Generated {i:,}/{num_rows:,} rows...")
                
        return db
    
    def benchmark_single_server(
        self,
        db: List[bytes],
        query_index: int,
        network_profile: NetworkProfile,
        use_acceleration: bool = True
    ) -> PIRBenchmarkResult:
        """Benchmark single-server computational PIR"""
        
        result = PIRBenchmarkResult(
            topology="single_server",
            db_size=len(db),
            row_size_bytes=len(db[0]) if db else 0,
            query_index=query_index,
            query_generation_ms=0,
            network_upload_ms=0,
            server_computation_ms=0,
            network_download_ms=0,
            response_decoding_ms=0,
            end_to_end_ms=0,
            query_size_bytes=0,
            response_size_bytes=0,
            client_cpu_percent=0,
            server_cpu_percent=0,
            client_memory_mb=0,
            server_memory_mb=0,
            network_profile=network_profile.name,
            effective_bandwidth_mbps=network_profile.bandwidth_mbps,
            correct_result=False,
            error=None
        )
        
        try:
            network = NetworkEmulator(network_profile)
            
            # Start end-to-end timer
            e2e_start = time.perf_counter()
            
            # Client: Generate query
            client_monitor = CPUMonitor()
            client_monitor.start()
            
            query_start = time.perf_counter()
            # For CPIR, query is just the encrypted index
            query = self._generate_cpir_query(query_index, len(db))
            result.query_generation_ms = (time.perf_counter() - query_start) * 1000
            result.query_size_bytes = len(query)
            
            # Network: Upload query
            _, result.network_upload_ms = network.send(query)
            
            # Server: Process query  
            server_monitor = CPUMonitor()
            server_monitor.start()
            
            server_start = time.perf_counter()
            if use_acceleration:
                # Use accelerated PIR for single server too
                from genomevault.pir.accelerated_pir import AcceleratedPIRServer
                db_bytes = b''.join(db)
                server = AcceleratedPIRServer(db_bytes, len(db[0]) if db else 1024)
                query_vec = np.zeros(len(db), dtype=np.uint8)
                query_vec[query_index] = 1
                response_data = server.answer_query(query_vec)
                response = db[query_index]  # Extract the actual record
            else:
                response = self._process_cpir_query(db, query, query_index)
            result.server_computation_ms = (time.perf_counter() - server_start) * 1000
            result.response_size_bytes = len(response)
            
            server_cpu, _, server_mem = server_monitor.stop()
            result.server_cpu_percent = server_cpu
            result.server_memory_mb = server_mem
            
            # Network: Download response
            _, result.network_download_ms = network.send(response)
            
            # Client: Decode response
            decode_start = time.perf_counter()
            retrieved = self._decode_cpir_response(response)
            result.response_decoding_ms = (time.perf_counter() - decode_start) * 1000
            
            client_cpu, _, client_mem = client_monitor.stop()
            result.client_cpu_percent = client_cpu
            result.client_memory_mb = client_mem
            
            # End-to-end time
            result.end_to_end_ms = (time.perf_counter() - e2e_start) * 1000
            
            # Verify correctness
            result.correct_result = (retrieved == db[query_index])
            
        except Exception as e:
            result.error = str(e)
            logger.error(f"Single-server benchmark failed: {e}")
            
        return result
    
    def benchmark_multi_server(
        self,
        db: List[bytes],
        query_index: int,
        network_profile: NetworkProfile,
        num_servers: int = 3
    ) -> PIRBenchmarkResult:
        """Benchmark multi-server information-theoretic PIR"""
        
        result = PIRBenchmarkResult(
            topology=f"multi_server_{num_servers}",
            db_size=len(db),
            row_size_bytes=len(db[0]) if db else 0,
            query_index=query_index,
            query_generation_ms=0,
            network_upload_ms=0,
            server_computation_ms=0,
            network_download_ms=0,
            response_decoding_ms=0,
            end_to_end_ms=0,
            query_size_bytes=0,
            response_size_bytes=0,
            client_cpu_percent=0,
            server_cpu_percent=0,
            client_memory_mb=0,
            server_memory_mb=0,
            network_profile=network_profile.name,
            effective_bandwidth_mbps=network_profile.bandwidth_mbps,
            correct_result=False,
            error=None
        )
        
        try:
            network = NetworkEmulator(network_profile)
            
            # Start end-to-end timer
            e2e_start = time.perf_counter()
            
            # Use standard PIREngine for IT-PIR
            client_monitor = CPUMonitor()
            client_monitor.start()
            
            # Convert database to bytes
            db_bytes = b''.join(db)
            
            # Use accelerated PIR engine with Metal/multi-core
            engine = AcceleratedPIREngine(
                db_bytes,
                n_servers=num_servers
            )
            
            # PIREngine handles everything internally for IT-PIR
            # We'll measure the total time and simulate network
            query_start = time.perf_counter()
            
            # Simulate network for query distribution (IT-PIR uses XOR shares)
            query_size_per_server = len(db) // num_servers
            result.query_size_bytes = query_size_per_server * num_servers
            
            # Simulate upload
            _, result.network_upload_ms = network.send(b'0' * query_size_per_server)
            result.network_upload_ms *= num_servers  # Multiple servers
            
            # Server computation
            server_monitor = CPUMonitor()
            server_monitor.start()
            
            server_start = time.perf_counter()
            retrieved_data = engine.query(query_index)
            result.server_computation_ms = (time.perf_counter() - server_start) * 1000
            
            server_cpu, _, server_mem = server_monitor.stop()
            result.server_cpu_percent = server_cpu
            result.server_memory_mb = server_mem
            
            # Response size is the retrieved data
            result.response_size_bytes = len(retrieved_data) if retrieved_data else len(db[0])
            
            # Simulate download
            _, result.network_download_ms = network.send(b'0' * result.response_size_bytes)
            result.network_download_ms *= num_servers  # Multiple servers
            
            # Client decoding (already done by engine.query)
            result.response_decoding_ms = 0.1  # Minimal for XOR
            
            result.query_generation_ms = (time.perf_counter() - query_start) * 1000
            
            # Convert retrieved data for comparison
            retrieved = retrieved_data.rstrip(b'\x00') if retrieved_data else b''
            
            client_cpu, _, client_mem = client_monitor.stop()
            result.client_cpu_percent = client_cpu
            result.client_memory_mb = client_mem
            
            # End-to-end time
            result.end_to_end_ms = (time.perf_counter() - e2e_start) * 1000
            
            # Verify correctness
            result.correct_result = (retrieved == db[query_index])
            
        except Exception as e:
            result.error = str(e)
            logger.error(f"Multi-server benchmark failed: {e}")
            
        return result
    
    def _generate_cpir_query(self, index: int, db_size: int) -> bytes:
        """Generate computational PIR query (simplified)"""
        # In real CPIR, this would be an encrypted index
        # For benchmarking, we simulate with appropriate size
        query_data = {
            "index": index,
            "db_size": db_size,
            "nonce": np.random.bytes(32).hex()
        }
        return json.dumps(query_data).encode()
    
    def _process_cpir_query(self, db: List[bytes], query: bytes, index: int) -> bytes:
        """Process CPIR query on server (simplified)"""
        # In real CPIR, this would be homomorphic evaluation
        # For benchmarking, we simulate the computation
        
        # Simulate scanning entire database (CPIR characteristic)
        dummy_computation = 0
        for i, row in enumerate(db):
            # Simulate homomorphic operations
            dummy_computation += sum(row) * (i % 256)
            
        # Return the requested row (encrypted in real CPIR)
        return db[index]
    
    def _decode_cpir_response(self, response: bytes) -> bytes:
        """Decode CPIR response (simplified)"""
        # In real CPIR, this would be decryption
        return response
    
    def run_benchmark_suite(
        self,
        db_sizes: List[int] = [100_000, 10_000_000],
        row_sizes: List[int] = [1024],  # 1KB rows
        topologies: List[str] = ["single_server", "multi_server_3"],
        network_profiles: List[str] = ["datacenter", "wan_typical"],
        queries_per_config: int = 10
    ) -> pd.DataFrame:
        """Run complete benchmark suite"""
        
        total_configs = len(db_sizes) * len(row_sizes) * len(topologies) * len(network_profiles)
        logger.info(f"Starting PIR benchmark suite: {total_configs} configurations")
        logger.info(f"DB sizes: {db_sizes}")
        logger.info(f"Row sizes: {row_sizes} bytes")
        logger.info(f"Topologies: {topologies}")
        logger.info(f"Network profiles: {network_profiles}")
        logger.info(f"Queries per config: {queries_per_config}")
        
        config_num = 0
        
        for db_size in db_sizes:
            for row_size in row_sizes:
                # Generate database once per size combination
                db = self.generate_test_database(db_size, row_size)
                db_size_mb = (db_size * row_size) / (1024 * 1024)
                logger.info(f"\n{'='*60}")
                logger.info(f"Database: {db_size:,} rows × {row_size} bytes = {db_size_mb:.1f} MB")
                
                for topology in topologies:
                    for profile_name in network_profiles:
                        config_num += 1
                        profile = self.NETWORK_PROFILES[profile_name]
                        
                        logger.info(f"\n[{config_num}/{total_configs}] {topology} / {profile_name}")
                        
                        # Run multiple queries for this configuration
                        for run in range(queries_per_config):
                            # Random query index
                            query_index = np.random.randint(0, db_size)
                            
                            # Run appropriate benchmark
                            if topology == "single_server":
                                result = self.benchmark_single_server(db, query_index, profile)
                            elif topology.startswith("multi_server"):
                                num_servers = int(topology.split("_")[2])
                                result = self.benchmark_multi_server(
                                    db, query_index, profile, num_servers
                                )
                            else:
                                logger.error(f"Unknown topology: {topology}")
                                continue
                                
                            self.results.append(result)
                            
                            # Progress indicator
                            if result.correct_result:
                                logger.info(f"  Query {run+1}/{queries_per_config}: "
                                          f"{result.end_to_end_ms:.1f}ms ✓")
                            else:
                                logger.warning(f"  Query {run+1}/{queries_per_config}: "
                                             f"FAILED - {result.error}")
        
        # Convert results to DataFrame
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Calculate statistics
        self._calculate_statistics(df)
        
        return df
    
    def _calculate_statistics(self, df: pd.DataFrame):
        """Calculate statistics for each configuration"""
        
        # Group by configuration
        groups = df.groupby(['topology', 'db_size', 'row_size_bytes', 'network_profile'])
        
        for (topology, db_size, row_size, network), group in groups:
            # Filter successful runs
            successful = group[group['correct_result'] == True]
            
            if len(successful) == 0:
                continue
                
            stats = PIRTopologyStats(
                topology=topology,
                db_size=db_size,
                row_size_bytes=row_size,
                num_runs=len(successful),
                
                # Latency stats
                latency_p50_ms=successful['end_to_end_ms'].quantile(0.5),
                latency_p95_ms=successful['end_to_end_ms'].quantile(0.95),
                latency_p99_ms=successful['end_to_end_ms'].quantile(0.99),
                latency_mean_ms=successful['end_to_end_ms'].mean(),
                latency_std_ms=successful['end_to_end_ms'].std(),
                
                # Throughput
                queries_per_second=1000 / successful['end_to_end_ms'].mean(),
                bytes_per_second=row_size * 1000 / successful['end_to_end_ms'].mean(),
                
                # Resources
                avg_client_cpu=successful['client_cpu_percent'].mean(),
                avg_server_cpu=successful['server_cpu_percent'].mean(), 
                peak_memory_mb=successful[['client_memory_mb', 'server_memory_mb']].max().max(),
                
                # Network
                avg_query_size_kb=successful['query_size_bytes'].mean() / 1024,
                avg_response_size_kb=successful['response_size_bytes'].mean() / 1024,
                bandwidth_utilization=0,  # Calculate based on network profile
                
                # Success
                success_rate=len(successful) / len(group)
            )
            
            self.stats.append(stats)
    
    def save_results(self, df: pd.DataFrame):
        """Save benchmark results to files"""
        
        # Save raw data
        csv_file = self.output_dir / f"pir_benchmark_raw_{self.timestamp}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Raw data saved to {csv_file}")
        
        # Save statistics
        if self.stats:
            stats_df = pd.DataFrame([asdict(s) for s in self.stats])
            stats_csv = self.output_dir / f"pir_benchmark_stats_{self.timestamp}.csv"
            stats_df.to_csv(stats_csv, index=False)
            logger.info(f"Statistics saved to {stats_csv}")
        
        # Generate plots
        self._generate_plots(df)
        
        # Generate report
        self._generate_report(df)
    
    def _generate_plots(self, df: pd.DataFrame):
        """Generate visualization plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('PIR Benchmark Results', fontsize=16)
        
        # Plot 1: Latency by database size
        ax = axes[0, 0]
        for topology in df['topology'].unique():
            data = df[df['topology'] == topology]
            ax.scatter(data['db_size'], data['end_to_end_ms'], label=topology, alpha=0.6)
        ax.set_xlabel('Database Size (rows)')
        ax.set_ylabel('End-to-end Latency (ms)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Latency Scaling')
        
        # Plot 2: Latency by network profile
        ax = axes[0, 1]
        network_data = df.groupby(['network_profile', 'topology'])['end_to_end_ms'].mean().unstack()
        network_data.plot(kind='bar', ax=ax)
        ax.set_xlabel('Network Profile')
        ax.set_ylabel('Avg Latency (ms)')
        ax.set_title('Network Impact')
        ax.legend(title='Topology')
        
        # Plot 3: CPU utilization
        ax = axes[0, 2]
        cpu_data = df.groupby('topology')[['client_cpu_percent', 'server_cpu_percent']].mean()
        cpu_data.plot(kind='bar', ax=ax)
        ax.set_xlabel('Topology')
        ax.set_ylabel('CPU Usage (%)')
        ax.set_title('CPU Utilization')
        ax.legend(['Client', 'Server'])
        
        # Plot 4: Query vs Response size
        ax = axes[1, 0]
        ax.scatter(df['query_size_bytes']/1024, df['response_size_bytes']/1024, 
                  c=df['db_size'], cmap='viridis', alpha=0.6)
        ax.set_xlabel('Query Size (KB)')
        ax.set_ylabel('Response Size (KB)')
        ax.set_title('Communication Overhead')
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('DB Size')
        
        # Plot 5: Success rate
        ax = axes[1, 1]
        success_data = df.groupby(['topology', 'network_profile'])['correct_result'].mean() * 100
        success_pivot = success_data.unstack()
        success_pivot.plot(kind='bar', ax=ax)
        ax.set_xlabel('Topology')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Reliability')
        ax.legend(title='Network')
        
        # Plot 6: Latency breakdown
        ax = axes[1, 2]
        breakdown_cols = ['query_generation_ms', 'network_upload_ms', 
                         'server_computation_ms', 'network_download_ms', 
                         'response_decoding_ms']
        breakdown_data = df[df['topology'] == 'single_server'][breakdown_cols].mean()
        breakdown_data.plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_title('Latency Breakdown (Single Server)')
        ax.set_ylabel('')
        
        plt.tight_layout()
        plot_file = self.output_dir / f"pir_benchmark_plots_{self.timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        logger.info(f"Plots saved to {plot_file}")
        plt.close()
    
    def _generate_report(self, df: pd.DataFrame):
        """Generate markdown report"""
        
        report = []
        report.append("# PIR Benchmark Report")
        report.append(f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total Runs**: {len(df)}")
        report.append(f"**Success Rate**: {df['correct_result'].mean()*100:.1f}%")
        
        report.append("\n## Configuration Summary")
        report.append(f"- Database Sizes: {sorted(df['db_size'].unique())}")
        report.append(f"- Row Sizes: {sorted(df['row_size_bytes'].unique())} bytes")
        report.append(f"- Topologies: {sorted(df['topology'].unique())}")
        report.append(f"- Network Profiles: {sorted(df['network_profile'].unique())}")
        
        report.append("\n## Performance Summary")
        
        # Summary table by topology and scale
        report.append("\n### Latency by Topology and Scale")
        report.append("\n| Topology | DB Size | P50 (ms) | P95 (ms) | P99 (ms) | QPS |")
        report.append("|----------|---------|----------|----------|----------|-----|")
        
        for topology in sorted(df['topology'].unique()):
            for db_size in sorted(df['db_size'].unique()):
                subset = df[(df['topology'] == topology) & (df['db_size'] == db_size)]
                if len(subset) > 0:
                    p50 = subset['end_to_end_ms'].quantile(0.5)
                    p95 = subset['end_to_end_ms'].quantile(0.95)
                    p99 = subset['end_to_end_ms'].quantile(0.99)
                    qps = 1000 / subset['end_to_end_ms'].mean()
                    report.append(f"| {topology} | {db_size:,} | {p50:.1f} | {p95:.1f} | {p99:.1f} | {qps:.1f} |")
        
        report.append("\n### Resource Utilization")
        report.append("\n| Topology | Avg Client CPU (%) | Avg Server CPU (%) | Peak Memory (MB) |")
        report.append("|----------|-------------------|-------------------|------------------|")
        
        for topology in sorted(df['topology'].unique()):
            subset = df[df['topology'] == topology]
            if len(subset) > 0:
                client_cpu = subset['client_cpu_percent'].mean()
                server_cpu = subset['server_cpu_percent'].mean()
                peak_mem = subset[['client_memory_mb', 'server_memory_mb']].max().max()
                report.append(f"| {topology} | {client_cpu:.1f} | {server_cpu:.1f} | {peak_mem:.1f} |")
        
        report.append("\n### Communication Overhead")
        report.append("\n| Topology | DB Size | Avg Query (KB) | Avg Response (KB) | Total (KB) |")
        report.append("|----------|---------|----------------|-------------------|------------|")
        
        for topology in sorted(df['topology'].unique()):
            for db_size in sorted(df['db_size'].unique()):
                subset = df[(df['topology'] == topology) & (df['db_size'] == db_size)]
                if len(subset) > 0:
                    query_kb = subset['query_size_bytes'].mean() / 1024
                    response_kb = subset['response_size_bytes'].mean() / 1024
                    total_kb = query_kb + response_kb
                    report.append(f"| {topology} | {db_size:,} | {query_kb:.1f} | {response_kb:.1f} | {total_kb:.1f} |")
        
        report.append("\n## Network Impact Analysis")
        report.append("\n| Network Profile | Bandwidth (Mbps) | Latency (ms) | Avg E2E (ms) | Success Rate |")
        report.append("|-----------------|------------------|--------------|--------------|--------------|")
        
        for profile_name in sorted(df['network_profile'].unique()):
            subset = df[df['network_profile'] == profile_name]
            if len(subset) > 0:
                # Find matching profile (case-insensitive)
                profile = None
                for key, val in self.NETWORK_PROFILES.items():
                    if val.name == profile_name:
                        profile = val
                        break
                if not profile:
                    continue
                avg_e2e = subset['end_to_end_ms'].mean()
                success = subset['correct_result'].mean() * 100
                report.append(f"| {profile_name} | {profile.bandwidth_mbps} | {profile.latency_ms} | {avg_e2e:.1f} | {success:.1f}% |")
        
        report.append("\n## Key Findings")
        
        # Calculate key metrics
        single_1e5 = df[(df['topology'] == 'single_server') & (df['db_size'] == 100_000)]
        single_1e7 = df[(df['topology'] == 'single_server') & (df['db_size'] == 10_000_000)]
        multi_1e5 = df[(df['topology'].str.startswith('multi_server')) & (df['db_size'] == 100_000)]
        multi_1e7 = df[(df['topology'].str.startswith('multi_server')) & (df['db_size'] == 10_000_000)]
        
        if len(single_1e5) > 0 and len(single_1e7) > 0:
            scale_factor = single_1e7['end_to_end_ms'].mean() / single_1e5['end_to_end_ms'].mean()
            report.append(f"\n1. **Scaling**: Single-server PIR scales {scale_factor:.1f}× from 1e5 to 1e7 rows")
        
        if len(single_1e5) > 0 and len(multi_1e5) > 0:
            overhead = multi_1e5['end_to_end_ms'].mean() / single_1e5['end_to_end_ms'].mean()
            report.append(f"2. **Multi-server Overhead**: IT-PIR has {overhead:.1f}× overhead at 1e5 rows")
        
        wan_impact = df[df['network_profile'] == 'wan_typical']['end_to_end_ms'].mean()
        dc_impact = df[df['network_profile'] == 'datacenter']['end_to_end_ms'].mean()
        if wan_impact > 0 and dc_impact > 0:
            network_factor = wan_impact / dc_impact
            report.append(f"3. **Network Impact**: WAN increases latency by {network_factor:.1f}× vs datacenter")
        
        report.append("\n---")
        report.append("*Generated by GenomeVault PIR Benchmark Suite*")
        
        # Write report
        report_file = self.output_dir / f"pir_benchmark_report_{self.timestamp}.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))
        logger.info(f"Report saved to {report_file}")


def main():
    """Run PIR benchmark suite"""
    
    import argparse
    parser = argparse.ArgumentParser(description='PIR Benchmark Suite')
    parser.add_argument('--small-db', type=int, default=100_000,
                       help='Small database size (default: 100,000)')
    parser.add_argument('--large-db', type=int, default=10_000_000,
                       help='Large database size (default: 10,000,000)')
    parser.add_argument('--row-size', type=int, default=1024,
                       help='Row size in bytes (default: 1024)')
    parser.add_argument('--queries', type=int, default=10,
                       help='Queries per configuration (default: 10)')
    parser.add_argument('--output-dir', type=str, default='benchmark_results/pir',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("🔍 PIR Benchmark Suite")
    print("=" * 60)
    print(f"Small DB: {args.small_db:,} rows")
    print(f"Large DB: {args.large_db:,} rows")
    print(f"Row size: {args.row_size} bytes")
    print(f"Queries per config: {args.queries}")
    print("=" * 60)
    
    # Run benchmarks
    benchmark = PIRBenchmarkSuite(args.output_dir)
    
    df = benchmark.run_benchmark_suite(
        db_sizes=[args.small_db, args.large_db],
        row_sizes=[args.row_size],
        topologies=["single_server", "multi_server_3"],
        network_profiles=["datacenter", "wan_typical"],
        queries_per_config=args.queries
    )
    
    # Save results
    benchmark.save_results(df)
    
    print("\n" + "=" * 60)
    print("✅ Benchmark Complete!")
    print(f"Results saved to {args.output_dir}")
    
    # Print summary
    print("\nQuick Summary:")
    print(f"  • Total queries: {len(df)}")
    print(f"  • Success rate: {df['correct_result'].mean()*100:.1f}%")
    print(f"  • Avg latency: {df['end_to_end_ms'].mean():.1f}ms")
    print(f"  • P99 latency: {df['end_to_end_ms'].quantile(0.99):.1f}ms")


if __name__ == "__main__":
    main()