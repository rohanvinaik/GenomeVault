"""
PIR Topology Manager for optimal server configuration and shard management.

Based on Section 2.2.1: Implements optimal shard configuration, server selection,
and adaptive topology management for Private Information Retrieval.
"""

import math
import time
import random
import logging
from typing import List, Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class ServerType(Enum):
    """Server types in the PIR network."""
    LIGHT_NODE = "LN"  # Light Node - basic computation
    TRUSTED_SIGNATORY = "TS"  # Trusted Signatory - enhanced security
    FULL_NODE = "FN"  # Full Node - complete capabilities


class ServerStatus(Enum):
    """Server availability status."""
    AVAILABLE = "available"
    BUSY = "busy"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class Server:
    """Represents a PIR server in the network."""
    id: str
    type: ServerType
    endpoint: str
    latency_ms: float  # Average RTT in milliseconds
    reliability: float  # Historical uptime (0-1)
    capacity: int  # Max concurrent queries
    current_load: int = 0
    status: ServerStatus = ServerStatus.AVAILABLE
    last_ping: float = field(default_factory=time.time)
    failure_count: int = 0
    
    @property
    def is_available(self) -> bool:
        """Check if server is available for queries."""
        return (
            self.status == ServerStatus.AVAILABLE and
            self.current_load < self.capacity
        )
    
    @property
    def effective_latency(self) -> float:
        """Calculate effective latency considering load."""
        load_factor = 1 + (self.current_load / self.capacity)
        return self.latency_ms * load_factor


class ShardConfiguration(NamedTuple):
    """Configuration for PIR shard distribution."""
    total_shards: int
    light_nodes: int
    trusted_signatories: int
    expected_latency_ms: float
    failure_probability: float
    download_complexity: float  # O(N^(1/n))


@dataclass
class TopologyConfig:
    """Configuration for topology management."""
    target_failure_probability: float = 4e-4  # 4×10^-4
    min_trusted_signatories: int = 2
    max_shards: int = 10
    latency_sla_ms: float = 500.0
    server_timeout_ms: float = 5000.0
    health_check_interval_s: float = 30.0
    adaptive_threshold: float = 0.8  # Switch config if latency > threshold × SLA


class TopologyManager:
    """
    Manages PIR network topology for optimal performance.
    
    Implements:
    - Optimal shard configuration calculation
    - Server selection based on latency and reliability
    - Adaptive topology switching
    - Failure handling and recovery
    """
    
    # Preset configurations based on Section 2.2.1
    PRESET_CONFIGS = {
        "3_shard": ShardConfiguration(
            total_shards=3,
            light_nodes=1,
            trusted_signatories=2,
            expected_latency_ms=210,
            failure_probability=4e-4,
            download_complexity=1/3  # O(N^(1/3))
        ),
        "5_shard": ShardConfiguration(
            total_shards=5,
            light_nodes=3,
            trusted_signatories=2,
            expected_latency_ms=350,
            failure_probability=4e-4,
            download_complexity=1/5  # O(N^(1/5))
        ),
    }
    
    def __init__(self, config: Optional[TopologyConfig] = None):
        """Initialize topology manager."""
        self.config = config or TopologyConfig()
        self.servers: Dict[str, Server] = {}
        self.current_config: Optional[ShardConfiguration] = None
        self.latency_history: Dict[str, List[float]] = defaultdict(list)
        self.query_count = 0
        self.last_reconfiguration = time.time()
        
        logger.info("Initialized TopologyManager with target failure probability: %.2e", 
                   self.config.target_failure_probability)
    
    def register_server(self, server: Server) -> None:
        """Register a new server in the topology."""
        self.servers[server.id] = server
        logger.debug(f"Registered {server.type.value} server: {server.id}")
    
    def calculate_optimal_shards(
        self,
        database_size: int,
        server_reliability: float = 0.99
    ) -> int:
        """
        Calculate minimum shards needed for target failure probability.
        
        Formula: k_min = ⌈ln(φ)/ln(1-q)⌉
        where:
        - φ = target failure probability
        - q = individual server reliability
        
        Args:
            database_size: Size of database in entries
            server_reliability: Individual server uptime (0-1)
            
        Returns:
            Minimum number of shards required
        """
        phi = self.config.target_failure_probability
        q = server_reliability
        
        # Calculate minimum shards for reliability
        k_min = math.ceil(math.log(phi) / math.log(1 - q))
        
        # Consider download complexity: O(N^(1/k))
        # Balance between reliability and efficiency
        k_optimal = max(k_min, 3)  # At least 3 for basic redundancy
        k_optimal = min(k_optimal, self.config.max_shards)  # Cap at max
        
        logger.debug(f"Calculated optimal shards: {k_optimal} (min: {k_min})")
        return k_optimal
    
    def select_servers(
        self,
        num_shards: int,
        prefer_trusted: bool = True
    ) -> List[Server]:
        """
        Select optimal servers for PIR query.
        
        Args:
            num_shards: Number of shards needed
            prefer_trusted: Prefer Trusted Signatories over Light Nodes
            
        Returns:
            List of selected servers
        """
        available_servers = [s for s in self.servers.values() if s.is_available]
        
        if len(available_servers) < num_shards:
            raise ValueError(f"Not enough available servers: {len(available_servers)}/{num_shards}")
        
        # Separate by type
        trusted_servers = [s for s in available_servers 
                          if s.type == ServerType.TRUSTED_SIGNATORY]
        light_nodes = [s for s in available_servers 
                      if s.type == ServerType.LIGHT_NODE]
        full_nodes = [s for s in available_servers 
                     if s.type == ServerType.FULL_NODE]
        
        selected = []
        
        # First, select minimum trusted signatories
        if prefer_trusted:
            trusted_needed = min(self.config.min_trusted_signatories, len(trusted_servers))
            trusted_servers.sort(key=lambda s: s.effective_latency)
            selected.extend(trusted_servers[:trusted_needed])
        
        # Fill remaining with best available servers
        remaining_needed = num_shards - len(selected)
        remaining_servers = light_nodes + full_nodes + trusted_servers[len(selected):]
        remaining_servers.sort(key=lambda s: (s.effective_latency, -s.reliability))
        selected.extend(remaining_servers[:remaining_needed])
        
        return selected[:num_shards]
    
    def estimate_latency(self, servers: List[Server], num_rounds: int = 1) -> float:
        """
        Estimate total latency for PIR query.
        
        Latency model: RTT × shard-count × rounds
        
        Args:
            servers: Selected servers
            num_rounds: Number of PIR rounds (IT-PIR uses 1)
            
        Returns:
            Estimated latency in milliseconds
        """
        if not servers:
            return float('inf')
        
        # Use maximum latency (weakest link)
        max_latency = max(s.effective_latency for s in servers)
        
        # Total latency = max_latency × num_shards × rounds
        total_latency = max_latency * len(servers) * num_rounds
        
        return total_latency
    
    def calculate_failure_probability(
        self,
        servers: List[Server],
        threshold: int
    ) -> float:
        """
        Calculate probability of query failure.
        
        Failure occurs if more than (n - threshold) servers fail.
        
        Args:
            servers: Selected servers
            threshold: Minimum servers needed for reconstruction
            
        Returns:
            Probability of failure
        """
        n = len(servers)
        
        # If we need all servers, any failure is catastrophic
        if threshold == n:
            prob_all_succeed = np.prod([s.reliability for s in servers])
            return 1 - prob_all_succeed
        
        # Calculate probability using binomial
        # P(failure) = P(available_servers < threshold)
        reliabilities = [s.reliability for s in servers]
        
        # Simplified calculation (exact would need combinatorics)
        avg_reliability = np.mean(reliabilities)
        
        # Approximate using normal distribution for large n
        if n > 10:
            mean = n * avg_reliability
            std = np.sqrt(n * avg_reliability * (1 - avg_reliability))
            z_score = (threshold - mean) / std
            from scipy import stats
            return stats.norm.cdf(z_score)
        
        # For small n, use exact calculation
        failure_prob = 0.0
        for k in range(threshold):
            # Probability that exactly k servers are available
            # This is simplified; exact calculation would enumerate all combinations
            prob_k = (avg_reliability ** k) * ((1 - avg_reliability) ** (n - k))
            failure_prob += prob_k * math.comb(n, k)
        
        return failure_prob
    
    def get_configuration(
        self,
        database_size: int,
        use_preset: Optional[str] = None
    ) -> Tuple[ShardConfiguration, List[Server]]:
        """
        Get optimal configuration and server selection.
        
        Args:
            database_size: Size of database
            use_preset: Force specific preset ("3_shard" or "5_shard")
            
        Returns:
            Configuration and selected servers
        """
        if use_preset and use_preset in self.PRESET_CONFIGS:
            config = self.PRESET_CONFIGS[use_preset]
        else:
            # Calculate optimal configuration
            num_shards = self.calculate_optimal_shards(database_size)
            
            # Determine split between LN and TS
            ts_count = min(self.config.min_trusted_signatories, num_shards)
            ln_count = num_shards - ts_count
            
            # Select servers
            servers = self.select_servers(num_shards)
            
            # Estimate performance
            latency = self.estimate_latency(servers)
            failure_prob = self.calculate_failure_probability(servers, num_shards)
            
            config = ShardConfiguration(
                total_shards=num_shards,
                light_nodes=ln_count,
                trusted_signatories=ts_count,
                expected_latency_ms=latency,
                failure_probability=failure_prob,
                download_complexity=1/num_shards
            )
        
        # Select servers based on configuration
        servers = self.select_servers(config.total_shards)
        
        self.current_config = config
        logger.info(f"Selected configuration: {config.total_shards} shards, "
                   f"latency: {config.expected_latency_ms:.1f}ms, "
                   f"P(fail): {config.failure_probability:.2e}")
        
        return config, servers
    
    def should_reconfigure(self) -> bool:
        """
        Determine if topology should be reconfigured.
        
        Triggers reconfiguration if:
        - Latency exceeds SLA threshold
        - Too many server failures
        - Significant performance degradation
        
        Returns:
            True if reconfiguration needed
        """
        # Check if current config exists
        if not self.current_config:
            return True
        
        # Don't reconfigure too frequently
        if time.time() - self.last_reconfiguration < 60:
            return False
        
        # Check latency SLA
        recent_latencies = []
        for server_id, latencies in self.latency_history.items():
            if latencies:
                recent_latencies.extend(latencies[-10:])
        
        if recent_latencies:
            avg_latency = np.mean(recent_latencies)
            # Use adaptive_threshold to determine if reconfiguration is needed
            threshold = self.config.latency_sla_ms * self.config.adaptive_threshold
            if avg_latency > threshold:
                logger.warning(f"Latency {avg_latency:.1f}ms exceeds threshold {threshold:.1f}ms, reconfiguring")
                return True
        
        # Check server failures
        failed_servers = sum(1 for s in self.servers.values() 
                           if s.status == ServerStatus.FAILED)
        if failed_servers > len(self.servers) * 0.3:
            logger.warning(f"{failed_servers} servers failed, reconfiguring")
            return True
        
        return False
    
    def update_latency(self, server_id: str, latency_ms: float) -> None:
        """
        Update observed latency for a server.
        
        Args:
            server_id: Server identifier
            latency_ms: Observed latency in milliseconds
        """
        if server_id in self.servers:
            self.servers[server_id].latency_ms = (
                0.8 * self.servers[server_id].latency_ms + 
                0.2 * latency_ms  # Exponential moving average
            )
            self.latency_history[server_id].append(latency_ms)
            
            # Keep only recent history
            if len(self.latency_history[server_id]) > 100:
                self.latency_history[server_id] = self.latency_history[server_id][-100:]
    
    def handle_server_failure(self, server_id: str) -> Optional[Server]:
        """
        Handle server failure and find replacement.
        
        Args:
            server_id: Failed server identifier
            
        Returns:
            Replacement server if available
        """
        if server_id in self.servers:
            failed_server = self.servers[server_id]
            failed_server.status = ServerStatus.FAILED
            failed_server.failure_count += 1
            
            logger.warning(f"Server {server_id} failed (count: {failed_server.failure_count})")
            
            # Find replacement server of same type
            replacements = [
                s for s in self.servers.values()
                if s.is_available and s.type == failed_server.type and s.id != server_id
            ]
            
            if replacements:
                replacement = min(replacements, key=lambda s: s.effective_latency)
                logger.info(f"Replacing {server_id} with {replacement.id}")
                return replacement
            else:
                logger.error(f"No replacement available for {server_id}")
                return None
        
        return None
    
    def health_check(self) -> Dict[str, ServerStatus]:
        """
        Perform health check on all servers.
        
        Returns:
            Dictionary of server statuses
        """
        statuses = {}
        current_time = time.time()
        
        for server_id, server in self.servers.items():
            # Reset failed servers after cooldown FIRST
            if server.status == ServerStatus.FAILED:
                if current_time - server.last_ping > 300:  # 5 minute cooldown
                    server.status = ServerStatus.AVAILABLE
                    server.failure_count = max(0, server.failure_count - 1)
                    logger.info(f"Server {server_id} recovered")
            # Then check if server hasn't been pinged recently
            elif current_time - server.last_ping > self.config.server_timeout_ms / 1000:
                server.status = ServerStatus.UNKNOWN
            
            statuses[server_id] = server.status
        
        return statuses
    
    def get_stats(self) -> Dict:
        """
        Get topology statistics.
        
        Returns:
            Dictionary of statistics
        """
        available = sum(1 for s in self.servers.values() if s.is_available)
        failed = sum(1 for s in self.servers.values() 
                    if s.status == ServerStatus.FAILED)
        
        avg_latencies = {}
        for server_id, latencies in self.latency_history.items():
            if latencies:
                avg_latencies[server_id] = np.mean(latencies[-20:])
        
        return {
            "total_servers": len(self.servers),
            "available_servers": available,
            "failed_servers": failed,
            "current_config": self.current_config._asdict() if self.current_config else None,
            "query_count": self.query_count,
            "average_latencies": avg_latencies,
            "should_reconfigure": self.should_reconfigure()
        }


class AdaptiveTopologyManager(TopologyManager):
    """
    Extended topology manager with adaptive selection strategies.
    
    Monitors performance and automatically switches between configurations
    based on observed latencies and failure patterns.
    """
    
    def __init__(self, config: Optional[TopologyConfig] = None):
        """Initialize adaptive topology manager."""
        super().__init__(config)
        self.performance_history: List[Dict] = []
        self.config_performance: Dict[str, List[float]] = defaultdict(list)
        self.current_preset = "3_shard"  # Start with low-latency config
    
    def adaptive_select(
        self,
        database_size: int
    ) -> Tuple[ShardConfiguration, List[Server]]:
        """
        Adaptively select configuration based on observed performance.
        
        Args:
            database_size: Size of database
            
        Returns:
            Configuration and selected servers
        """
        # Check if we should try a different configuration
        if self.should_switch_config():
            self.current_preset = self.select_best_config()
            logger.info(f"Switching to {self.current_preset} configuration")
        
        # Get configuration with current preset
        config, servers = self.get_configuration(database_size, self.current_preset)
        
        # Track query
        self.query_count += 1
        
        return config, servers
    
    def should_switch_config(self) -> bool:
        """
        Determine if configuration should be switched.
        
        Returns:
            True if should switch configuration
        """
        if self.query_count < 10:
            return False  # Not enough data
        
        # Check performance of current config
        if self.current_preset in self.config_performance:
            recent_perf = self.config_performance[self.current_preset][-10:]
            if recent_perf:
                avg_latency = np.mean(recent_perf)
                
                # Switch if consistently exceeding SLA
                if avg_latency > self.config.latency_sla_ms:
                    return True
                
                # Switch if high variance (unstable)
                if np.std(recent_perf) > avg_latency * 0.5:
                    return True
        
        # Periodically explore other configs
        if self.query_count % 100 == 0:
            return True
        
        return False
    
    def select_best_config(self) -> str:
        """
        Select best configuration based on historical performance.
        
        Returns:
            Configuration name
        """
        if not self.config_performance:
            # No history, alternate between presets
            return "5_shard" if self.current_preset == "3_shard" else "3_shard"
        
        # Calculate average performance for each config
        config_scores = {}
        for config_name in self.PRESET_CONFIGS:
            if config_name in self.config_performance:
                latencies = self.config_performance[config_name][-20:]
                if latencies:
                    # Score based on latency and stability
                    avg_latency = np.mean(latencies)
                    std_latency = np.std(latencies)
                    score = avg_latency + std_latency * 0.5  # Penalize instability
                    config_scores[config_name] = score
        
        if config_scores:
            # Return config with best (lowest) score
            return min(config_scores, key=config_scores.get)
        
        # Default to 3-shard for lower latency
        return "3_shard"
    
    def record_performance(
        self,
        config_name: str,
        latency_ms: float,
        success: bool
    ) -> None:
        """
        Record performance metrics for a configuration.
        
        Args:
            config_name: Configuration used
            latency_ms: Observed latency
            success: Whether query succeeded
        """
        self.config_performance[config_name].append(latency_ms)
        
        # Keep bounded history
        if len(self.config_performance[config_name]) > 100:
            self.config_performance[config_name] = self.config_performance[config_name][-100:]
        
        # Record in performance history
        self.performance_history.append({
            "timestamp": time.time(),
            "config": config_name,
            "latency_ms": latency_ms,
            "success": success,
            "query_count": self.query_count
        })
        
        # Keep bounded history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]