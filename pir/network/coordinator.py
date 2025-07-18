"""
PIR network coordinator for managing distributed servers.
Handles server discovery, health monitoring, and query routing.
"""
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import time
import json
import numpy as np
from collections import defaultdict
import heapq

from ..client import PIRServer, PIRClient
from ..server.pir_server import PIRServer as PIRServerInstance
from utils.config import config
from utils.logging import logger, performance_logger


@dataclass
class ServerHealth:
    """Server health metrics."""
    server_id: str
    is_healthy: bool
    last_check: float
    latency_ms: float
    success_rate: float
    query_count: int
    error_count: int
    
    @property
    def reliability_score(self) -> float:
        """Calculate server reliability score."""
        if self.query_count == 0:
            return 0.5  # Default for new servers
        
        # Combine success rate and latency
        latency_factor = 1.0 / (1.0 + self.latency_ms / 100)  # Lower latency is better
        return self.success_rate * latency_factor


@dataclass
class NetworkTopology:
    """PIR network topology information."""
    servers: Dict[str, PIRServer] = field(default_factory=dict)
    server_health: Dict[str, ServerHealth] = field(default_factory=dict)
    server_regions: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    ts_servers: Set[str] = field(default_factory=set)
    ln_servers: Set[str] = field(default_factory=set)
    
    def add_server(self, server: PIRServer):
        """Add server to topology."""
        self.servers[server.server_id] = server
        self.server_regions[server.region].add(server.server_id)
        
        if server.is_trusted_signatory:
            self.ts_servers.add(server.server_id)
        else:
            self.ln_servers.add(server.server_id)
    
    def remove_server(self, server_id: str):
        """Remove server from topology."""
        if server_id in self.servers:
            server = self.servers[server_id]
            self.server_regions[server.region].discard(server_id)
            self.ts_servers.discard(server_id)
            self.ln_servers.discard(server_id)
            del self.servers[server_id]
            
            if server_id in self.server_health:
                del self.server_health[server_id]


class PIRNetworkCoordinator:
    """
    Coordinates PIR network operations.
    Manages server discovery, health monitoring, and optimal routing.
    """
    
    def __init__(self):
        """Initialize network coordinator."""
        self.topology = NetworkTopology()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Health check configuration
        self.health_check_interval = config.pir.health_check_interval_seconds
        self.health_check_timeout = config.pir.health_check_timeout_seconds
        
        # Background tasks
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.topology_update_task: Optional[asyncio.Task] = None
        
        # Query routing cache
        self.routing_cache: Dict[str, List[str]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[str, float] = {}
        
        logger.info("PIR Network Coordinator initialized")
    
    async def start(self):
        """Start network coordinator services."""
        self.session = aiohttp.ClientSession()
        
        # Discover initial servers
        await self.discover_servers()
        
        # Start background tasks
        self.health_monitor_task = asyncio.create_task(self.monitor_health())
        self.topology_update_task = asyncio.create_task(self.update_topology())
        
        logger.info(f"Network coordinator started with {len(self.topology.servers)} servers")
    
    async def stop(self):
        """Stop network coordinator services."""
        # Cancel background tasks
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            
        if self.topology_update_task:
            self.topology_update_task.cancel()
        
        # Close session
        if self.session:
            await self.session.close()
        
        logger.info("Network coordinator stopped")
    
    async def discover_servers(self):
        """Discover available PIR servers."""
        # In production, would use service discovery (e.g., Consul, etcd)
        # For now, use configuration
        
        server_configs = [
            # Light nodes
            {"id": "ln1", "endpoint": "http://ln1.genomevault.com", 
             "region": "us-east", "is_ts": False, "honesty": 0.95},
            {"id": "ln2", "endpoint": "http://ln2.genomevault.com", 
             "region": "eu-west", "is_ts": False, "honesty": 0.95},
            {"id": "ln3", "endpoint": "http://ln3.genomevault.com", 
             "region": "asia-pac", "is_ts": False, "honesty": 0.95},
            # Trusted signatories
            {"id": "ts1", "endpoint": "http://ts1.genomevault.com", 
             "region": "us-west", "is_ts": True, "honesty": 0.98},
            {"id": "ts2", "endpoint": "http://ts2.genomevault.com", 
             "region": "us-central", "is_ts": True, "honesty": 0.98},
        ]
        
        for config in server_configs:
            server = PIRServer(
                server_id=config["id"],
                endpoint=config["endpoint"],
                region=config["region"],
                is_trusted_signatory=config["is_ts"],
                honesty_probability=config["honesty"],
                latency_ms=0  # Will be measured
            )
            
            self.topology.add_server(server)
            
            # Initialize health metrics
            self.topology.server_health[server.server_id] = ServerHealth(
                server_id=server.server_id,
                is_healthy=True,
                last_check=time.time(),
                latency_ms=0,
                success_rate=1.0,
                query_count=0,
                error_count=0
            )
        
        logger.info(f"Discovered {len(self.topology.servers)} servers")
    
    async def monitor_health(self):
        """Monitor server health in background."""
        while True:
            try:
                # Check health of all servers
                tasks = []
                for server_id, server in self.topology.servers.items():
                    task = self.check_server_health(server)
                    tasks.append(task)
                
                # Wait for all health checks
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log summary
                healthy_count = sum(
                    1 for h in self.topology.server_health.values() 
                    if h.is_healthy
                )
                logger.info(f"Health check complete: {healthy_count}/{len(self.topology.servers)} healthy")
                
                # Wait before next check
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(10)  # Wait before retry
    
    async def check_server_health(self, server: PIRServer) -> bool:
        """
        Check health of a single server.
        
        Args:
            server: Server to check
            
        Returns:
            True if healthy
        """
        try:
            start_time = time.time()
            
            # Send health check request
            async with self.session.get(
                f"{server.endpoint}/health",
                timeout=aiohttp.ClientTimeout(total=self.health_check_timeout)
            ) as response:
                if response.status == 200:
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Update health metrics
                    health = self.topology.server_health[server.server_id]
                    health.is_healthy = True
                    health.last_check = time.time()
                    health.latency_ms = latency_ms
                    
                    # Update server latency
                    server.latency_ms = latency_ms
                    
                    return True
                else:
                    # Mark as unhealthy
                    health = self.topology.server_health[server.server_id]
                    health.is_healthy = False
                    health.last_check = time.time()
                    health.error_count += 1
                    
                    logger.warning(f"Server {server.server_id} returned status {response.status}")
                    return False
                    
        except Exception as e:
            # Mark as unhealthy
            health = self.topology.server_health[server.server_id]
            health.is_healthy = False
            health.last_check = time.time()
            health.error_count += 1
            
            logger.error(f"Health check failed for {server.server_id}: {e}")
            return False
    
    async def update_topology(self):
        """Update network topology periodically."""
        while True:
            try:
                # Clear expired routing cache
                current_time = time.time()
                expired_keys = [
                    key for key, timestamp in self.cache_timestamps.items()
                    if current_time - timestamp > self.cache_ttl
                ]
                
                for key in expired_keys:
                    del self.routing_cache[key]
                    del self.cache_timestamps[key]
                
                # In production, would re-discover servers
                # For now, just log status
                logger.debug(f"Topology: {len(self.topology.servers)} servers, "
                           f"{len(self.routing_cache)} cached routes")
                
                # Wait before next update
                await asyncio.sleep(60)  # Update every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error updating topology: {e}")
                await asyncio.sleep(10)
    
    def select_optimal_servers(self, num_servers: int,
                             require_ts: int = 0,
                             exclude_servers: Optional[Set[str]] = None) -> List[PIRServer]:
        """
        Select optimal servers for a PIR query.
        
        Args:
            num_servers: Total number of servers needed
            require_ts: Minimum number of TS servers required
            exclude_servers: Servers to exclude from selection
            
        Returns:
            List of selected servers
        """
        if exclude_servers is None:
            exclude_servers = set()
        
        # Get healthy servers
        healthy_servers = [
            server for server_id, server in self.topology.servers.items()
            if (self.topology.server_health[server_id].is_healthy and 
                server_id not in exclude_servers)
        ]
        
        # Separate TS and LN servers
        ts_servers = [s for s in healthy_servers if s.is_trusted_signatory]
        ln_servers = [s for s in healthy_servers if not s.is_trusted_signatory]
        
        # Check if we have enough TS servers
        if len(ts_servers) < require_ts:
            logger.warning(f"Insufficient TS servers: {len(ts_servers)} < {require_ts}")
            return []
        
        # Sort servers by reliability score
        def server_score(server):
            health = self.topology.server_health[server.server_id]
            return -health.reliability_score  # Negative for max heap
        
        ts_servers.sort(key=server_score)
        ln_servers.sort(key=server_score)
        
        # Select servers
        selected = []
        
        # Add required TS servers
        selected.extend(ts_servers[:require_ts])
        
        # Add remaining servers (prefer LN for cost efficiency)
        remaining_needed = num_servers - len(selected)
        
        # Add LN servers first
        ln_to_add = min(remaining_needed, len(ln_servers))
        selected.extend(ln_servers[:ln_to_add])
        
        # If still need more, add additional TS servers
        if len(selected) < num_servers:
            additional_ts = min(num_servers - len(selected), 
                              len(ts_servers) - require_ts)
            selected.extend(ts_servers[require_ts:require_ts + additional_ts])
        
        return selected[:num_servers]
    
    def get_server_configuration(self, 
                               target_failure_prob: float = 1e-4,
                               max_latency_ms: Optional[float] = None) -> Dict[str, Any]:
        """
        Get optimal server configuration for target privacy level.
        
        Args:
            target_failure_prob: Target failure probability
            max_latency_ms: Maximum acceptable latency
            
        Returns:
            Optimal configuration
        """
        configurations = []
        
        # Configuration 1: 3 LN + 2 TS (5 servers total)
        servers_3ln_2ts = self.select_optimal_servers(5, require_ts=2)
        if len(servers_3ln_2ts) == 5:
            latency = sum(s.latency_ms for s in servers_3ln_2ts)
            failure_prob = (1 - 0.98) ** 2  # 2 TS servers
            
            configurations.append({
                'name': '3 LN + 2 TS',
                'servers': [s.server_id for s in servers_3ln_2ts],
                'total_servers': 5,
                'ts_count': 2,
                'latency_ms': latency,
                'failure_probability': failure_prob
            })
        
        # Configuration 2: 1 LN + 2 TS (3 servers total)
        servers_1ln_2ts = self.select_optimal_servers(3, require_ts=2)
        if len(servers_1ln_2ts) == 3:
            latency = sum(s.latency_ms for s in servers_1ln_2ts)
            failure_prob = (1 - 0.98) ** 2  # 2 TS servers
            
            configurations.append({
                'name': '1 LN + 2 TS',
                'servers': [s.server_id for s in servers_1ln_2ts],
                'total_servers': 3,
                'ts_count': 2,
                'latency_ms': latency,
                'failure_probability': failure_prob
            })
        
        # Configuration 3: 3 TS (3 servers total)
        servers_3ts = self.select_optimal_servers(3, require_ts=3)
        if len(servers_3ts) == 3:
            latency = sum(s.latency_ms for s in servers_3ts)
            failure_prob = (1 - 0.98) ** 3  # 3 TS servers
            
            configurations.append({
                'name': '3 TS',
                'servers': [s.server_id for s in servers_3ts],
                'total_servers': 3,
                'ts_count': 3,
                'latency_ms': latency,
                'failure_probability': failure_prob
            })
        
        # Filter by requirements
        valid_configs = [
            c for c in configurations
            if c['failure_probability'] <= target_failure_prob
        ]
        
        if max_latency_ms:
            valid_configs = [
                c for c in valid_configs
                if c['latency_ms'] <= max_latency_ms
            ]
        
        # Select optimal (minimize latency)
        if valid_configs:
            optimal = min(valid_configs, key=lambda c: c['latency_ms'])
        else:
            optimal = None
        
        return {
            'configurations': configurations,
            'valid_configurations': valid_configs,
            'optimal': optimal,
            'network_status': {
                'total_servers': len(self.topology.servers),
                'healthy_servers': sum(1 for h in self.topology.server_health.values() if h.is_healthy),
                'ts_servers': len(self.topology.ts_servers),
                'ln_servers': len(self.topology.ln_servers)
            }
        }
    
    async def create_pir_client(self, database_size: int,
                              config_name: Optional[str] = None) -> PIRClient:
        """
        Create PIR client with optimal server configuration.
        
        Args:
            database_size: Size of database to query
            config_name: Specific configuration to use
            
        Returns:
            Configured PIR client
        """
        # Get optimal configuration
        config_info = self.get_server_configuration()
        
        if config_name:
            # Find specific configuration
            config = next(
                (c for c in config_info['configurations'] if c['name'] == config_name),
                None
            )
            if not config:
                raise ValueError(f"Configuration '{config_name}' not found")
        else:
            # Use optimal configuration
            config = config_info['optimal']
            if not config:
                raise RuntimeError("No valid configuration available")
        
        # Get server objects
        servers = [
            self.topology.servers[server_id]
            for server_id in config['servers']
        ]
        
        # Create client
        client = PIRClient(servers, database_size)
        
        logger.info(f"Created PIR client with configuration '{config['name']}'")
        
        return client
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """
        Get network statistics.
        
        Returns:
            Network statistics
        """
        # Calculate aggregate statistics
        total_queries = sum(h.query_count for h in self.topology.server_health.values())
        total_errors = sum(h.error_count for h in self.topology.server_health.values())
        
        avg_latency = np.mean([
            h.latency_ms for h in self.topology.server_health.values()
            if h.latency_ms > 0
        ]) if self.topology.server_health else 0
        
        # Per-region statistics
        region_stats = {}
        for region, server_ids in self.topology.server_regions.items():
            region_servers = [
                self.topology.server_health[sid]
                for sid in server_ids
                if sid in self.topology.server_health
            ]
            
            if region_servers:
                region_stats[region] = {
                    'server_count': len(server_ids),
                    'healthy_count': sum(1 for s in region_servers if s.is_healthy),
                    'avg_latency_ms': np.mean([s.latency_ms for s in region_servers if s.latency_ms > 0]),
                    'total_queries': sum(s.query_count for s in region_servers)
                }
        
        return {
            'total_servers': len(self.topology.servers),
            'healthy_servers': sum(1 for h in self.topology.server_health.values() if h.is_healthy),
            'ts_servers': {
                'total': len(self.topology.ts_servers),
                'healthy': sum(
                    1 for sid in self.topology.ts_servers
                    if self.topology.server_health.get(sid, ServerHealth(sid, False, 0, 0, 0, 0, 0)).is_healthy
                )
            },
            'ln_servers': {
                'total': len(self.topology.ln_servers),
                'healthy': sum(
                    1 for sid in self.topology.ln_servers
                    if self.topology.server_health.get(sid, ServerHealth(sid, False, 0, 0, 0, 0, 0)).is_healthy
                )
            },
            'queries': {
                'total': total_queries,
                'errors': total_errors,
                'success_rate': (total_queries - total_errors) / total_queries if total_queries > 0 else 1.0
            },
            'performance': {
                'avg_latency_ms': avg_latency,
                'cache_size': len(self.routing_cache)
            },
            'regions': region_stats
        }
    
    async def handle_server_failure(self, server_id: str):
        """
        Handle server failure.
        
        Args:
            server_id: Failed server ID
        """
        logger.warning(f"Handling failure of server {server_id}")
        
        # Mark as unhealthy
        if server_id in self.topology.server_health:
            self.topology.server_health[server_id].is_healthy = False
            self.topology.server_health[server_id].error_count += 1
        
        # Clear routing cache entries involving this server
        keys_to_remove = [
            key for key, servers in self.routing_cache.items()
            if server_id in servers
        ]
        
        for key in keys_to_remove:
            del self.routing_cache[key]
            del self.cache_timestamps[key]
        
        # Log impact
        logger.info(f"Removed {len(keys_to_remove)} cached routes involving {server_id}")


# Example usage
if __name__ == "__main__":
    async def example():
        # Create coordinator
        coordinator = PIRNetworkCoordinator()
        
        # Start services
        await coordinator.start()
        
        # Get network statistics
        stats = coordinator.get_network_statistics()
        print("Network Statistics:")
        print(json.dumps(stats, indent=2))
        
        # Get optimal configuration
        config = coordinator.get_server_configuration(target_failure_prob=1e-4)
        print("\nOptimal Configuration:")
        print(json.dumps(config['optimal'], indent=2))
        
        # Create PIR client
        client = await coordinator.create_pir_client(database_size=1000000)
        print(f"\nCreated client with {len(client.servers)} servers")
        
        # Cleanup
        await coordinator.stop()
        await client.close()
    
    # Run example
    # asyncio.run(example())
