"""
PIR Coordinator for server discovery, health monitoring, and compliance.
Manages geographic diversity and bandwidth optimization.
"""
import logging
from typing import Dict, List, Optional, Any, Union

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
from geopy.distance import geodesic

from genomevault.utils.logging import audit_logger, get_logger, logger

logger = get_logger(__name__)


class ServerType(Enum):
    """PIR server types."""

    TRUSTED_SIGNATORY = "TS"  # HIPAA compliant, 0.98 honesty probability
    LIGHT_NODE = "LN"  # Generic node, 0.95 honesty probability


@dataclass
class ServerInfo:
    """PIR server information."""

    server_id: str
    server_type: ServerType
    endpoint: str
    location: Tuple[float, float]  # (latitude, longitude)
    region: str  # Geographic region
    capabilities: Set[str] = field(default_factory=set)
    health_score: float = 1.0  # 0-1 health score
    last_health_check: float = 0
    response_time_ms: float = 0
    success_rate: float = 1.0


@dataclass
class ServerSelectionCriteria:
    """Criteria for server selection."""

    min_servers: int = 2
    max_servers: int = 5
    require_geographic_diversity: bool = True
    min_distance_km: float = 1000
    prefer_trusted_signatories: bool = True
    max_latency_ms: float = 500
    min_health_score: float = 0.8


class PIRCoordinator:
    """
    Coordinates PIR server selection and health monitoring.

    Responsibilities:
    - Server discovery and registration
    - Health monitoring and failover
    - Geographic diversity enforcement
    - Bandwidth optimization
    - Regulatory compliance management
    """

    def __init__(self) -> None:
            """TODO: Add docstring for __init__"""
    self.servers: Dict[str, ServerInfo] = {}
        self.active_queries: Dict[str, List[str]] = {}  # query_id -> server_ids

        # Configuration
        self.health_check_interval = 30  # seconds
        self.min_healthy_servers = 5

        # Compliance regions
        self.compliance_regions = {
            "US": {"HIPAA", "CCPA"},
            "EU": {"GDPR"},
            "CA": {"PIPEDA"},
            "UK": {"DPA"},
        }

        # Start background tasks
        self._health_check_task = None

    async def start(self) -> None:
           """TODO: Add docstring for start"""
     """Start coordinator background tasks."""
        self._health_check_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("PIR Coordinator started")

    async def stop(self) -> None:
           """TODO: Add docstring for stop"""
     """Stop coordinator."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

    def register_server(self, server_info: ServerInfo) -> None:
           """TODO: Add docstring for register_server"""
     """
        Register a new PIR server.

        Args:
            server_info: Server information
        """
        self.servers[server_info.server_id] = server_info
        logger.info(f"Registered server {server_info.server_id} ({server_info.server_type.value})")

        # Audit log
        audit_logger.log_event(
            event_type="server_registered",
            actor="coordinator",
            action="register",
            resource=server_info.server_id,
            metadata={
                "server_type": server_info.server_type.value,
                "location": server_info.region,
                "endpoint": server_info.endpoint,
            },
        )

    async def select_servers(
        self, criteria: ServerSelectionCriteria, user_region: Optional[str] = None
    ) -> List[ServerInfo]:
           """TODO: Add docstring for select_servers"""
     """
        Select optimal servers based on criteria.

        Args:
            criteria: Selection criteria
            user_region: User's geographic region for compliance

        Returns:
            List of selected servers
        """
        # Filter healthy servers
        healthy_servers = [
            s
            for s in self.servers.values()
            if s.health_score >= criteria.min_health_score
            and s.response_time_ms <= criteria.max_latency_ms
        ]

        if len(healthy_servers) < criteria.min_servers:
            raise ValueError(
                f"Insufficient healthy servers: {len(healthy_servers)} < {criteria.min_servers}"
            )

        # Apply compliance filtering if needed
        if user_region:
            healthy_servers = self._filter_compliant_servers(healthy_servers, user_region)

        # Sort by preference
        scored_servers = []
        for server in healthy_servers:
            score = self._calculate_server_score(server, criteria)
            scored_servers.append((score, server))

        scored_servers.sort(reverse=True, key=lambda x: x[0])

        # Select servers with geographic diversity
        selected = []
        for score, server in scored_servers:
            if len(selected) >= criteria.max_servers:
                break

            # Check geographic diversity
            if criteria.require_geographic_diversity and selected:
                if not self._check_geographic_diversity(server, selected, criteria.min_distance_km):
                    continue

            selected.append(server)

            if len(selected) >= criteria.min_servers:
                # Check if we have minimum diversity
                if not criteria.require_geographic_diversity or self._has_sufficient_diversity(
                    selected
                ):
                    break

        if len(selected) < criteria.min_servers:
            raise ValueError(
                f"Could not select {criteria.min_servers} servers with required diversity"
            )

        logger.info(f"Selected {len(selected)} servers: {[s.server_id for s in selected]}")
        return selected

    def _calculate_server_score(
        self, server: ServerInfo, criteria: ServerSelectionCriteria
    ) -> float:
           """TODO: Add docstring for _calculate_server_score"""
     """Calculate server selection score."""
        score = 0.0

        # Health score (0-40 points)
        score += server.health_score * 40

        # Response time (0-30 points)
        latency_score = max(0, 1 - (server.response_time_ms / criteria.max_latency_ms))
        score += latency_score * 30

        # Success rate (0-20 points)
        score += server.success_rate * 20

        # Server type preference (0-10 points)
        if (
            criteria.prefer_trusted_signatories
            and server.server_type == ServerType.TRUSTED_SIGNATORY
        ):
            score += 10

        return score

    def _check_geographic_diversity(
        self, candidate: ServerInfo, selected: List[ServerInfo], min_distance_km: float
    ) -> bool:
           """TODO: Add docstring for _check_geographic_diversity"""
     """Check if candidate provides geographic diversity."""
        for server in selected:
            distance = geodesic(candidate.location, server.location).kilometers
            if distance < min_distance_km:
                return False
        return True

    def _has_sufficient_diversity(self, servers: List[ServerInfo]) -> bool:
           """TODO: Add docstring for _has_sufficient_diversity"""
     """Check if selected servers have sufficient diversity."""
        regions = set(s.region for s in servers)
        return len(regions) >= 2

    def _filter_compliant_servers(
        self, servers: List[ServerInfo], user_region: str
    ) -> List[ServerInfo]:
           """TODO: Add docstring for _filter_compliant_servers"""
     """Filter servers based on compliance requirements."""
        required_compliance = self.compliance_regions.get(user_region, set())

        compliant_servers = []
        for server in servers:
            # Check if server meets compliance requirements
            if server.server_type == ServerType.TRUSTED_SIGNATORY:
                # TS servers are HIPAA compliant
                if "HIPAA" in required_compliance:
                    compliant_servers.append(server)
            elif not required_compliance:
                # No specific compliance needed
                compliant_servers.append(server)

        return compliant_servers

    async def _health_monitor_loop(self) -> None:
           """TODO: Add docstring for _health_monitor_loop"""
     """Background task for health monitoring."""
        while True:
            try:
                await self._check_all_servers_health()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)

    async def _check_all_servers_health(self) -> None:
           """TODO: Add docstring for _check_all_servers_health"""
     """Check health of all registered servers."""
        tasks = []
        for server in self.servers.values():
            tasks.append(self._check_server_health(server))

        await asyncio.gather(*tasks, return_exceptions=True)

        # Check if we have minimum healthy servers
        healthy_count = sum(1 for s in self.servers.values() if s.health_score > 0.5)
        if healthy_count < self.min_healthy_servers:
            logger.warning(
                f"Low healthy server count: {healthy_count} < {self.min_healthy_servers}"
            )

    async def _check_server_health(self, server: ServerInfo) -> None:
           """TODO: Add docstring for _check_server_health"""
     """Check health of individual server."""
        try:
            start_time = time.time()

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{server.endpoint}/health", timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        health_data = await response.json()

                        # Update server info
                        server.response_time_ms = (time.time() - start_time) * 1000
                        server.last_health_check = time.time()

                        # Calculate health score
                        if health_data.get("status") == "healthy":
                            server.health_score = 1.0
                        else:
                            server.health_score = 0.5

                        # Update success rate (simple moving average)
                        server.success_rate = 0.95 * server.success_rate + 0.05 * 1.0

                    else:
                        server.health_score *= 0.8  # Degrade score
                        server.success_rate = 0.95 * server.success_rate + 0.05 * 0.0

        except Exception as e:
            logger.warning(f"Health check failed for {server.server_id}: {e}")
            server.health_score *= 0.7
            server.success_rate = 0.95 * server.success_rate + 0.05 * 0.0

    async def execute_query(
        self, query_id: str, query_vectors: List[Dict], selected_servers: List[ServerInfo]
    ) -> List[Dict]:
           """TODO: Add docstring for execute_query"""
     """
        Execute PIR query across selected servers.

        Args:
            query_id: Unique query identifier
            query_vectors: Query vectors for each server
            selected_servers: Selected servers

        Returns:
            List of server responses
        """
        if len(query_vectors) != len(selected_servers):
            raise ValueError("Number of query vectors must match number of servers")

        # Track active query
        self.active_queries[query_id] = [s.server_id for s in selected_servers]

        try:
            # Execute queries in parallel
            tasks = []
            for server, query_vector in zip(selected_servers, query_vectors):
                task = self._execute_server_query(server, query_id, query_vector)
                tasks.append(task)

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for failures
            valid_responses = []
            failed_servers = []

            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    failed_servers.append(selected_servers[i].server_id)
                    logger.error(f"Query failed on {selected_servers[i].server_id}: {response}")
                else:
                    valid_responses.append(response)

            if failed_servers:
                # Handle failover
                logger.warning(f"Query {query_id} failed on servers: {failed_servers}")
                # Could implement automatic failover here

            return valid_responses

        finally:
            # Clean up tracking
            del self.active_queries[query_id]

    async def _execute_server_query(
        self, server: ServerInfo, query_id: str, query_data: Dict
    ) -> Dict:
           """TODO: Add docstring for _execute_server_query"""
     """Execute query on single server."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server.endpoint}/pir/query",
                json={"query_id": query_id, **query_data},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Server returned {response.status}: {error_text}")

    def get_coordinator_stats(self) -> Dict[str, Any]:
           """TODO: Add docstring for get_coordinator_stats"""
     """Get coordinator statistics."""
        healthy_servers = sum(1 for s in self.servers.values() if s.health_score > 0.8)
        degraded_servers = sum(1 for s in self.servers.values() if 0.5 < s.health_score <= 0.8)
        unhealthy_servers = sum(1 for s in self.servers.values() if s.health_score <= 0.5)

        ts_servers = sum(
            1 for s in self.servers.values() if s.server_type == ServerType.TRUSTED_SIGNATORY
        )
        ln_servers = sum(1 for s in self.servers.values() if s.server_type == ServerType.LIGHT_NODE)

        avg_latency = (
            sum(s.response_time_ms for s in self.servers.values()) / len(self.servers)
            if self.servers
            else 0
        )

        return {
            "total_servers": len(self.servers),
            "healthy_servers": healthy_servers,
            "degraded_servers": degraded_servers,
            "unhealthy_servers": unhealthy_servers,
            "trusted_signatories": ts_servers,
            "light_nodes": ln_servers,
            "active_queries": len(self.active_queries),
            "average_latency_ms": avg_latency,
            "geographic_regions": len(set(s.region for s in self.servers.values())),
        }


# Example usage
if __name__ == "__main__":

    async def demo() -> None:
            """TODO: Add docstring for demo"""
    # Create coordinator
        coordinator = PIRCoordinator()
        await coordinator.start()

        # Register servers
        servers = [
            ServerInfo(
                server_id="ts-us-east-1",
                server_type=ServerType.TRUSTED_SIGNATORY,
                endpoint="https://pir-ts-east.genomevault.org",
                location=(40.7128, -74.0060),  # New York
                region="US-EAST",
                capabilities={"batch_query", "gpu_acceleration"},
            ),
            ServerInfo(
                server_id="ts-eu-west-1",
                server_type=ServerType.TRUSTED_SIGNATORY,
                endpoint="https://pir-ts-eu.genomevault.org",
                location=(51.5074, -0.1278),  # London
                region="EU-WEST",
                capabilities={"batch_query"},
            ),
            ServerInfo(
                server_id="ln-asia-1",
                server_type=ServerType.LIGHT_NODE,
                endpoint="https://pir-ln-asia.genomevault.org",
                location=(35.6762, 139.6503),  # Tokyo
                region="ASIA-PACIFIC",
                capabilities={"batch_query"},
            ),
            ServerInfo(
                server_id="ln-us-west-1",
                server_type=ServerType.LIGHT_NODE,
                endpoint="https://pir-ln-west.genomevault.org",
                location=(37.7749, -122.4194),  # San Francisco
                region="US-WEST",
                capabilities={"batch_query", "compression"},
            ),
            ServerInfo(
                server_id="ts-us-central-1",
                server_type=ServerType.TRUSTED_SIGNATORY,
                endpoint="https://pir-ts-central.genomevault.org",
                location=(41.8781, -87.6298),  # Chicago
                region="US-CENTRAL",
                capabilities={"batch_query", "gpu_acceleration"},
            ),
        ]

        for server in servers:
            coordinator.register_server(server)

        # Select servers with criteria
        criteria = ServerSelectionCriteria(
            min_servers=3,
            max_servers=5,
            require_geographic_diversity=True,
            min_distance_km=1000,
            prefer_trusted_signatories=True,
        )

        selected = await coordinator.select_servers(criteria, user_region="US")

        print("Selected servers:")
        for server in selected:
            print(f"  - {server.server_id} ({server.server_type.value}) in {server.region}")

        # Show statistics
        stats = coordinator.get_coordinator_stats()
        print("\nCoordinator statistics:")
        print(json.dumps(stats, indent=2))

        # Cleanup
        await coordinator.stop()

    # Run demo
    asyncio.run(demo())
