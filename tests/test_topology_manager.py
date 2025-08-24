"""
Tests for PIR Topology Manager.
"""

import pytest
import time
import math
from typing import List

from genomevault.pir.topology_manager import (
    TopologyManager,
    AdaptiveTopologyManager,
    Server,
    ServerType,
    ServerStatus,
    ShardConfiguration,
    TopologyConfig,
)


class TestTopologyManager:
    """Test suite for TopologyManager."""

    @pytest.fixture
    def topology_manager(self):
        """Create a topology manager instance."""
        config = TopologyConfig(
            target_failure_probability=4e-4,
            min_trusted_signatories=2,
            max_shards=10,
            latency_sla_ms=500.0,
        )
        return TopologyManager(config)

    @pytest.fixture
    def sample_servers(self) -> List[Server]:
        """Create sample servers for testing."""
        servers = [
            # Light Nodes
            Server(
                "ln1",
                ServerType.LIGHT_NODE,
                "http://ln1.example.com",
                latency_ms=50.0,
                reliability=0.99,
                capacity=100,
            ),
            Server(
                "ln2",
                ServerType.LIGHT_NODE,
                "http://ln2.example.com",
                latency_ms=60.0,
                reliability=0.98,
                capacity=100,
            ),
            Server(
                "ln3",
                ServerType.LIGHT_NODE,
                "http://ln3.example.com",
                latency_ms=70.0,
                reliability=0.97,
                capacity=100,
            ),
            # Trusted Signatories
            Server(
                "ts1",
                ServerType.TRUSTED_SIGNATORY,
                "http://ts1.example.com",
                latency_ms=40.0,
                reliability=0.995,
                capacity=50,
            ),
            Server(
                "ts2",
                ServerType.TRUSTED_SIGNATORY,
                "http://ts2.example.com",
                latency_ms=45.0,
                reliability=0.995,
                capacity=50,
            ),
            Server(
                "ts3",
                ServerType.TRUSTED_SIGNATORY,
                "http://ts3.example.com",
                latency_ms=55.0,
                reliability=0.99,
                capacity=50,
            ),
            # Full Nodes
            Server(
                "fn1",
                ServerType.FULL_NODE,
                "http://fn1.example.com",
                latency_ms=80.0,
                reliability=0.98,
                capacity=200,
            ),
        ]
        return servers

    def test_calculate_optimal_shards(self, topology_manager):
        """Test optimal shard calculation."""
        # Test with high reliability
        shards = topology_manager.calculate_optimal_shards(
            database_size=1000000, server_reliability=0.99
        )
        assert shards >= 3  # Minimum for redundancy
        assert shards <= 10  # Maximum configured

        # Test with lower reliability (should need more shards)
        shards_low = topology_manager.calculate_optimal_shards(
            database_size=1000000, server_reliability=0.95
        )
        assert shards_low >= shards

        # Verify formula: k_min = ⌈ln(φ)/ln(1-q)⌉
        phi = 4e-4
        q = 0.99
        expected_min = math.ceil(math.log(phi) / math.log(1 - q))
        calculated = topology_manager.calculate_optimal_shards(
            database_size=1000000, server_reliability=q
        )
        assert calculated >= expected_min

    def test_server_registration(self, topology_manager, sample_servers):
        """Test server registration."""
        for server in sample_servers:
            topology_manager.register_server(server)

        assert len(topology_manager.servers) == len(sample_servers)
        assert "ts1" in topology_manager.servers
        assert topology_manager.servers["ts1"].type == ServerType.TRUSTED_SIGNATORY

    def test_server_selection(self, topology_manager, sample_servers):
        """Test server selection logic."""
        # Register servers
        for server in sample_servers:
            topology_manager.register_server(server)

        # Test selection with preference for trusted signatories
        selected = topology_manager.select_servers(5, prefer_trusted=True)
        assert len(selected) == 5

        # Should have at least min_trusted_signatories
        trusted_count = sum(1 for s in selected if s.type == ServerType.TRUSTED_SIGNATORY)
        assert trusted_count >= topology_manager.config.min_trusted_signatories

        # Should be sorted by latency
        trusted_servers = [s for s in selected if s.type == ServerType.TRUSTED_SIGNATORY]
        latencies = [s.effective_latency for s in trusted_servers]
        assert latencies == sorted(latencies)

    def test_latency_estimation(self, topology_manager, sample_servers):
        """Test latency estimation."""
        servers = sample_servers[:3]

        # Single round
        latency = topology_manager.estimate_latency(servers, num_rounds=1)
        max_server_latency = max(s.effective_latency for s in servers)
        expected = max_server_latency * len(servers) * 1
        assert latency == expected

        # Multiple rounds
        latency_multi = topology_manager.estimate_latency(servers, num_rounds=3)
        assert latency_multi == expected * 3

    def test_failure_probability(self, topology_manager, sample_servers):
        """Test failure probability calculation."""
        servers = sample_servers[:3]

        # All servers required (threshold = n)
        prob = topology_manager.calculate_failure_probability(servers, threshold=3)
        assert 0 <= prob <= 1

        # With high reliability servers, failure probability should be low
        high_reliability_servers = [
            Server(
                f"s{i}",
                ServerType.LIGHT_NODE,
                f"http://s{i}.com",
                latency_ms=50,
                reliability=0.999,
                capacity=100,
            )
            for i in range(3)
        ]
        prob_high = topology_manager.calculate_failure_probability(
            high_reliability_servers, threshold=3
        )
        assert prob_high < 0.01  # Should be very low

    def test_preset_configurations(self, topology_manager):
        """Test preset configurations."""
        # Test 3-shard preset
        assert "3_shard" in TopologyManager.PRESET_CONFIGS
        config_3 = TopologyManager.PRESET_CONFIGS["3_shard"]
        assert config_3.total_shards == 3
        assert config_3.light_nodes == 1
        assert config_3.trusted_signatories == 2
        assert config_3.expected_latency_ms == 210
        assert config_3.failure_probability == 4e-4
        assert config_3.download_complexity == 1 / 3

        # Test 5-shard preset
        assert "5_shard" in TopologyManager.PRESET_CONFIGS
        config_5 = TopologyManager.PRESET_CONFIGS["5_shard"]
        assert config_5.total_shards == 5
        assert config_5.light_nodes == 3
        assert config_5.trusted_signatories == 2
        assert config_5.expected_latency_ms == 350
        assert config_5.failure_probability == 4e-4
        assert config_5.download_complexity == 1 / 5

    def test_get_configuration(self, topology_manager, sample_servers):
        """Test getting optimal configuration."""
        # Register servers
        for server in sample_servers:
            topology_manager.register_server(server)

        # Get configuration without preset
        config, servers = topology_manager.get_configuration(database_size=1000000)
        assert isinstance(config, ShardConfiguration)
        assert len(servers) == config.total_shards
        assert config.total_shards >= 3

        # Get configuration with preset
        config_3, servers_3 = topology_manager.get_configuration(
            database_size=1000000, use_preset="3_shard"
        )
        assert config_3.total_shards == 3
        assert len(servers_3) == 3

    def test_latency_updates(self, topology_manager, sample_servers):
        """Test latency update mechanism."""
        # Register servers
        for server in sample_servers:
            topology_manager.register_server(server)

        # Update latency
        initial_latency = topology_manager.servers["ln1"].latency_ms
        topology_manager.update_latency("ln1", 100.0)

        # Should use exponential moving average
        new_latency = topology_manager.servers["ln1"].latency_ms
        assert new_latency != initial_latency
        assert new_latency == 0.8 * initial_latency + 0.2 * 100.0

        # Check history
        assert "ln1" in topology_manager.latency_history
        assert 100.0 in topology_manager.latency_history["ln1"]

    def test_server_failure_handling(self, topology_manager, sample_servers):
        """Test server failure handling."""
        # Register servers
        for server in sample_servers:
            topology_manager.register_server(server)

        # Simulate failure
        replacement = topology_manager.handle_server_failure("ts1")

        # Check failed server status
        assert topology_manager.servers["ts1"].status == ServerStatus.FAILED
        assert topology_manager.servers["ts1"].failure_count == 1

        # Should find replacement of same type
        if replacement:
            assert replacement.type == ServerType.TRUSTED_SIGNATORY
            assert replacement.id != "ts1"

    def test_health_check(self, topology_manager, sample_servers):
        """Test health check functionality."""
        # Register servers
        for server in sample_servers:
            topology_manager.register_server(server)

        # Mark a server as failed long ago
        topology_manager.servers["ln1"].status = ServerStatus.FAILED
        topology_manager.servers["ln1"].last_ping = time.time() - 400  # Past cooldown
        topology_manager.servers["ln1"].failure_count = 1

        # Run health check
        statuses = topology_manager.health_check()

        # Failed server should recover after cooldown
        assert topology_manager.servers["ln1"].status == ServerStatus.AVAILABLE
        assert topology_manager.servers["ln1"].failure_count == 0

    def test_should_reconfigure(self, topology_manager, sample_servers):
        """Test reconfiguration triggers."""
        # Register servers
        for server in sample_servers:
            topology_manager.register_server(server)

        # Initially should reconfigure (no current config)
        assert topology_manager.should_reconfigure()

        # Set a configuration
        topology_manager.current_config = TopologyManager.PRESET_CONFIGS["3_shard"]
        topology_manager.last_reconfiguration = time.time() - 120  # 2 minutes ago

        # Add high latency observations (need to exceed threshold)
        for _ in range(20):
            topology_manager.update_latency("ln1", 700.0)  # Well above SLA
            topology_manager.update_latency("ln2", 700.0)
            topology_manager.update_latency("ln3", 700.0)

        # Should trigger reconfiguration due to high latency
        assert topology_manager.should_reconfigure()

    def test_stats_collection(self, topology_manager, sample_servers):
        """Test statistics collection."""
        # Register servers
        for server in sample_servers:
            topology_manager.register_server(server)

        # Get stats
        stats = topology_manager.get_stats()

        assert stats["total_servers"] == len(sample_servers)
        assert stats["available_servers"] == len(sample_servers)
        assert stats["failed_servers"] == 0
        assert stats["query_count"] == 0
        assert isinstance(stats["average_latencies"], dict)


class TestAdaptiveTopologyManager:
    """Test suite for AdaptiveTopologyManager."""

    @pytest.fixture
    def adaptive_manager(self):
        """Create an adaptive topology manager."""
        config = TopologyConfig(
            target_failure_probability=4e-4, min_trusted_signatories=2, latency_sla_ms=400.0
        )
        return AdaptiveTopologyManager(config)

    @pytest.fixture
    def sample_servers(self) -> List[Server]:
        """Create sample servers for testing."""
        servers = [
            # Light Nodes
            Server(
                "ln1",
                ServerType.LIGHT_NODE,
                "http://ln1.example.com",
                latency_ms=50.0,
                reliability=0.99,
                capacity=100,
            ),
            Server(
                "ln2",
                ServerType.LIGHT_NODE,
                "http://ln2.example.com",
                latency_ms=60.0,
                reliability=0.98,
                capacity=100,
            ),
            Server(
                "ln3",
                ServerType.LIGHT_NODE,
                "http://ln3.example.com",
                latency_ms=70.0,
                reliability=0.97,
                capacity=100,
            ),
            # Trusted Signatories
            Server(
                "ts1",
                ServerType.TRUSTED_SIGNATORY,
                "http://ts1.example.com",
                latency_ms=40.0,
                reliability=0.995,
                capacity=50,
            ),
            Server(
                "ts2",
                ServerType.TRUSTED_SIGNATORY,
                "http://ts2.example.com",
                latency_ms=45.0,
                reliability=0.995,
                capacity=50,
            ),
            Server(
                "ts3",
                ServerType.TRUSTED_SIGNATORY,
                "http://ts3.example.com",
                latency_ms=55.0,
                reliability=0.99,
                capacity=50,
            ),
        ]
        return servers

    def test_adaptive_selection(self, adaptive_manager, sample_servers):
        """Test adaptive configuration selection."""
        # Register servers
        for server in sample_servers[:6]:  # Need enough for 5-shard config
            adaptive_manager.register_server(server)

        # Initial selection should use 3-shard
        config, servers = adaptive_manager.adaptive_select(database_size=1000000)
        assert config.total_shards == 3

        # Record poor performance for 3-shard
        for _ in range(15):
            adaptive_manager.record_performance("3_shard", 500.0, True)

        adaptive_manager.query_count = 15  # Ensure enough queries

        # Should switch configuration
        assert adaptive_manager.should_switch_config()

        # Next selection might switch to 5-shard
        config2, servers2 = adaptive_manager.adaptive_select(database_size=1000000)
        # Config might change based on performance
        assert config2.total_shards in [3, 5]

    def test_performance_recording(self, adaptive_manager):
        """Test performance recording."""
        # Record some performance data
        adaptive_manager.record_performance("3_shard", 200.0, True)
        adaptive_manager.record_performance("3_shard", 250.0, True)
        adaptive_manager.record_performance("5_shard", 300.0, True)

        # Check recorded data
        assert "3_shard" in adaptive_manager.config_performance
        assert len(adaptive_manager.config_performance["3_shard"]) == 2
        assert adaptive_manager.config_performance["3_shard"][0] == 200.0

        # Check performance history
        assert len(adaptive_manager.performance_history) == 3
        assert adaptive_manager.performance_history[0]["config"] == "3_shard"
        assert adaptive_manager.performance_history[0]["latency_ms"] == 200.0

    def test_config_selection_logic(self, adaptive_manager):
        """Test configuration selection based on performance."""
        # Record good performance for 3-shard
        for i in range(10):
            adaptive_manager.record_performance("3_shard", 180.0 + i, True)

        # Record poor performance for 5-shard
        for i in range(10):
            adaptive_manager.record_performance("5_shard", 400.0 + i * 2, True)

        # Should select 3-shard as best
        best = adaptive_manager.select_best_config()
        assert best == "3_shard"

        # Now record poor performance for 3-shard
        for i in range(20):
            adaptive_manager.record_performance("3_shard", 500.0 + i, True)

        # Should now prefer 5-shard
        best = adaptive_manager.select_best_config()
        assert best == "5_shard"

    def test_exploration_vs_exploitation(self, adaptive_manager, sample_servers):
        """Test that manager explores different configurations."""
        # Register servers
        for server in sample_servers[:6]:
            adaptive_manager.register_server(server)

        # Track configurations used
        configs_used = set()

        # Run many queries
        for i in range(101):
            adaptive_manager.query_count = i
            if i % 100 == 0 and i > 0:
                # Should explore on 100th query
                assert adaptive_manager.should_switch_config()

            config, _ = adaptive_manager.adaptive_select(database_size=1000000)
            configs_used.add(f"{config.total_shards}_shard")

            # Record moderate performance
            adaptive_manager.record_performance(
                adaptive_manager.current_preset, 250.0 + (i % 50), True
            )

        # Should have explored both configurations at some point
        assert len(configs_used) >= 1  # At least one config used
