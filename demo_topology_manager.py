#!/usr/bin/env python3
"""
Demo of PIR Topology Manager.

Shows optimal shard configuration, server selection, and adaptive management.
"""

import time
import random
from genomevault.pir.topology_manager import (
    AdaptiveTopologyManager,
    Server,
    ServerType,
    TopologyConfig,
)


def create_demo_servers():
    """Create a realistic set of PIR servers."""
    servers = []

    # Light Nodes - geographically distributed
    light_nodes = [
        ("ln-us-east", "us-east-1", 20),
        ("ln-us-west", "us-west-2", 35),
        ("ln-eu-west", "eu-west-1", 85),
        ("ln-ap-south", "ap-southeast-1", 120),
        ("ln-sa-east", "sa-east-1", 150),
    ]

    for node_id, region, base_latency in light_nodes:
        servers.append(
            Server(
                id=node_id,
                type=ServerType.LIGHT_NODE,
                endpoint=f"https://{node_id}.pir.genomevault.io",
                latency_ms=base_latency + random.uniform(-5, 15),
                reliability=0.97 + random.uniform(0, 0.02),
                capacity=100,
            )
        )

    # Trusted Signatories - higher reliability, lower capacity
    trusted_sigs = [
        ("ts-hospital-1", "Boston Medical Center", 25),
        ("ts-hospital-2", "Mayo Clinic", 40),
        ("ts-research-1", "Broad Institute", 30),
        ("ts-pharma-1", "Pfizer Research", 45),
    ]

    for sig_id, org, base_latency in trusted_sigs:
        servers.append(
            Server(
                id=sig_id,
                type=ServerType.TRUSTED_SIGNATORY,
                endpoint=f"https://{sig_id}.secure.genomevault.io",
                latency_ms=base_latency + random.uniform(-3, 8),
                reliability=0.995 + random.uniform(0, 0.004),
                capacity=50,
            )
        )

    # Full Nodes - highest capacity
    servers.append(
        Server(
            id="fn-primary",
            type=ServerType.FULL_NODE,
            endpoint="https://primary.genomevault.io",
            latency_ms=15,
            reliability=0.999,
            capacity=500,
        )
    )

    return servers


def simulate_pir_queries(manager, num_queries=20):
    """Simulate PIR queries with varying conditions."""

    print("\n" + "=" * 70)
    print("  SIMULATING PIR QUERIES")
    print("=" * 70)

    database_size = 1_000_000  # 1M genomic records

    for i in range(num_queries):
        print(f"\n📊 Query {i+1}/{num_queries}")
        print("-" * 40)

        # Get adaptive configuration
        config, servers = manager.adaptive_select(database_size)

        print(
            f"Configuration: {config.total_shards} shards "
            f"({config.light_nodes} LN + {config.trusted_signatories} TS)"
        )
        print("Selected servers:")
        for server in servers:
            print(
                f"  • {server.id:15} ({server.type.value}): "
                f"{server.latency_ms:.1f}ms, {server.reliability:.1%} reliable"
            )

        # Simulate query execution
        base_latency = manager.estimate_latency(servers)

        # Add some variance to simulate real conditions
        if random.random() < 0.1:  # 10% chance of network congestion
            actual_latency = base_latency * random.uniform(1.5, 2.5)
            print("⚠️  Network congestion detected!")
        else:
            actual_latency = base_latency * random.uniform(0.9, 1.1)

        # Simulate occasional failures
        success = random.random() > 0.02  # 98% success rate

        if not success:
            print("❌ Query failed - handling failure...")
            # Simulate server failure
            failed_server = random.choice(servers)
            replacement = manager.handle_server_failure(failed_server.id)
            if replacement:
                print(f"   Replaced {failed_server.id} with {replacement.id}")
        else:
            print("✅ Query successful")

        print(f"Latency: {actual_latency:.1f}ms " f"(expected: {config.expected_latency_ms:.1f}ms)")
        print(f"Download complexity: O(N^{config.download_complexity:.2f})")
        print(f"Failure probability: {config.failure_probability:.2e}")

        # Update manager with observed performance
        for server in servers:
            observed = server.latency_ms * random.uniform(0.8, 1.2)
            manager.update_latency(server.id, observed)

        manager.record_performance(manager.current_preset, actual_latency, success)

        # Small delay between queries
        time.sleep(0.1)

        # Check if reconfiguration is needed
        if manager.should_reconfigure():
            print("\n🔄 Triggering topology reconfiguration...")
            manager.last_reconfiguration = time.time()


def main():
    """Run the topology manager demo."""

    print("\n" + "=" * 70)
    print("  PIR TOPOLOGY MANAGER DEMO")
    print("=" * 70)
    print("\nDemonstrating optimal shard configuration and adaptive selection")
    print("based on Section 2.2.1 specifications.")

    # Create configuration
    config = TopologyConfig(
        target_failure_probability=4e-4,  # 4×10^-4
        min_trusted_signatories=2,
        max_shards=7,
        latency_sla_ms=400.0,
        adaptive_threshold=0.8,
    )

    # Create adaptive manager
    manager = AdaptiveTopologyManager(config)

    # Register demo servers
    servers = create_demo_servers()
    for server in servers:
        manager.register_server(server)

    print(f"\n📡 Registered {len(servers)} servers:")
    print(f"   • Light Nodes: {sum(1 for s in servers if s.type == ServerType.LIGHT_NODE)}")
    print(
        f"   • Trusted Signatories: {sum(1 for s in servers if s.type == ServerType.TRUSTED_SIGNATORY)}"
    )
    print(f"   • Full Nodes: {sum(1 for s in servers if s.type == ServerType.FULL_NODE)}")

    # Show preset configurations
    print("\n🎯 Preset Configurations:")
    for name, preset in AdaptiveTopologyManager.PRESET_CONFIGS.items():
        print(f"\n   {name}:")
        print(
            f"      Shards: {preset.total_shards} ({preset.light_nodes} LN + {preset.trusted_signatories} TS)"
        )
        print(f"      Expected latency: {preset.expected_latency_ms}ms")
        print(f"      Failure probability: {preset.failure_probability:.2e}")
        print(f"      Download complexity: O(N^{preset.download_complexity:.2f})")

    # Calculate optimal shards for different reliability levels
    print("\n📐 Optimal Shard Calculations (k_min = ⌈ln(φ)/ln(1-q)⌉):")
    print(f"   Target failure probability (φ): {config.target_failure_probability:.2e}")
    for reliability in [0.99, 0.95, 0.90]:
        k_optimal = manager.calculate_optimal_shards(1_000_000, reliability)
        print(f"   Server reliability {reliability:.0%}: {k_optimal} shards needed")

    # Run simulation
    simulate_pir_queries(manager, num_queries=15)

    # Show final statistics
    print("\n" + "=" * 70)
    print("  FINAL STATISTICS")
    print("=" * 70)

    stats = manager.get_stats()
    print("\n📊 Performance Summary:")
    print(f"   Total queries: {stats['query_count']}")
    print(f"   Available servers: {stats['available_servers']}/{stats['total_servers']}")
    print(f"   Failed servers: {stats['failed_servers']}")

    if stats["average_latencies"]:
        print("\n⏱️  Average Latencies by Server:")
        for server_id, avg_latency in sorted(
            stats["average_latencies"].items(), key=lambda x: x[1]
        ):
            print(f"   • {server_id:15}: {avg_latency:.1f}ms")

    # Show configuration performance
    print("\n🎯 Configuration Performance:")
    for config_name, latencies in manager.config_performance.items():
        if latencies:
            avg = sum(latencies) / len(latencies)
            print(f"   • {config_name}: {avg:.1f}ms average ({len(latencies)} queries)")

    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    main()
