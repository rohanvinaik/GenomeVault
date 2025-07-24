from typing import Dict

"""
PIR system integration example.

Demonstrates the complete workflow for privacy-preserving genomic data retrieval.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Dict

from genomevault.pir import (
    PangenomeNode,
    PIRClient,
    PIRNetworkCoordinator,
    PIRQueryBuilder,
    PIRServer,
    PIRServerInstance,
    ReferenceDataManager,
    ShardManager,
    TrustedSignatoryServer,
    VariantAnnotation,
)


async def setup_pir_network(data_dir: Path) -> Dict:
    """Set up a complete PIR network for testing."""

    # 1. Create reference data
    print("\n=== Setting up Reference Data ===")
    ref_manager = ReferenceDataManager(data_dir / "reference")

    # Add pangenome nodes
    for i in range(1000):
        node = PangenomeNode(
            node_id=i,
            sequence="ACGT" * 25,  # 100bp sequence
            chromosome="chr1" if i < 500 else "chr2",
            position=1000000 + i * 100,
            populations={"EUR", "AFR"} if i % 3 == 0 else {"EAS", "SAS"},
            frequency=0.01 + (i % 100) * 0.001,
        )
        ref_manager.add_node(node)

    # Add variant annotations
    for i in range(200):
        annotation = VariantAnnotation(
            chromosome="chr1",
            position=1000000 + i * 500,
            ref_allele="A",
            alt_allele="G" if i % 2 == 0 else "T",
            gene_impact="HIGH" if i % 10 == 0 else "MODERATE",
            conservation_score=0.5 + (i % 50) * 0.01,
            pathogenicity_score=0.1 + (i % 20) * 0.04,
            population_frequencies={
                "EUR": 0.05 + (i % 10) * 0.01,
                "AFR": 0.03 + (i % 8) * 0.01,
                "EAS": 0.02 + (i % 12) * 0.01,
            },
            clinical_significance="Pathogenic" if i % 15 == 0 else None,
        )
        ref_manager.add_variant_annotation(annotation)

    ref_manager.save_reference_data()
    print("Created reference data with {len(ref_manager.nodes)} nodes")

    # 2. Create shards from reference data
    print("\n=== Creating Database Shards ===")
    shard_manager = ShardManager(data_dir / "shards", num_shards=5)

    # Prepare PIR data
    pir_data = ref_manager.prepare_for_pir(ReferenceDataType.PANGENOME_GRAPH)
    combined_data = b"".join(pir_data)

    # Write to temp file
    temp_data_file = data_dir / "temp_data.bin"
    temp_data_file.write_bytes(combined_data)

    # Create shards
    shard_ids = shard_manager.create_shards_from_data(
        temp_data_file, data_type="genomic"
    )
    print("Created {len(shard_ids)} shards")

    # 3. Set up PIR servers
    print("\n=== Starting PIR Servers ===")
    servers = []
    server_instances = {}

    # Create light node servers
    for i in range(3):
        server_id = "ln{i+1}"
        server_dir = data_dir / "server_{server_id}"
        server_dir.mkdir(exist_ok=True)

        # Copy shard data
        import shutil

        shutil.copytree(data_dir / "shards", server_dir, dirs_exist_ok=True)

        # Create server instance
        server = PIRServerInstance(server_id, server_dir, is_trusted_signatory=False)
        server_instances[server_id] = server

        # Create server info
        servers.append(
            PIRServer(
                server_id=server_id,
                endpoint="http://localhost:800{i}",
                region="us-east" if i == 0 else ("eu-west" if i == 1 else "asia-pac"),
                is_trusted_signatory=False,
                honesty_probability=0.95,
                latency_ms=50 + i * 10,
            )
        )

    # Create trusted signatory servers
    for i in range(2):
        server_id = "ts{i+1}"
        server_dir = data_dir / "server_{server_id}"
        server_dir.mkdir(exist_ok=True)

        # Copy shard data
        shutil.copytree(data_dir / "shards", server_dir, dirs_exist_ok=True)

        # Create TS server instance
        server = TrustedSignatoryServer(
            server_id=server_id,
            data_directory=server_dir,
            npi="1234567890",
            baa_hash="a" * 64,
            risk_analysis_hash="b" * 64,
            hsm_serial="HSM123",
        )
        server_instances[server_id] = server

        # Create server info
        servers.append(
            PIRServer(
                server_id=server_id,
                endpoint="http://localhost:801{i}",
                region="us-west" if i == 0 else "us-central",
                is_trusted_signatory=True,
                honesty_probability=0.98,
                latency_ms=40 + i * 5,
            )
        )

    # Distribute shards across servers
    server_list = [s.server_id for s in servers]
    shard_manager.distribute_shards(server_list)

    print("Started {len(servers)} PIR servers (3 LN + 2 TS)")

    # 4. Create index mapping
    index_mapping = create_index_mapping(ref_manager)

    return {
        "servers": servers,
        "server_instances": server_instances,
        "ref_manager": ref_manager,
        "shard_manager": shard_manager,
        "index_mapping": index_mapping,
        "database_size": len(pir_data),
    }


def create_index_mapping(ref_manager: ReferenceDataManager) -> Dict:
    """Create index mapping for PIR queries."""
    index_mapping = {
        "variants": {},
        "positions": {},
        "genes": {
            "GENE1": {"chromosome": "chr1", "start": 1000000, "end": 1050000},
            "GENE2": {"chromosome": "chr2", "start": 1050000, "end": 1100000},
        },
    }

    # Map variants to indices
    idx = 0
    for key, annotation in ref_manager.variant_annotations.items():
        index_mapping["variants"][key] = idx

        pos_key = "{annotation.chromosome}:{annotation.position}"
        if pos_key not in index_mapping["positions"]:
            index_mapping["positions"][pos_key] = []
        index_mapping["positions"][pos_key].append(idx)

        idx += 1

    return index_mapping


async def demonstrate_pir_queries(network_info: Dict):
    """Demonstrate various PIR queries."""

    # 1. Create PIR client
    print("\n=== Creating PIR Client ===")
    pir_client = PIRClient(network_info["servers"], network_info["database_size"])

    # Show optimal configuration
    pir_client.get_optimal_server_configuration()
    print("Optimal configuration: {optimal_config['optimal']['name']}")
    print(
        "Privacy failure probability: {optimal_config['optimal']['failure_probability']:.2e}"
    )
    print("Expected latency: {optimal_config['optimal']['latency_ms']:.0f}ms")

    # 2. Create query builder
    builder = PIRQueryBuilder(pir_client, network_info["index_mapping"])

    # 3. Execute variant lookup
    print("\n=== Variant Lookup Query ===")
    var_query = builder.build_variant_query(
        chromosome="chr1", position=1000500, ref_allele="A", alt_allele="G"
    )

    # Simulate query execution (in real system would actually query servers)
    print("Looking up variant: chr1:1000500:A>G")
    print(
        "Query preserves privacy - servers don't know which variant is being accessed"
    )

    # 4. Execute region scan
    print("\n=== Region Scan Query ===")
    region_query = builder.build_region_query(
        chromosome="chr1", start=1000000, end=1005000
    )

    print("Scanning region: chr1:1000000-1005000")
    print("Multiple PIR queries executed in parallel")

    # 5. Execute gene query
    print("\n=== Gene Annotation Query ===")
    builder.build_gene_query("GENE1")

    print("Querying gene: GENE1")
    print("Retrieves all variants in gene region with privacy")

    # 6. Population frequency query
    print("\n=== Population Frequency Query ===")
    variants = [
        {"chromosome": "chr1", "position": 1000000},
        {"chromosome": "chr1", "position": 1001000},
        {"chromosome": "chr1", "position": 1002000},
    ]

    builder.build_population_frequency_query(variants, "EUR")

    print("Querying frequencies for {len(variants)} variants in EUR population")
    print("Each variant queried privately")

    # Show statistics
    builder.get_query_statistics()
    print("\nQuery Statistics: {json.dumps(stats, indent = 2)}")

    # Cleanup
    await pir_client.close()


async def demonstrate_network_coordination():
    """Demonstrate PIR network coordination."""

    print("\n=== Network Coordination ===")

    # Create coordinator
    coordinator = PIRNetworkCoordinator()
    await coordinator.start()

    # Get network statistics
    coordinator.get_network_statistics()
    print("Network Statistics:")
    print("  Total servers: {net_stats['total_servers']}")
    print("  Healthy servers: {net_stats['healthy_servers']}")
    print(
        "  TS servers: {net_stats['ts_servers']['total']} (healthy: {net_stats['ts_servers']['healthy']})"
    )
    print(
        "  LN servers: {net_stats['ln_servers']['total']} (healthy: {net_stats['ln_servers']['healthy']})"
    )

    # Get optimal configuration
    config = coordinator.get_server_configuration(
        target_failure_prob=1e-4, max_latency_ms=300
    )

    print("\nAvailable Configurations:")
    for conf in config["configurations"]:
        print(
            "  {conf['name']}: {conf['total_servers']} servers, "
            "P_fail = {conf['failure_probability']:.2e}, "
            "latency = {conf['latency_ms']:.0f}ms"
        )

    if config["optimal"]:
        print("\nOptimal: {config['optimal']['name']}")

    await coordinator.stop()


def demonstrate_privacy_calculations():
    """Demonstrate privacy guarantee calculations."""

    print("\n=== Privacy Guarantee Calculations ===")

    # Create a dummy client for calculations
    servers = [
        PIRServer("ts1", "http://ts1", "us", True, 0.98, 50),
        PIRServer("ts2", "http://ts2", "us", True, 0.98, 60),
        PIRServer("ln1", "http://ln1", "us", False, 0.95, 70),
    ]

    client = PIRClient(servers, 1000000)

    print("Privacy Failure Probability P_fail(k, q) = (1-q)^k")
    print("  k = number of honest servers required")
    print("  q = server honesty probability")

    print("\nCalculations:")

    # Different configurations
    configs = [
        (2, 0.98, "2 HIPAA TS servers"),
        (3, 0.98, "3 HIPAA TS servers"),
        (2, 0.95, "2 generic servers"),
        (3, 0.95, "3 generic servers"),
    ]

    for k, q, desc in configs:
        client.calculate_privacy_failure_probability(k, q)
        print("  {desc}: P_fail = {p_fail:.2e}")

    # Minimum servers needed
    print("\nMinimum servers for target privacy:")
    targets = [1e-4, 1e-5, 1e-6]

    for target in targets:
        client.calculate_min_servers_needed(target, 0.98)
        client.calculate_min_servers_needed(target, 0.95)
        print("  Target P_fail ≤ {target:.0e}:")
        print("    HIPAA TS (q = 0.98): {min_ts} servers")
        print("    Generic (q = 0.95): {min_generic} servers")


async def main():
    """Run complete PIR demonstration."""

    print(" = " * 60)
    print("GenomeVault PIR System Demonstration")
    print("Information-Theoretic Private Information Retrieval")
    print(" = " * 60)

    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)

        # Set up network
        network_info = await setup_pir_network(data_dir)

        # Demonstrate queries
        await demonstrate_pir_queries(network_info)

        # Demonstrate coordination
        await demonstrate_network_coordination()

        # Demonstrate privacy calculations
        demonstrate_privacy_calculations()

        # Show server statistics
        print("\n=== Server Statistics ===")
        for server_id, server in network_info["server_instances"].items():
            server.get_server_statistics()
            print(
                "{server_id}: {stats['server_type']}, "
                "{stats['shards']} shards, "
                "{stats['total_queries']} queries"
            )

        # Cleanup
        for server in network_info["server_instances"].values():
            server.shutdown()

    print("\n" + " = " * 60)
    print("PIR demonstration completed successfully!")
    print(" = " * 60)


if __name__ == "__main__":
    asyncio.run(main())
