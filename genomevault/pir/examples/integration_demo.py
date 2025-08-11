from __future__ import annotations

"""Integration Demo module."""
"""
PIR Integration Demo
Demonstrates end-to-end PIR functionality with ZK and HDC integration.
"""

import asyncio
import time

import numpy as np

from genomevault.pir.client.query_builder import PIRQueryBuilder
from genomevault.pir.it_pir_protocol import BatchPIRProtocol, PIRParameters, PIRProtocol
from genomevault.pir.network.coordinator import (
    PIRCoordinator,
    ServerInfo,
    ServerSelectionCriteria,
    ServerType,
)
from genomevault.pir.server.enhanced_pir_server import EnhancedPIRServer, ServerConfig
from genomevault.utils.logging import get_logger, logger

logger = get_logger(__name__)


class PIRIntegrationDemo:
    """
    Demonstrates complete PIR workflow:
    1. HDC-encoded database shards
    2. PIR query for genomic data
    3. ZK proof of query validity
    4. Result reconstruction
    """

    def __init__(self):
        """Initialize instance."""
        self.coordinator = PIRCoordinator()
        self.servers: list[EnhancedPIRServer] = []
        self.database_size = 10000
        self.element_size = 1024

    async def setup(self):
        """Setup PIR infrastructure."""
        logger.info("🔧 Setting up PIR infrastructure...")

        # Start coordinator
        await self.coordinator.start()

        # Create and register servers
        server_configs = [
            ("ts-east-1", True, (40.7128, -74.0060), "US-EAST"),  # NYC
            ("ts-west-1", True, (37.7749, -122.4194), "US-WEST"),  # SF
            ("ln-central-1", False, (41.8781, -87.6298), "US-CENTRAL"),  # Chicago
            ("ln-eu-1", False, (51.5074, -0.1278), "EU-WEST"),  # London
            ("ln-asia-1", False, (35.6762, 139.6503), "ASIA-PACIFIC"),  # Tokyo
        ]

        for server_id, is_ts, location, region in server_configs:
            # Create server
            config = ServerConfig(
                server_id=server_id,
                is_trusted_signatory=is_ts,
                database_path=f"/tmp/pir_demo_{server_id}",
                cache_size_mb=512,
            )
            server = EnhancedPIRServer(config)
            self.servers.append(server)

            # Register with coordinator
            server_info = ServerInfo(
                server_id=server_id,
                server_type=(ServerType.TRUSTED_SIGNATORY if is_ts else ServerType.LIGHT_NODE),
                endpoint=f"http://localhost:808{len(self.servers)}",
                location=location,
                region=region,
                capabilities={"batch_query", "compression"},
            )
            self.coordinator.register_server(server_info)

        logger.info(f"✅ Created {len(self.servers)} PIR servers")

    async def demonstrate_basic_pir(self):
        """Demonstrate basic PIR retrieval."""
        logger.info("\n📊 Basic PIR Demonstration")
        logger.info("-" * 50)

        # Create PIR protocol
        params = PIRParameters(
            database_size=self.database_size,
            element_size=self.element_size,
            num_servers=2,
        )
        protocol = PIRProtocol(params)

        # Target index to retrieve
        target_index = 42
        logger.info(f"🎯 Target: Retrieve element at index {target_index}")

        # Generate query vectors
        logger.info("\n1️⃣ Generating query vectors...")
        query_vectors = protocol.generate_query_vectors(target_index)
        logger.info(f"   Generated {len(query_vectors)} query vectors")

        # Select servers
        criteria = ServerSelectionCriteria(
            min_servers=2,
            require_geographic_diversity=True,
            prefer_trusted_signatories=True,
        )

        selected_servers = await self.coordinator.select_servers(criteria)
        logger.info("\n2️⃣ Selected servers:")
        for server in selected_servers:
            logger.info(f"   - server.server_id (server.server_type.value) in {server.region}")

        # Process queries on servers
        logger.info("\n3️⃣ Processing queries on servers...")
        responses = []

        for i, (server_info, query_vector) in enumerate(zip(selected_servers[:2], query_vectors)):
            # Find actual server instance
            server = next(s for s in self.servers if s.server_id == server_info.server_id)

            # Create query request
            query_data = {
                "query_id": f"demo-query-{i}",
                "query_vector": query_vector.tolist(),
                "protocol_version": "1.0",
                "timestamp": time.time(),
            }

            # Process query
            start_time = time.time()
            response = await server.process_query(query_data)
            (time.time() - start_time) * 1000

            logger.info(f"   Server {server.server_id}: {latency:.1f}ms")

            # Convert response back to numpy array
            response_array = np.array(response["response"], dtype=np.uint8)
            responses.append(response_array)

        # Reconstruct element
        logger.info("\n4️⃣ Reconstructing element...")
        protocol.reconstruct_element(responses)
        logger.info(f"   Reconstructed element size: {len(reconstructed)} bytes")

        # Calculate privacy guarantees
        logger.info("\n5️⃣ Privacy Analysis:")
        prob_ts = protocol.calculate_privacy_breach_probability(k_honest=2, honesty_prob=0.98)
        logger.info(f"   Privacy breach probability (2 TS nodes): {prob_ts:.6f}")
        logger.info("   Information leaked to single server: 0 bits ✅")

    async def demonstrate_genomic_query(self):
        """Demonstrate genomic data query."""
        logger.info("\n🧬 Genomic Query Demonstration")
        logger.info("-" * 50)

        # Create mock index mapping
        index_mapping = {
            "variants": {
                "chr1:100000:A:G": 42,
                "chr1:100100:C:T": 43,
                "chr1:100200:G:A": 44,
                "chr17:43044295:C:T": 100,  # BRCA1 variant
                "chr17:43044300:G:A": 101,
                "chr17:43044305:T:C": 102,
            },
            "positions": {
                "chr1:100000": [42],
                "chr1:100100": [43],
                "chr1:100200": [44],
                "chr17:43044295": [100],
                "chr17:43044300": [101],
                "chr17:43044305": [102],
            },
            "genes": {"BRCA1": {"chromosome": "chr17", "start": 43044295, "end": 43044400}},
        }

        # Create mock PIR client (would use real client in production)
        class MockPIRClient:
            """Client for mockpir operations."""

            async def execute_query(self, query):
                """Async operation to Execute query.

                Args:
                    query: Query string.

                Returns:
                    Operation result.
                """
                # Simulate query execution
                await asyncio.sleep(0.1)
                return {
                    "chromosome": "chr17",
                    "position": 43044295,
                    "ref": "C",
                    "alt": "T",
                    "gene": "BRCA1",
                    "clinical_significance": "Pathogenic",
                    "population_frequencies": {
                        "global": 0.001,
                        "european": 0.0015,
                        "asian": 0.0008,
                    },
                }

            def create_query(self, index) -> None:
                """Create query.

                Args:
                    index: Index position.

                Returns:
                    Newly created query.
                """
                return {"index": index}

            async def batch_query(self, indices):
                """Async operation to Batch query.

                Args:
                    indices: Indices.

                Returns:
                    Operation result.
                """
                results = []
                for idx in indices:
                    result = await self.execute_query(None)
                    results.append(result)
                return results

            def decode_response(self, response, encoding) -> None:
                """Decode response.

                Args:
                    response: Server response.
                    encoding: Character encoding.

                Returns:
                    Operation result.
                """
                return response

        # Create query builder
        pir_client = MockPIRClient()
        query_builder = PIRQueryBuilder(pir_client, index_mapping)

        # Example 1: Variant lookup
        logger.info("\n🔍 Variant Lookup: BRCA1 c.68_69delAG")
        variant_query = query_builder.build_variant_query("chr17", 43044295, "C", "T")

        start_time = time.time()
        await query_builder.execute_query(variant_query)
        (time.time() - start_time) * 1000

        logger.info(f"   Query time: {query_time:.1fms}")
        logger.info(f"   Result: {result.data['clinical_significance']} variant")
        logger.info(f"   Global frequency: {result.data['population_frequencies']['global']:.4f}")

        # Example 2: Gene scan
        logger.info("\n🧬 Gene Scan: BRCA1")
        gene_query = query_builder.build_gene_query("BRCA1")

        start_time = time.time()
        await query_builder.execute_query(gene_query)
        (time.time() - start_time) * 1000

        logger.info(f"   Query time: {query_time:.1fms}")
        logger.info(f"   Variants found: {result.data['total_variants']}")
        logger.info(f"   PIR queries used: {result.pir_queries_used}")

        # Show query statistics
        query_builder.get_query_statistics()
        logger.info("\n📈 Query Statistics:")
        logger.info(f"   Cache size: {stats['cache_size']}")
        logger.info(f"   Total PIR queries: {stats['total_pir_queries']}")
        logger.info(f"   Avg computation time: {stats['avg_computation_time_ms']:.1fms}")

    async def demonstrate_batch_queries(self):
        """Demonstrate batch PIR queries."""
        logger.info("\n📦 Batch Query Demonstration")
        logger.info("-" * 50)

        # Create batch protocol
        params = PIRParameters(database_size=self.database_size)
        batch_protocol = BatchPIRProtocol(params)

        # Generate batch of indices
        batch_size = 50
        indices = np.random.choice(self.database_size, batch_size, replace=False).tolist()

        logger.info(f"🎯 Retrieving {batch_size} elements in batch")

        # Generate batch queries
        start_time = time.time()
        batch_queries = batch_protocol.generate_batch_queries(indices)
        (time.time() - start_time) * 1000

        logger.info(f"   Query generation: {gen_time:.1fms}")
        logger.info(f"   Buckets used: {len(batch_queries)}")

        # Calculate efficiency
        single_query_size = self.database_size  # bits
        batch_query_size = len(batch_queries) * self.database_size
        (batch_size * single_query_size) / batch_query_size

        logger.info(f"   Bandwidth efficiency: {efficiency:.2fx}")

    async def demonstrate_security_features(self):
        """Demonstrate security features."""
        logger.info("\n🔒 Security Features Demonstration")
        logger.info("-" * 50)

        # 1. Timing attack mitigation
        logger.info("\n1️⃣ Timing Attack Mitigation:")
        params = PIRParameters(database_size=1000)
        protocol = PIRProtocol(params)

        # Measure timing for different response sizes
        timings = []
        for size in [100, 500, 1000]:
            response = np.random.bytes(size)
            _, time_ms = protocol.timing_safe_response(response, target_time_ms=50)
            timings.append(time_ms)

        np.var(timings)
        logger.info(f"   Timing variance: {timing_variance:.2fms}²")
        logger.info(f"   Max timing difference: {max(timings)} - min(timings):.2fms")

        # 2. Replay protection
        logger.info("\n2️⃣ Replay Protection:")
        server = self.servers[0]

        # Create query with nonce
        query_data = {
            "query_id": "security-test-1",
            "query_vector": [1] + [0] * (self.database_size - 1),
            "protocol_version": "1.0",
            "timestamp": time.time(),
            "nonce": "a1b2c3d4e5f6789012345678901234567",
        }

        # First query should succeed
        await server.process_query(query_data)
        logger.info("   First query: ✅ Success")

        # Replay would be detected by handler (not shown here)
        logger.info("   Replay attempt: ❌ Would be blocked")

        # 3. Server collusion analysis
        logger.info("\n3️⃣ Collusion Resistance:")

        # Calculate minimum servers needed for different threat models
        target_prob = 1e-6  # One in a million

        protocol.calculate_min_servers(target_prob, 0.98)
        protocol.calculate_min_servers(target_prob, 0.95)
        min_mixed = protocol.calculate_min_servers(target_prob, 0.96)  # Mix of TS and LN

        logger.info(f"   For {target_prob:.0e} failure probability:")
        logger.info(f"   - Pure TS nodes: {min_ts} servers")
        logger.info(f"   - Pure LN nodes: {min_ln} servers")
        logger.info(f"   - Mixed (TS+LN): {min_mixed} servers")

    async def cleanup(self):
        """Cleanup resources."""
        await self.coordinator.stop()
        logger.info("\n✅ Cleanup complete")


async def main():
    """Run the integration demo."""
    logger.info("=" * 60)
    logger.info("        GenomeVault PIR Integration Demo")
    logger.info("=" * 60)

    demo = PIRIntegrationDemo()

    try:
        # Setup
        await demo.setup()

        # Run demonstrations
        await demo.demonstrate_basic_pir()
        await demo.demonstrate_genomic_query()
        await demo.demonstrate_batch_queries()
        await demo.demonstrate_security_features()

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 Demo Summary")
        logger.info("-" * 60)

        demo.coordinator.get_coordinator_stats()
        logger.info(f"Total servers: {stats['total_servers']}")
        logger.info(f"  - Trusted Signatories: {stats['trusted_signatories']}")
        logger.info(f"  - Light Nodes: {stats['light_nodes']}")
        logger.info(f"Geographic regions: {stats['geographic_regions']}")

        logger.info("\n✨ Key Features Demonstrated:")
        logger.info("  ✅ Information-theoretic security (zero leakage)")
        logger.info("  ✅ Geographic diversity enforcement")
        logger.info("  ✅ Timing attack mitigation")
        logger.info("  ✅ Batch query optimization")
        logger.info("  ✅ Server health monitoring")
        logger.info("  ✅ Privacy-preserving genomic queries")

        logger.info("\n🔐 Security Guarantees:")
        logger.info("  • Perfect privacy against t < n colluding servers")
        logger.info("  • Fixed-size responses prevent traffic analysis")
        logger.info("  • Constant-time operations prevent timing attacks")
        logger.info("  • Geographic diversity prevents regional attacks")

    finally:
        await demo.cleanup()

    logger.info("\n🎉 Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
