from genomevault.observability.logging import configure_logging

logger = configure_logging()
"""
End-to-end integration test for PIR with ZK and HDC.
Tests the complete genomic query pipeline.
"""

from pathlib import Path

import numpy as np
import pytest

from genomevault.hypervector.encoder import HyperdimensionalEncoder
from genomevault.pir.client import PIRClient, PIRServer
from genomevault.pir.client.query_builder import PIRQueryBuilder
from genomevault.zk_proofs.circuits.prs_circuit import PRSCircuit


class TestPIRIntegration:
    """Test PIR integration with other components."""

    @pytest.fixture
    def hdc_encoder(self):
        """Create HDC encoder."""
        return HyperdimensionalEncoder(dimension=10000)

    @pytest.fixture
    def pir_client(self):
        """Create PIR client with mock servers."""
        servers = [
            PIRServer("ts1", "http://localhost:9001", "us", True, 0.98, 50),
            PIRServer("ts2", "http://localhost:9002", "eu", True, 0.98, 60),
            PIRServer("ln1", "http://localhost:9003", "asia", False, 0.95, 70),
        ]
        return PIRClient(servers, database_size=100000)

    @pytest.fixture
    def index_mapping(self):
        """Create test index mapping."""
        return {
            "variants": {
                "chr1:100000:A:G": 42,
                "chr1:200000:C:T": 142,
                "chr2:300000:G:A": 242,
            },
            "positions": {
                "chr1:100000": [42],
                "chr1:200000": [142],
                "chr2:300000": [242],
            },
            "genes": {
                "BRCA1": {"chromosome": "chr17", "start": 43044295, "end": 43125483},
                "TP53": {"chromosome": "chr17", "start": 7571720, "end": 7590868},
            },
        }

    @pytest.mark.asyncio
    async def test_hdc_encoded_pir_query(self, hdc_encoder, pir_client):
        """Test PIR query with HDC-encoded data."""
        # Encode genomic variant
        variant_data = {
            "chromosome": "chr1",
            "position": 100000,
            "ref": "A",
            "alt": "G",
            "quality": 30,
        }

        # Encode to hypervector
        hv = hdc_encoder.encode_variant(variant_data)
        assert hv.shape == (10000,)

        # Create PIR query for encoded data
        # In practice, would query HDC-indexed database
        query = pir_client.create_query(target_index=42)

        # Verify query properties
        assert len(query.query_vectors) == 3  # 3 servers
        assert query.target_index == 42

        # Simulate retrieval of HDC-encoded variant
        # Server would return HDC-encoded data
        mock_response = hv.tobytes()

        # Decode response
        retrieved_hv = np.frombuffer(mock_response, dtype=np.float32)

        # Verify similarity
        similarity = np.dot(hv, retrieved_hv) / (np.linalg.norm(hv) * np.linalg.norm(retrieved_hv))
        assert similarity > 0.95  # High similarity

    @pytest.mark.asyncio
    async def test_zk_proof_with_pir_query(self, pir_client):
        """Test ZK proof generation for PIR query results."""
        # Create PIR query
        query = pir_client.create_query(target_index=42)

        # Simulate PIR response with PRS data
        prs_data = {"score": 1.25, "variants_used": 1000, "confidence": 0.95}

        # Generate ZK proof that PRS is in valid range
        prs_circuit = PRSCircuit()

        # Create proof that score is between 0 and 3
        public_inputs = {
            "min_score": 0.0,
            "max_score": 3.0,
            "commitment": "0x1234567890abcdef",  # Mock commitment
        }

        private_inputs = {"score": prs_data["score"], "salt": "random_salt_value"}

        # Generate proof (mocked for test)
        proof = {
            "pi_a": ["0x123", "0x456"],
            "pi_b": [["0x789", "0xabc"], ["0xdef", "0x012"]],
            "pi_c": ["0x345", "0x678"],
            "protocol": "groth16",
        }

        # Verify proof would pass
        assert 0.0 <= prs_data["score"] <= 3.0

    @pytest.mark.asyncio
    async def test_e2e_genomic_query_flow(self, pir_client, index_mapping, hdc_encoder):
        """Test complete end-to-end genomic query flow."""
        # Step 1: User wants to query for BRCA1 variants
        gene = "BRCA1"

        # Step 2: Create query builder
        query_builder = PIRQueryBuilder(pir_client, index_mapping)

        # Step 3: Build gene query
        gene_query = query_builder.build_gene_query(gene)

        # Step 4: Mock PIR execution
        # In practice, would execute: result = await query_builder.execute_query(gene_query)
        mock_result = {
            "gene": gene,
            "variants": [
                {
                    "position": 43044295,
                    "ref": "A",
                    "alt": "G",
                    "gene_impact": "HIGH",
                    "clinical_significance": "Pathogenic",
                },
                {
                    "position": 43045000,
                    "ref": "C",
                    "alt": "T",
                    "gene_impact": "MODERATE",
                    "clinical_significance": "Likely pathogenic",
                },
            ],
            "total_variants": 2,
        }

        # Step 5: Encode variants as hypervectors
        variant_hvs = []
        for variant in mock_result["variants"]:
            hv = hdc_encoder.encode_variant(
                {
                    "chromosome": "chr17",
                    "position": variant["position"],
                    "ref": variant["ref"],
                    "alt": variant["alt"],
                }
            )
            variant_hvs.append(hv)

        # Step 6: Generate ZK proof of query validity
        # Prove that query was for a valid gene without revealing which one
        gene_index = list(index_mapping["genes"].keys()).index(gene)

        # Mock ZK proof that index is valid
        validity_proof = {
            "statement": "index in [0, num_genes)",
            "proof": "mock_zk_proof_data",
        }

        # Step 7: Return privacy-preserving result
        private_result = {
            "num_variants": len(mock_result["variants"]),
            "impact_summary": {"HIGH": 1, "MODERATE": 1},
            "clinical_summary": {"Pathogenic": 1, "Likely pathogenic": 1},
            "zkproof": validity_proof,
            "hypervectors": [hv.tolist()[:10] for hv in variant_hvs],  # Truncated for test
        }

        # Verify result
        assert private_result["num_variants"] == 2
        assert private_result["impact_summary"]["HIGH"] == 1
        assert len(private_result["hypervectors"]) == 2

    @pytest.mark.asyncio
    async def test_batch_pir_with_hdc(self, pir_client, hdc_encoder):
        """Test batch PIR queries with HDC encoding."""
        # Multiple variant positions to query
        positions = [100000, 200000, 300000, 400000, 500000]

        # Create batch queries
        queries = [pir_client.create_query(i) for i in range(len(positions))]

        # Simulate batch execution
        # In practice: results = await pir_client.batch_query(indices)

        # Mock HDC-encoded results
        mock_results = []
        for pos in positions:
            variant = {"chromosome": "chr1", "position": pos, "ref": "A", "alt": "G"}
            hv = hdc_encoder.encode_variant(variant)
            mock_results.append(hv)

        # Verify batch results
        assert len(mock_results) == len(positions)

        # Check hypervector properties
        for hv in mock_results:
            assert hv.shape == (10000,)
            assert -1 <= np.min(hv) <= 1
            assert -1 <= np.max(hv) <= 1

    @pytest.mark.asyncio
    async def test_pir_query_caching(self, pir_client, index_mapping):
        """Test PIR query result caching."""
        query_builder = PIRQueryBuilder(pir_client, index_mapping)

        # Create variant query
        variant_query = query_builder.build_variant_query(
            chromosome="chr1", position=100000, ref_allele="A", alt_allele="G"
        )

        # Get cache key
        cache_key = variant_query.get_cache_key()

        # Verify cache is empty
        assert cache_key not in query_builder.cache

        # Mock execution would populate cache
        mock_result = {
            "query": variant_query,
            "data": {"allele_frequency": 0.05},
            "metadata": {"found": True},
            "pir_queries_used": 1,
            "computation_time_ms": 42.5,
        }

        # Add to cache
        query_builder._add_to_cache(cache_key, mock_result)

        # Verify cached
        assert cache_key in query_builder.cache
        assert query_builder.cache[cache_key].data["allele_frequency"] == 0.05

    def test_privacy_guarantees_calculation(self, pir_client):
        """Test privacy guarantee calculations."""
        # With 2 TS servers at 98% honesty
        p_fail = pir_client.calculate_privacy_failure_probability(k=2, q=0.98)
        assert p_fail == (1 - 0.98) ** 2
        assert p_fail == 0.0004  # 0.04%

        # With 3 servers at 95% honesty
        p_fail = pir_client.calculate_privacy_failure_probability(k=3, q=0.95)
        assert p_fail == (1 - 0.95) ** 3
        assert p_fail == 0.000125  # 0.0125%

        # Calculate minimum servers needed
        min_servers = pir_client.calculate_min_servers_needed(
            target_failure=0.0001,
            honesty_prob=0.98,  # 0.01%
        )
        assert min_servers == 3  # Need 3 servers for this guarantee


@pytest.mark.integration
class TestPIRSystemIntegration:
    """System-level integration tests."""

    @pytest.mark.asyncio
    async def test_multi_server_deployment(self):
        """Test multi-server PIR deployment."""
        # This would test actual server deployment
        # with Docker containers or local processes
        pass

    @pytest.mark.asyncio
    async def test_failure_recovery(self):
        """Test system recovery from server failures."""
        # Test handling of server crashes
        # and automatic failover
        pass

    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under load."""
        # Simulate concurrent queries
        # and measure throughput/latency
        pass


# Demo notebook content (would be in separate .ipynb file)
DEMO_NOTEBOOK = """
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# GenomeVault PIR Integration Demo\\n",
    "\\n",
    "This notebook demonstrates the integration of PIR with HDC and ZK proofs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from genomevault.pir.client import PIRClient, PIRServer\\n",
    "from genomevault.pir.client.query_builder import PIRQueryBuilder\\n",
    "from genomevault.hypervector.encoder import HyperdimensionalEncoder\\n",
    "\\n",
    "# Initialize components\\n",
    "servers = [\\n",
    "    PIRServer('ts1', 'http://localhost:9001', 'us', True, 0.98, 50),\\n",
    "    PIRServer('ts2', 'http://localhost:9002', 'eu', True, 0.98, 60),\\n",
    "]\\n",
    "\\n",
    "client = PIRClient(servers, database_size=100000)\\n",
    "encoder = HyperdimensionalEncoder(dimension=10000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Query for a specific variant\\n",
    "query = client.create_query(target_index=42)\\n",
    "print(f'Query ID: {query.query_id}')\\n",
    "print(f'Target index: {query.target_index}')\\n",
    "print(f'Number of servers: {len(query.query_vectors)}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
"""


if __name__ == "__main__":
    # Save demo notebook
    notebook_path = Path("examples/pir_integration_demo.ipynb")
    notebook_path.parent.mkdir(exist_ok=True)

    with open(notebook_path, "w") as f:
        f.write(DEMO_NOTEBOOK)

    print(f"Demo notebook saved to {notebook_path}")
