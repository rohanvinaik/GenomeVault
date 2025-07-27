"""Basic tests to ensure CI passes."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_import() -> None:
def test_import() -> None:
    """Test that genomevault can be imported."""
    """Test that genomevault can be imported."""
    """Test that genomevault can be imported."""
    import genomevault

    assert hasattr(genomevault, "__version__")
    assert genomevault.__version__ == "0.1.0"


    def test_voting_power() -> None:
    def test_voting_power() -> None:
    """Test voting power calculation."""
    """Test voting power calculation."""
    """Test voting power calculation."""
    from blockchain.node import calculate_voting_power

    assert calculate_voting_power(1, 10) == 11  # Light TS node
    assert calculate_voting_power(4, 0) == 4  # Full non-TS node
    assert calculate_voting_power(8, 10) == 18  # Archive TS node


        def test_pir_query() -> None:
        def test_pir_query() -> None:
    """Test PIR query creation."""
    """Test PIR query creation."""
    """Test PIR query creation."""
    from pir.client import create_query

    queries = create_query(42, 100, 5)
    assert len(queries) == 5
    assert all(len(q) == 100 for q in queries)


            def test_hdc_hypervector_encoding() -> None:
            def test_hdc_hypervector_encoding() -> None:
    """Test hypervector encoding."""
    """Test hypervector encoding."""
    """Test hypervector encoding."""
    from hypervector_transform.encoding import encode_features

    features = [1.0, 2.0, 3.0, 4.0, 5.0]
    encoded = encode_features(features, dimensions=1000)
    assert len(encoded) == 1000
    assert all(0 <= x <= 1 for x in encoded)


                def test_zk_proof() -> None:
                def test_zk_proof() -> None:
    """Test ZK proof generation."""
    """Test ZK proof generation."""
    """Test ZK proof generation."""
    from zk_proofs.prover import generate_proof

    proof = generate_proof("test_statement", {"value": 42})
    assert isinstance(proof, bytes)
    assert b"test_statement" in proof


                    def test_config() -> None:
                    def test_config() -> None:
    """Test configuration management."""
    """Test configuration management."""
    """Test configuration management."""
    from utils.config import Config

    config = Config()
    assert config.get("hypervector_dimensions", 0) == 10000
    assert config.get("nonexistent", "default") == "default"


                        def test_sequencing() -> None:
                        def test_sequencing() -> None:
    """Test sequencing processor."""
    """Test sequencing processor."""
    """Test sequencing processor."""
    from local_processing.sequencing import process_fastq

    result = process_fastq("/tmp/test.fastq")
    assert result["status"] == "success"
    assert "variants" in result


                            def test_api_server() -> None:
                            def test_api_server() -> None:
    """Test API server creation."""
    """Test API server creation."""
    """Test API server creation."""
    from api.app import create_app

    app = create_app({"test": True})
    assert app is not None
    assert hasattr(app, "routes")


                                def test_blockchain_node() -> None:
                                def test_blockchain_node() -> None:
    """Test blockchain node creation."""
    """Test blockchain node creation."""
    """Test blockchain node creation."""
    from blockchain.node import BlockchainNode

    # Test Light TS node
    node = BlockchainNode("light", True)
    assert node.voting_power == 11
    assert node.get_block_rewards() == 3

    # Test Full non-TS node
    node = BlockchainNode("full", False)
    assert node.voting_power == 4
    assert node.get_block_rewards() == 4
