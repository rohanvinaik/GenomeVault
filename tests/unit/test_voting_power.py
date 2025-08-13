import itertools

from genomevault.blockchain.node import BlockchainNode, NodeInfo
from genomevault.core.constants import NODE_CLASS_WEIGHT, NodeType


def reference_vp(node_type, signatory):
    """Reference vp.
    Args:        node_type: Parameter value.        signatory: Parameter value.
    Returns:
        Result of the operation."""
    return NODE_CLASS_WEIGHT[node_type] + (2 if signatory else 0)


def test_nodeinfo_and_blockchainnode_agree():
    """Test nodeinfo and blockchainnode agree.
    Returns:
        Result of the operation."""
    for nt, signatory in itertools.product(list(NodeType), [False, True]):
        bn = BlockchainNode("n1", nt, signatory)
        ni = NodeInfo(
            node_id="n2",
            node_type=nt.value,
            class_weight=NODE_CLASS_WEIGHT[nt],
            signatory=signatory,
        )
        assert bn._calculate_voting_power(nt, signatory) == reference_vp(nt, signatory)
        assert ni.calculate_voting_power() == reference_vp(nt, signatory)
