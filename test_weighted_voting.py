#!/usr/bin/env python3
"""
Test script for dual-axis weighted voting consensus.

Validates:
1. Node weight calculations (w = c + s)
2. BFT safety conditions (H > 2F/3)
3. HIPAA fast-track verification
4. Credit rewards and slashing mechanisms
5. Consensus under various network conditions
"""

import random
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np

from genomevault.blockchain.consensus.weighted_voting import (
    DualAxisNode,
    ResourceClass,
    SignatoryStatus,
    HIPAAVerifier,
    BFTConsensus,
    simulate_network
)


def test_node_weights():
    """Test that node weights are calculated correctly."""
    print("\n" + "="*60)
    print("Testing Node Weight Calculations")
    print("="*60)
    
    test_cases = [
        ("Light Non-Signer", ResourceClass.LIGHT, SignatoryStatus.NON_SIGNER, 1),
        ("Light Trusted", ResourceClass.LIGHT, SignatoryStatus.TRUSTED_SIGNATORY, 11),
        ("Full Non-Signer", ResourceClass.FULL, SignatoryStatus.NON_SIGNER, 4),
        ("Full Trusted", ResourceClass.FULL, SignatoryStatus.TRUSTED_SIGNATORY, 14),
        ("Archive Non-Signer", ResourceClass.ARCHIVE, SignatoryStatus.NON_SIGNER, 8),
        ("Archive Trusted", ResourceClass.ARCHIVE, SignatoryStatus.TRUSTED_SIGNATORY, 18),
    ]
    
    print("\nNode Type            | c | s  | w = c + s | Expected | ✓/✗")
    print("-" * 65)
    
    all_passed = True
    for name, resource, signatory, expected in test_cases:
        node = DualAxisNode(
            node_id=name.replace(" ", "_"),
            resource_class=resource,
            signatory_status=signatory
        )
        
        c, s = node.weight_components
        w = node.voting_weight
        passed = w == expected
        all_passed &= passed
        
        print(f"{name:20} | {c} | {s:2} | {w:9} | {expected:8} | {'✓' if passed else '✗'}")
    
    print(f"\n{'✅' if all_passed else '❌'} Weight calculation test {'passed' if all_passed else 'failed'}")
    return all_passed


def test_hipaa_verification():
    """Test HIPAA fast-track verification."""
    print("\n" + "="*60)
    print("Testing HIPAA Fast-Track Verification")
    print("="*60)
    
    verifier = HIPAAVerifier()
    
    # Test valid NPI
    print("\n1. Testing valid NPI verification:")
    node1 = DualAxisNode(
        node_id="hospital_node",
        resource_class=ResourceClass.FULL,
        npi_number="1234567890"  # Valid NPI in mock registry
    )
    
    print(f"   Before: weight={node1.voting_weight}, q={node1.honesty_probability}")
    success = verifier.fast_track_verification(node1)
    print(f"   After:  weight={node1.voting_weight}, q={node1.honesty_probability}")
    print(f"   Status: {'✅ Verified' if success else '❌ Failed'}")
    
    assert success, "Valid NPI should verify"
    assert node1.signatory_status == SignatoryStatus.TRUSTED_SIGNATORY
    assert node1.voting_weight == 14  # Full (4) + TS (10)
    assert node1.honesty_probability == 0.98
    
    # Test invalid NPI
    print("\n2. Testing invalid NPI:")
    node2 = DualAxisNode(
        node_id="fake_hospital",
        resource_class=ResourceClass.FULL,
        npi_number="9999999999"  # Invalid NPI
    )
    
    success = verifier.fast_track_verification(node2)
    print(f"   Status: {'✅ Verified' if success else '❌ Correctly rejected'}")
    
    assert not success, "Invalid NPI should not verify"
    assert node2.signatory_status == SignatoryStatus.NON_SIGNER
    assert node2.voting_weight == 4  # Full (4) + NS (0)
    
    # Test node without NPI
    print("\n3. Testing node without NPI:")
    node3 = DualAxisNode(
        node_id="regular_node",
        resource_class=ResourceClass.LIGHT
    )
    
    success = verifier.fast_track_verification(node3)
    print(f"   Status: {'✅ Verified' if success else '❌ Correctly rejected (no NPI)'}")
    
    assert not success, "Node without NPI should not verify"
    
    print("\n✅ HIPAA verification working correctly")
    return True


def test_bft_safety():
    """Test BFT safety conditions."""
    print("\n" + "="*60)
    print("Testing BFT Safety Conditions (H > 2F/3)")
    print("="*60)
    
    # Test different Byzantine ratios
    test_scenarios = [
        (0.20, True, "20% Byzantine (safe)"),
        (0.30, True, "30% Byzantine (safe)"),
        (0.40, False, "40% Byzantine (unsafe)"),
        (0.50, False, "50% Byzantine (unsafe)"),
    ]
    
    print("\nByzantine % | Total W | Honest H | Min Required | Safe? | Result")
    print("-" * 70)
    
    all_correct = True
    for byz_ratio, expected_safe, description in test_scenarios:
        # Create network
        nodes = simulate_network(num_nodes=10)
        consensus = BFTConsensus(nodes, byzantine_ratio=byz_ratio)
        
        is_safe, explanation = consensus.verify_safety()
        min_required = consensus.calculate_minimum_honest_weight()
        
        correct = is_safe == expected_safe
        all_correct &= correct
        
        print(f"{byz_ratio*100:10.0f}% | {consensus.total_weight:7} | "
              f"{consensus.honest_weight:8} | {min_required:12} | "
              f"{'Yes' if is_safe else 'No':5} | {'✓' if correct else '✗'}")
    
    print(f"\n{'✅' if all_correct else '❌'} Safety verification {'passed' if all_correct else 'failed'}")
    return all_correct


def test_credit_rewards():
    """Test credit reward mechanism."""
    print("\n" + "="*60)
    print("Testing Credit Rewards (c + (s>0)×2)")
    print("="*60)
    
    test_nodes = [
        DualAxisNode("light_ns", ResourceClass.LIGHT, SignatoryStatus.NON_SIGNER),
        DualAxisNode("light_ts", ResourceClass.LIGHT, SignatoryStatus.TRUSTED_SIGNATORY),
        DualAxisNode("full_ns", ResourceClass.FULL, SignatoryStatus.NON_SIGNER),
        DualAxisNode("full_ts", ResourceClass.FULL, SignatoryStatus.TRUSTED_SIGNATORY),
        DualAxisNode("archive_ns", ResourceClass.ARCHIVE, SignatoryStatus.NON_SIGNER),
        DualAxisNode("archive_ts", ResourceClass.ARCHIVE, SignatoryStatus.TRUSTED_SIGNATORY),
    ]
    
    print("\nNode Type    | c | s>0 | Expected Reward | Actual | ✓/✗")
    print("-" * 60)
    
    all_correct = True
    for node in test_nodes:
        initial_credits = node.credits
        
        # Calculate expected reward: c + (s>0)×2
        c = node.resource_class.value
        signatory_bonus = 2 if node.signatory_status > 0 else 0
        expected = c + signatory_bonus
        
        # Award credits
        actual = node.award_credits(base_reward=1.0)
        
        correct = abs(actual - expected) < 0.01
        all_correct &= correct
        
        node_type = f"{node.resource_class.name}_{('TS' if node.signatory_status > 0 else 'NS')}"
        print(f"{node_type:12} | {c} | {'Yes' if signatory_bonus else 'No':3} | "
              f"{expected:15} | {actual:6.1f} | {'✓' if correct else '✗'}")
    
    print(f"\n{'✅' if all_correct else '❌'} Credit rewards {'correct' if all_correct else 'incorrect'}")
    return all_correct


def test_slashing():
    """Test stake slashing mechanism."""
    print("\n" + "="*60)
    print("Testing Slashing Mechanism (25% on audit failure)")
    print("="*60)
    
    node = DualAxisNode(
        node_id="test_node",
        resource_class=ResourceClass.FULL,
        stake=1000.0
    )
    
    print(f"\nInitial stake: {node.stake:.0f}")
    print(f"Node active: {node.is_active}")
    
    # Test multiple slashing events
    slashing_events = [
        (0.25, 750.0),   # First slash: 25% of 1000 = 250, remaining = 750
        (0.25, 562.5),   # Second slash: 25% of 750 = 187.5, remaining = 562.5
        (0.25, 421.875), # Third slash: 25% of 562.5 = 140.625, remaining = 421.875
    ]
    
    print("\nSlash # | Percentage | Expected | Actual | Active | ✓/✗")
    print("-" * 60)
    
    all_correct = True
    for i, (percentage, expected) in enumerate(slashing_events, 1):
        node.slash_stake(percentage)
        
        correct = abs(node.stake - expected) < 0.01
        all_correct &= correct
        
        print(f"{i:7} | {percentage*100:10.0f}% | {expected:8.2f} | "
              f"{node.stake:6.2f} | {'Yes' if node.is_active else 'No':6} | "
              f"{'✓' if correct else '✗'}")
    
    # Test deactivation on low stake
    print("\n4. Testing deactivation on low stake...")
    node.stake = 12.0  # Set to low value
    node.slash_stake(0.25)  # 12 * 0.75 = 9.0, which is below 10
    print(f"   Stake after slash: {node.stake:.2f}")
    print(f"   Node active: {node.is_active} (should be False)")
    
    assert not node.is_active, "Node should be deactivated with stake <= 10"
    
    print(f"\n{'✅' if all_correct else '❌'} Slashing mechanism {'working' if all_correct else 'broken'}")
    return all_correct


def test_consensus_scenarios():
    """Test consensus under different scenarios."""
    print("\n" + "="*60)
    print("Testing Consensus Scenarios")
    print("="*60)
    
    scenarios = [
        {
            "name": "Honest Supermajority",
            "num_nodes": 20,
            "byzantine_fraction": 0.15,
            "expected_consensus": True
        },
        {
            "name": "Byzantine Below Threshold",
            "num_nodes": 20,
            "byzantine_fraction": 0.25,
            "expected_consensus": True  # Should work below 1/3
        },
        {
            "name": "Too Many Byzantine",
            "num_nodes": 20,
            "byzantine_fraction": 0.40,
            "expected_consensus": False
        },
    ]
    
    print("\nScenario              | Nodes | Byz % | Consensus? | Expected | ✓/✗")
    print("-" * 70)
    
    all_correct = True
    for scenario in scenarios:
        # Create network
        nodes = simulate_network(num_nodes=scenario["num_nodes"])
        consensus = BFTConsensus(nodes, byzantine_ratio=scenario["byzantine_fraction"])
        
        # Simulate Byzantine nodes
        byzantine_nodes = consensus.simulate_byzantine_nodes(scenario["byzantine_fraction"])
        
        # Create blocks
        honest_block = "honest_block_hash"
        byzantine_block = "byzantine_block_hash"
        
        # Vote
        for node_id in consensus.nodes:
            if node_id in byzantine_nodes:
                consensus.submit_vote(node_id, byzantine_block)
            else:
                consensus.submit_vote(node_id, honest_block)
        
        # Check consensus
        result = consensus.check_consensus()
        achieved = result is not None
        
        # For honest supermajority, should achieve consensus on honest block
        if achieved and scenario["byzantine_fraction"] < 0.33:
            correct = result == honest_block and scenario["expected_consensus"]
        else:
            correct = achieved == scenario["expected_consensus"]
        
        all_correct &= correct
        
        print(f"{scenario['name']:20} | {scenario['num_nodes']:5} | "
              f"{scenario['byzantine_fraction']*100:5.0f} | "
              f"{'Yes' if achieved else 'No':10} | "
              f"{'Yes' if scenario['expected_consensus'] else 'No':8} | "
              f"{'✓' if correct else '✗'}")
    
    print(f"\n{'✅' if all_correct else '❌'} Consensus scenarios {'passed' if all_correct else 'failed'}")
    return all_correct


def analyze_weight_distribution():
    """Analyze weight distribution in a network."""
    print("\n" + "="*60)
    print("Analyzing Weight Distribution")
    print("="*60)
    
    # Create diverse network
    nodes = simulate_network(
        num_nodes=100,
        hipaa_fraction=0.2,
        resource_distribution={
            ResourceClass.LIGHT: 0.6,
            ResourceClass.FULL: 0.3,
            ResourceClass.ARCHIVE: 0.1,
        }
    )
    
    consensus = BFTConsensus(nodes)
    distribution = consensus.get_weight_distribution()
    
    # Calculate statistics
    total_weight = sum(distribution.values())
    hipaa_weight = sum(w for k, w in distribution.items() if "TS" in k)
    non_hipaa_weight = sum(w for k, w in distribution.items() if "NS" in k)
    
    print(f"\nNetwork of {len(nodes)} nodes:")
    print(f"Total weight: {total_weight}")
    print(f"HIPAA weight: {hipaa_weight} ({hipaa_weight/total_weight*100:.1f}%)")
    print(f"Non-HIPAA weight: {non_hipaa_weight} ({non_hipaa_weight/total_weight*100:.1f}%)")
    
    print("\nWeight by node type:")
    for node_type in ["Light", "Full", "Archive"]:
        for sig_type in ["NS", "TS"]:
            key = f"{node_type.upper()}_{sig_type}"
            weight = distribution.get(key, 0)
            if weight > 0:
                percentage = weight / total_weight * 100
                print(f"  {key:12}: {weight:4} ({percentage:5.1f}%)")
    
    # Voting power concentration
    node_weights = [n.voting_weight for n in nodes if n.is_active]
    top_10_percent = sorted(node_weights, reverse=True)[:len(node_weights)//10]
    top_10_weight = sum(top_10_percent)
    
    print(f"\nVoting power concentration:")
    print(f"  Top 10% of nodes control {top_10_weight/total_weight*100:.1f}% of weight")
    
    # Could any single entity control consensus?
    max_weight = max(node_weights) if node_weights else 0
    print(f"  Maximum single node weight: {max_weight} ({max_weight/total_weight*100:.1f}%)")
    print(f"  Consensus threshold: {consensus.threshold} ({consensus.threshold/total_weight*100:.1f}%)")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("  GENOMEVAULT WEIGHTED VOTING CONSENSUS TEST SUITE")
    print("="*70)
    
    tests = [
        ("Node Weights", test_node_weights),
        ("HIPAA Verification", test_hipaa_verification),
        ("BFT Safety", test_bft_safety),
        ("Credit Rewards", test_credit_rewards),
        ("Slashing", test_slashing),
        ("Consensus Scenarios", test_consensus_scenarios),
        ("Weight Distribution", analyze_weight_distribution),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:20}: {status}")
        all_passed &= passed
    
    if all_passed:
        print("\n🎉 All tests passed! Weighted voting consensus working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
    
    print("="*70)


if __name__ == "__main__":
    main()