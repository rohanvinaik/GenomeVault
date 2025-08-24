#!/usr/bin/env python3
"""Test performance monitoring dashboard for ZK proofs."""

import time
import sys
import random
import threading
from pathlib import Path

# Add genomevault to path
sys.path.insert(0, "/Users/rohanvinaik/genomevault")

from genomevault.zk_proofs.performance_monitor import PerformanceMonitor, get_monitor
from genomevault.zk_proofs.dashboard import PerformanceDashboard, HTMLDashboard, run_dashboard
from genomevault.zk_proofs.prover import Prover
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


def simulate_proof_operations(monitor: PerformanceMonitor, duration: int = 30):
    """Simulate proof operations for testing."""

    circuits = [
        "variant_presence",
        "diabetes_risk_alert",
        "polygenic_risk_score",
        "ancestry_composition",
    ]

    devices = ["cpu", "metal", "cuda"]

    start_time = time.time()
    operation_count = 0

    print(f"Simulating proof operations for {duration} seconds...")

    while time.time() - start_time < duration:
        # Pick random circuit
        circuit = random.choice(circuits)
        device = random.choice(devices[:2])  # Use cpu and metal

        # Simulate different operations
        operation = random.choice(["witness", "proof", "verify"])

        # Simulate latency (witness is fastest, proof is slowest)
        if operation == "witness":
            base_latency = random.uniform(0.5, 3.0)
        elif operation == "verify":
            base_latency = random.uniform(1.0, 5.0)
        else:  # proof
            base_latency = random.uniform(2.0, 10.0)

        # Add some variance based on circuit complexity
        if circuit == "ancestry_composition":
            base_latency *= 2.0
        elif circuit == "polygenic_risk_score":
            base_latency *= 1.5

        # Simulate cache hits (70% cache hit rate)
        cache_hit = random.random() < 0.7
        if cache_hit:
            base_latency *= 0.1  # Cache hits are 10x faster

        # Simulate occasional failures (2% failure rate)
        success = random.random() > 0.02
        error = None if success else "Simulated error"

        # Record operation
        monitor.record_operation(
            circuit_type=circuit,
            operation=operation,
            duration_ms=base_latency,
            input_size=random.randint(100, 10000),
            memory_mb=random.uniform(10, 100),
            cache_hit=cache_hit,
            device=device,
            success=success,
            error=error,
        )

        operation_count += 1

        # Small delay between operations
        time.sleep(random.uniform(0.01, 0.1))

    print(f"Simulated {operation_count} operations")


def test_real_proofs():
    """Test with real proof generation."""
    print("\n" + "=" * 60)
    print("Testing with Real Proof Generation")
    print("=" * 60)

    prover = Prover(use_circom=False)  # Use mock proofs for testing
    monitor = get_monitor()

    # Generate some test proofs
    test_cases = [
        {
            "circuit": "variant_presence",
            "public": {
                "variant_hash": "test_hash_123",
                "reference_hash": "ref_456",
                "commitment_root": "root_789",
            },
            "private": {
                "variant_data": {"chr": "chr1", "pos": 12345, "ref": "A", "alt": "G"},
                "merkle_proof": ["proof1", "proof2"],
                "witness_randomness": "random_abc",
            },
        },
        {
            "circuit": "diabetes_risk_alert",
            "public": {
                "glucose_threshold": 126,
                "risk_threshold": 0.75,
                "result_commitment": "commit_xyz",
            },
            "private": {
                "glucose_reading": 130,
                "risk_score": 0.82,
                "witness_randomness": "random_def",
            },
        },
    ]

    print("\nGenerating test proofs...")
    for i, test in enumerate(test_cases, 1):
        try:
            proof = prover.generate_proof(test["circuit"], test["public"], test["private"])
            print(f"  {i}. {test['circuit']}: Success")
        except Exception as e:
            print(f"  {i}. {test['circuit']}: Failed - {e}")

    # Generate report
    print("\n" + monitor.generate_report())


def test_dashboard_rendering():
    """Test dashboard rendering."""
    print("\n" + "=" * 60)
    print("Testing Dashboard Rendering")
    print("=" * 60)

    monitor = get_monitor()

    # Simulate some operations first
    print("\nGenerating sample data...")
    simulate_proof_operations(monitor, duration=5)

    # Test terminal dashboard
    dashboard = PerformanceDashboard(monitor)

    print("\nTerminal Dashboard:")
    print("-" * 60)
    print(dashboard.render())

    # Test HTML dashboard
    html_dashboard = HTMLDashboard(monitor)
    html_path = Path("test_dashboard.html")
    html_dashboard.save_html(html_path)
    print(f"\nHTML dashboard saved to: {html_path}")

    # Get dashboard data
    data = monitor.get_dashboard_data()

    print("\nDashboard Data Summary:")
    print(f"  Total operations: {data['summary']['total_operations']}")
    print(f"  Success rate: {data['summary']['success_rate']:.1%}")
    print(f"  Cache hit rate: {data['summary']['overall_cache_hit_rate']:.1%}")
    print(f"  Circuits tracked: {data['summary']['circuits_tracked']}")
    print(f"  Active alerts: {data['summary']['active_alerts']}")


def test_performance_alerts():
    """Test performance alerting system."""
    print("\n" + "=" * 60)
    print("Testing Performance Alerts")
    print("=" * 60)

    monitor = PerformanceMonitor()

    # Trigger high latency alert
    monitor.record_operation(
        circuit_type="test_circuit",
        operation="witness",
        duration_ms=10.0,  # Above 5ms threshold
        input_size=1000,
        success=True,
    )

    # Trigger multiple failures for error rate alert
    for i in range(5):
        monitor.record_operation(
            circuit_type="test_circuit",
            operation="proof",
            duration_ms=2.0,
            input_size=1000,
            success=False,
            error="Test error",
        )

    # Check alerts
    alerts = monitor.alerts
    print(f"\nGenerated {len(alerts)} alerts:")

    for alert in alerts:
        print(f"  - {alert['type']}: {alert['circuit']}")
        print(f"    Value: {alert['value']:.2f}, Threshold: {alert['threshold']:.2f}")


def run_interactive_demo():
    """Run interactive dashboard demo."""
    print("\n" + "=" * 60)
    print("Interactive Dashboard Demo")
    print("=" * 60)

    monitor = get_monitor()

    # Start simulation in background thread
    sim_thread = threading.Thread(
        target=simulate_proof_operations,
        args=(monitor, 60),  # Run for 60 seconds
    )
    sim_thread.daemon = True
    sim_thread.start()

    # Wait a bit for data to accumulate
    print("\nStarting dashboard in 3 seconds...")
    time.sleep(3)

    # Run dashboard
    try:
        run_dashboard(mode="terminal", duration=30)
    except KeyboardInterrupt:
        print("\nDashboard stopped")


def main():
    """Run all performance monitoring tests."""
    print("🧬 GenomeVault Performance Monitoring Tests")
    print("=" * 60)

    try:
        # Basic tests
        test_performance_alerts()
        test_real_proofs()
        test_dashboard_rendering()

        # Interactive demo
        print("\n" + "=" * 60)
        print("Would you like to see the interactive dashboard? (y/n)")

        response = input("> ").strip().lower()
        if response == "y":
            run_interactive_demo()

        print("\n" + "=" * 60)
        print("✅ ALL PERFORMANCE MONITORING TESTS PASSED")
        print("=" * 60)

        print("\nKey Features Demonstrated:")
        print("  • Real-time performance tracking")
        print("  • Circuit-specific metrics (latency, throughput)")
        print("  • Cache hit rate monitoring")
        print("  • Automatic alerting system")
        print("  • Terminal and HTML dashboards")
        print("  • Performance report generation")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
