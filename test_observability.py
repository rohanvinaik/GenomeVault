#!/usr/bin/env python3
"""
Test script for GenomeVault observability features.

This script tests metrics collection, structured logging, and tracing
to ensure the observability system is working correctly.
"""

import asyncio
import time
import random
import sys
from pathlib import Path

# Add the genomevault package to path
sys.path.insert(0, str(Path(__file__).parent))

from genomevault.observability.metrics.prometheus import (
    get_metrics_collector,
    get_prometheus_metrics,
)
from genomevault.observability.logging.structured import (
    get_structured_logger,
    set_request_context,
    generate_request_id,
)
from genomevault.observability.tracing.opentelemetry import get_tracing_manager


def test_prometheus_metrics():
    """Test Prometheus metrics collection."""
    print("🧪 Testing Prometheus metrics collection...")

    collector = get_metrics_collector()

    # Test HTTP request metrics
    print("  📊 Recording HTTP request metrics...")
    collector.record_http_request(
        method="GET",
        endpoint="/api/health",
        status_code=200,
        duration=0.045,
        request_size=0,
        response_size=256,
        component="api",
    )

    # Test HDC encoding metrics
    print("  🧬 Recording HDC encoding metrics...")
    collector.record_hdc_encoding(
        dimension=10000, variant_count=150, duration=2.3, status="success"
    )

    # Test ZK proof metrics
    print("  🔒 Recording ZK proof metrics...")
    collector.record_zk_proof_generation(
        circuit_type="variant", duration=5.7, fallback_used=False, status="success"
    )

    # Test PIR query metrics
    print("  🔍 Recording PIR query metrics...")
    collector.record_pir_query(database_size=50000, servers_used=3, duration=1.2, status="success")

    # Test database metrics
    print("  💾 Recording database metrics...")
    collector.record_database_query(
        query_type="SELECT", table="variants", duration=0.025, status="success"
    )

    # Update gauge metrics
    print("  📈 Updating gauge metrics...")
    collector.update_active_encodings(5)
    collector.update_active_databases(2)
    collector.update_active_users(12)
    collector.update_cache_hit_ratio(0.85)

    # Generate metrics output
    print("  📄 Generating metrics output...")
    metrics_output = get_prometheus_metrics()

    # Check if metrics are present
    expected_metrics = [
        "genomevault_http_requests_total",
        "genomevault_hdc_encodings_total",
        "genomevault_zk_proofs_total",
        "genomevault_pir_queries_total",
        "genomevault_database_queries_total",
        "genomevault_hdc_active_encodings",
        "genomevault_cache_hit_ratio",
    ]

    success_count = 0
    for metric in expected_metrics:
        if metric in metrics_output:
            success_count += 1
            print(f"    ✅ Found metric: {metric}")
        else:
            print(f"    ❌ Missing metric: {metric}")

    print(f"  📊 Metrics test: {success_count}/{len(expected_metrics)} metrics found")
    print(f"  📏 Total metrics output size: {len(metrics_output)} characters")

    return success_count == len(expected_metrics)


def test_structured_logging():
    """Test structured logging with correlation IDs."""
    print("\n🧪 Testing structured logging...")

    logger = get_structured_logger("test.observability")

    # Test basic logging with context
    print("  📝 Testing basic logging with context...")
    request_id = generate_request_id()
    set_request_context(request_id, "test-user-123")

    logger.info("Test info message", test_field="test_value", numeric_field=42, duration=1.23)

    logger.warning("Test warning message", warning_type="test_warning", component="test")

    # Test specialized logging methods
    print("  🧬 Testing HDC operation logging...")
    logger.log_hdc_operation(
        operation="encoding",
        dimension=8192,
        duration=1.5,
        variant_count=100,
        compression_ratio=50.2,
    )

    print("  🔒 Testing ZK operation logging...")
    logger.log_zk_operation(
        operation="proof_generation", circuit_type="variant", duration=3.8, fallback_used=False
    )

    print("  🔍 Testing PIR operation logging...")
    logger.log_pir_operation(operation="query", database_size=25000, duration=0.8, servers_used=3)

    print("  🌐 Testing API request logging...")
    logger.log_api_request(
        method="POST",
        path="/api/hdc/encode",
        status_code=201,
        duration=2.1,
        request_size=1024,
        response_size=512,
    )

    # Test context manager
    print("  🎭 Testing logging context manager...")
    with logger.context(
        request_id=generate_request_id(),
        user_id="context-user-456",
        operation="test_context",
        privacy_level="high",
    ):
        logger.info("Message within context", context_test=True)

    print("  ✅ Structured logging test completed")
    return True


async def test_opentelemetry_tracing():
    """Test OpenTelemetry tracing."""
    print("\n🧪 Testing OpenTelemetry tracing...")

    tracing_manager = get_tracing_manager()

    if not tracing_manager:
        print("  ⚠️  OpenTelemetry not available, skipping tracing test")
        return True

    print("  🔄 Testing basic tracing operation...")
    with tracing_manager.trace_operation(
        "test_operation", attributes={"test.attribute": "test_value", "test.number": 42}
    ) as span:
        if span:
            print("    📊 Adding custom attributes to span...")
            tracing_manager.add_genomic_context(
                variant_type="SNP", chromosome="chr1", position=12345, sample_id="sample_123"
            )

        # Simulate some work
        await asyncio.sleep(0.1)
        print("    ✅ Basic tracing operation completed")

    print("  🧬 Testing HDC encoding tracing...")

    @tracing_manager.trace_hdc_encoding(dimension=10000, variant_count=50)
    def simulate_hdc_encoding():
        time.sleep(0.05)  # Simulate encoding work
        return "encoded_result"

    result = simulate_hdc_encoding()
    print(f"    ✅ HDC encoding traced, result: {result}")

    print("  🔒 Testing ZK proof tracing...")

    @tracing_manager.trace_zk_proof(circuit_type="variant", proof_id="test_proof_123")
    def simulate_zk_proof():
        time.sleep(0.08)  # Simulate proof generation
        return "proof_result"

    proof_result = simulate_zk_proof()
    print(f"    ✅ ZK proof traced, result: {proof_result}")

    print("  🔍 Testing PIR query tracing...")

    @tracing_manager.trace_pir_query(database_size=1000, query_index=42)
    def simulate_pir_query():
        time.sleep(0.03)  # Simulate PIR query
        return "pir_result"

    pir_result = simulate_pir_query()
    print(f"    ✅ PIR query traced, result: {pir_result}")

    print("  ✅ OpenTelemetry tracing test completed")
    return True


def test_integration():
    """Test integration between metrics, logging, and tracing."""
    print("\n🧪 Testing observability integration...")

    logger = get_structured_logger("integration.test")
    collector = get_metrics_collector()
    tracing_manager = get_tracing_manager()

    # Simulate a complete genomic operation
    request_id = generate_request_id()
    set_request_context(request_id, "integration-user")

    logger.info("Starting integrated genomic operation", operation_type="variant_analysis")

    # Simulate operation with all observability features
    start_time = time.time()

    if tracing_manager:
        with tracing_manager.trace_operation(
            "genomic_variant_analysis",
            attributes={
                "genomic.operation": "variant_analysis",
                "genomic.sample_count": 100,
                "privacy.level": "high",
            },
        ):
            # Simulate HDC encoding
            logger.info("Encoding variants to hypervectors")
            time.sleep(random.uniform(0.1, 0.3))

            collector.record_hdc_encoding(
                dimension=8192, variant_count=100, duration=0.2, status="success"
            )

            # Simulate ZK proof generation
            logger.info("Generating zero-knowledge proof")
            time.sleep(random.uniform(0.2, 0.4))

            collector.record_zk_proof_generation(
                circuit_type="variant", duration=0.3, status="success"
            )

            # Simulate PIR query
            logger.info("Executing private information retrieval")
            time.sleep(random.uniform(0.05, 0.15))

            collector.record_pir_query(
                database_size=10000, servers_used=3, duration=0.1, status="success"
            )
    else:
        # Simulate without tracing
        time.sleep(0.6)
        collector.record_hdc_encoding(8192, 100, 0.2, "success")
        collector.record_zk_proof_generation("variant", 0.3, status="success")
        collector.record_pir_query(10000, 3, 0.1, "success")

    total_duration = time.time() - start_time

    # Record overall API metrics
    collector.record_http_request(
        method="POST",
        endpoint="/api/genomic/analyze",
        status_code=200,
        duration=total_duration,
        request_size=2048,
        response_size=1024,
    )

    logger.info(
        "Completed integrated genomic operation",
        total_duration=total_duration,
        variants_processed=100,
        privacy_preserved=True,
    )

    print(f"  ✅ Integration test completed in {total_duration:.3f}s")
    return True


async def main():
    """Run all observability tests."""
    print("🚀 Starting GenomeVault Observability Tests\n")

    test_results = []

    # Run individual tests
    test_results.append(("Prometheus Metrics", test_prometheus_metrics()))
    test_results.append(("Structured Logging", test_structured_logging()))
    test_results.append(("OpenTelemetry Tracing", await test_opentelemetry_tracing()))
    test_results.append(("Integration Test", test_integration()))

    # Print results summary
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All observability tests passed!")
        print("\n📊 Metrics endpoint: http://localhost:8000/metrics")
        print("📈 To view metrics in Prometheus: http://localhost:9090")
        print("📊 To view dashboards in Grafana: http://localhost:3000")
        print("🔍 To view traces in Jaeger: http://localhost:16686")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
