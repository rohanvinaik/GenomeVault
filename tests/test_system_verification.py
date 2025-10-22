#!/usr/bin/env python3
"""
GenomeVault System Verification Script

Verifies all components of the GenomeVault system:
1. Core imports and modules
2. Reference data availability
3. Pipeline components (Differential Encoding, HDC, ZK, PIR)
4. API server health and endpoints
5. Configuration files
6. Performance targets
"""

import sys
import time
from pathlib import Path
import json

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print('='*70)

def print_check(name, status, details=""):
    icon = "✓" if status else "✗"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{icon}{reset} {name}")
    if details:
        print(f"  {details}")

def verify_imports():
    """Verify all core imports work."""
    print_header("1. Core Imports Verification")

    checks = [
        ("genomevault.differential_encoding", "Differential Encoding"),
        ("genomevault.hypervector_transform", "HDC Transform"),
        ("genomevault.zk", "Zero-Knowledge Proofs"),
        ("genomevault.pir", "Private Information Retrieval"),
        ("genomevault.blockchain", "Blockchain Integration"),
        ("genomevault.compute", "Compute Backends"),
        ("genomevault.api", "API Server"),
    ]

    passed = 0
    for module_name, friendly_name in checks:
        try:
            __import__(module_name)
            print_check(friendly_name, True, f"Module: {module_name}")
            passed += 1
        except Exception as e:
            print_check(friendly_name, False, f"Error: {e}")

    return passed, len(checks)

def verify_reference_data():
    """Verify reference genome data exists."""
    print_header("2. Reference Data Verification")

    checks = [
        Path("benchmark_results/full_pipeline_synthetic/reference/chr22.fa"),
        Path("benchmark_results/differential_encoding_samples/vcf_pool/reference_001.vcf"),
        Path("benchmark_results/differential_encoding_samples/vcf_pool/reference_002.vcf"),
        Path("benchmark_results/differential_encoding_samples/vcf_pool/reference_003.vcf"),
    ]

    passed = 0
    for path in checks:
        exists = path.exists()
        size = f"{path.stat().st_size / (1024*1024):.2f} MB" if exists else "N/A"
        print_check(str(path.name), exists, f"Size: {size}" if exists else "File not found")
        if exists:
            passed += 1

    return passed, len(checks)

def verify_pipeline_components():
    """Verify pipeline components can be initialized."""
    print_header("3. Pipeline Components Verification")

    results = []

    # Test Differential Encoding
    try:
        from genomevault.differential_encoding import SecureReferenceGenomeManager
        ref_manager = SecureReferenceGenomeManager(
            Path("benchmark_results/differential_encoding_samples/vcf_pool")
        )
        has_refs = ref_manager.reference_count > 0
        print_check("Differential Encoding", has_refs,
                   f"Loaded {ref_manager.reference_count} reference genomes")
        results.append(has_refs)
    except Exception as e:
        print_check("Differential Encoding", False, f"Error: {e}")
        results.append(False)

    # Test HDC Transform
    try:
        from genomevault.hypervector_transform import create_backend_encoder
        encoder = create_backend_encoder(dimension=1000)
        print_check("HDC Transform", True, f"Backend: {encoder.__class__.__name__}")
        results.append(True)
    except Exception as e:
        print_check("HDC Transform", False, f"Error: {e}")
        results.append(False)

    # Test ZK Proofs
    try:
        from genomevault.zk_proofs import PQEngine, prove, verify
        engine = PQEngine()
        print_check("Zero-Knowledge Proofs", True, "PQEngine initialized")
        results.append(True)
    except Exception as e:
        print_check("Zero-Knowledge Proofs", False, f"Error: {e}")
        results.append(False)

    # Test PIR
    try:
        from genomevault.pir import create_pir_system, PIRProtocol
        import numpy as np
        # Create simple PIR system
        database = np.array([[1, 2, 3], [4, 5, 6]])
        pir_system = create_pir_system(database)
        print_check("Private Information Retrieval", True,
                   f"PIR system initialized")
        results.append(True)
    except Exception as e:
        print_check("Private Information Retrieval", False, f"Error: {e}")
        results.append(False)

    return sum(results), len(results)

def verify_api_server():
    """Verify API server is running and responsive."""
    print_header("4. API Server Verification")

    import requests

    results = []
    base_url = "http://localhost:8000"

    # Test health endpoint
    try:
        r = requests.get(f"{base_url}/healthz", timeout=5)
        success = r.status_code == 200
        print_check("Health Endpoint", success,
                   f"Status: {r.json()['status']}" if success else f"HTTP {r.status_code}")
        results.append(success)
    except Exception as e:
        print_check("Health Endpoint", False, f"Error: {e}")
        results.append(False)

    # Test API docs endpoint
    try:
        r = requests.get(f"{base_url}/api/docs", timeout=5)
        success = r.status_code == 200
        print_check("API Documentation", success, "Swagger UI accessible")
        results.append(success)
    except Exception as e:
        print_check("API Documentation", False, f"Error: {e}")
        results.append(False)

    # Test analysis endpoints structure
    try:
        # These should return 404 for non-existent IDs (expected behavior)
        r = requests.get(f"{base_url}/api/v1/analysis/test-id/status", timeout=5)
        success = r.status_code == 404
        print_check("Analysis Endpoints", success, "Status endpoint responds correctly")
        results.append(success)
    except Exception as e:
        print_check("Analysis Endpoints", False, f"Error: {e}")
        results.append(False)

    return sum(results), len(results)

def verify_configuration():
    """Verify configuration files exist and are valid."""
    print_header("5. Configuration Files Verification")

    results = []

    configs = [
        Path("genomevault/config/blockchain.yaml"),
        Path("genomevault/config/compute.yaml"),
        Path("pyproject.toml"),
        Path("requirements.txt"),
    ]

    for config_path in configs:
        exists = config_path.exists()
        size = f"{config_path.stat().st_size} bytes" if exists else "N/A"
        print_check(str(config_path.name), exists, f"Size: {size}" if exists else "Not found")
        results.append(exists)

    return sum(results), len(results)

def verify_performance_targets():
    """Verify latest pipeline results meet performance targets."""
    print_header("6. Performance Targets Verification")

    # Find latest pipeline results
    results_dir = Path("benchmark_results/full_pipeline_results")
    if not results_dir.exists():
        print_check("Performance Results", False, "Results directory not found")
        return 0, 1

    # Get most recent pipeline run
    result_files = list(results_dir.glob("*/pipeline_results.json"))
    if not result_files:
        print_check("Performance Results", False, "No results files found")
        return 0, 1

    latest_result = max(result_files, key=lambda p: p.stat().st_mtime)

    try:
        with open(latest_result) as f:
            data = json.load(f)

        total_duration = data['summary']['total_duration_ms']
        success_rate = data['summary']['success_rate']

        # Performance targets from CLAUDE.md
        targets = {
            "Total Duration": (total_duration < 5000, f"{total_duration:.0f}ms (target: <5000ms)"),
            "Success Rate": (success_rate == 100.0, f"{success_rate}% (target: 100%)"),
        }

        results = []
        for name, (passed, details) in targets.items():
            print_check(name, passed, details)
            results.append(passed)

        return sum(results), len(results)

    except Exception as e:
        print_check("Performance Results", False, f"Error parsing results: {e}")
        return 0, 1

def main():
    """Run all verification checks."""
    print("\n" + "="*70)
    print("  GenomeVault System Verification")
    print("="*70)

    all_results = []

    # Run all verification checks
    all_results.append(verify_imports())
    all_results.append(verify_reference_data())
    all_results.append(verify_pipeline_components())
    all_results.append(verify_api_server())
    all_results.append(verify_configuration())
    all_results.append(verify_performance_targets())

    # Summary
    print_header("Verification Summary")

    total_passed = sum(p for p, _ in all_results)
    total_checks = sum(t for _, t in all_results)

    print(f"\nTotal Checks: {total_checks}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_checks - total_passed}")
    print(f"Success Rate: {(total_passed/total_checks)*100:.1f}%")

    if total_passed == total_checks:
        print("\n🎉 All verification checks passed!")
        print("\n✅ GenomeVault system is FULLY OPERATIONAL")
        return 0
    else:
        print(f"\n⚠️  {total_checks - total_passed} checks failed")
        print("Review the errors above and address any issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
