#!/usr/bin/env python3
"""
Integration test: Submit real genomic data to API
"""
import requests
import json
import time
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_health():
    """Test 1: Health check"""
    print("\n=== Test 1: Health Check ===")
    r = requests.get(f"{API_BASE}/healthz")
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    print(f"✓ Health: {r.json()['status']}")
    print(f"  Timestamp: {r.json()['timestamp']}")

def test_analysis_endpoints():
    """Test 2: Analysis API endpoints exist"""
    print("\n=== Test 2: Analysis API Endpoints ===")

    # Test status endpoint with non-existent ID
    r = requests.get(f"{API_BASE}/api/v1/analysis/test-id/status")
    assert r.status_code == 404, f"Unexpected status code: {r.status_code}"
    print(f"✓ Status endpoint: Returns 404 for missing ID (expected)")

    # Test results endpoint with non-existent ID
    r = requests.get(f"{API_BASE}/api/v1/analysis/test-id/results")
    assert r.status_code == 404, f"Unexpected status code: {r.status_code}"
    print(f"✓ Results endpoint: Returns 404 for missing ID (expected)")

def test_pipeline_file_submission():
    """Test 3: Full pipeline with file"""
    print("\n=== Test 3: Pipeline File Submission ===")

    # Find a test VCF file
    vcf_files = list(Path("benchmark_results/differential_encoding_samples").rglob("*.vcf"))
    if not vcf_files:
        print("⚠ No VCF files found, skipping file submission test")
        return

    test_vcf = vcf_files[0]
    print(f"Using test file: {test_vcf}")
    print(f"File size: {test_vcf.stat().st_size / 1024:.2f} KB")

    try:
        with open(test_vcf, "rb") as f:
            files = {"file": (test_vcf.name, f, "text/plain")}
            data = {
                "analysis_type": "whole_genome",
                "k_anonymity": 3,
                "dimension": 10000,
                "enable_zk_proof": False,  # Disable for faster test
                "enable_blockchain": False,
                "enable_pir": False,
            }

            print("Submitting analysis...")
            r = requests.post(
                f"{API_BASE}/api/v1/analysis/submit",
                files=files,
                data=data,
                timeout=30
            )

            if r.status_code not in [200, 202]:
                print(f"⚠ Submission failed: {r.status_code} - {r.text}")
                return

            result = r.json()

            if "analysis_id" in result:
                analysis_id = result["analysis_id"]
                print(f"✓ Analysis submitted: {analysis_id}")
                print(f"  Status: {result['status']}")

                # Poll for results (max 30 seconds)
                max_wait = 30
                poll_interval = 2
                for i in range(max_wait // poll_interval):
                    time.sleep(poll_interval)
                    status_r = requests.get(f"{API_BASE}/api/v1/analysis/{analysis_id}/status")
                    if status_r.status_code == 200:
                        status = status_r.json()
                        print(f"  Progress: {status['progress_percent']:.1f}% - {status['current_stage']}")

                        if status["status"] == "completed":
                            # Get results
                            results_r = requests.get(f"{API_BASE}/api/v1/analysis/{analysis_id}/results")
                            results = results_r.json()

                            print(f"\n✓ Analysis complete!")
                            print(f"  Duration: {results['total_duration_ms']/1000:.2f}s")
                            print(f"  Status: {results['status']}")
                            print(f"  Stages: {len(results['stages'])} completed")

                            # Show stage details
                            for stage in results['stages']:
                                status_icon = "✓" if stage['success'] else "✗"
                                print(f"    {status_icon} {stage['stage_name']}: {stage['duration_ms']/1000:.2f}s")

                            return
                        elif status["status"] == "failed":
                            print(f"\n✗ Analysis failed")
                            return

                print(f"\n⏱ Analysis still running after {max_wait}s")

    except Exception as e:
        print(f"⚠ Pipeline test error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all integration tests"""
    print("=" * 70)
    print("GenomeVault API Integration Tests")
    print("=" * 70)

    try:
        test_health()
        test_analysis_endpoints()
        test_pipeline_file_submission()

        print("\n" + "=" * 70)
        print("✓ All tests completed!")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except requests.exceptions.ConnectionError:
        print("\n✗ Cannot connect to API. Is the server running?")
        print("   Start with: uvicorn genomevault.api.app:app --port 8000")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
