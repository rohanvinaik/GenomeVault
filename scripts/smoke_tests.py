#!/usr/bin/env python3
"""
Smoke tests for GenomeVault deployment validation.

Runs essential tests to validate deployment health before promoting
to higher environments. Tests core functionality, privacy guarantees,
and basic security measures.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import httpx
import numpy as np
from prometheus_client.parser import text_string_to_metric_families

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result with metadata."""

    name: str
    passed: bool
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class SmokeTestConfig:
    """Smoke test configuration."""

    base_url: str
    timeout: int = 30
    comprehensive: bool = False
    max_retries: int = 3
    retry_delay: int = 5
    parallel_tests: int = 5

    # Authentication
    api_key: Optional[str] = None
    auth_header: Optional[str] = None

    # Environment-specific settings
    environment: str = "dev"
    validate_ssl: bool = True

    # Privacy settings
    validate_privacy: bool = True
    privacy_threshold: float = 0.01  # Maximum allowable privacy breach probability


class SmokeTestRunner:
    """Main smoke test runner."""

    def __init__(self, config: SmokeTestConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=config.timeout, verify=config.validate_ssl, follow_redirects=True
        )

        # Set up authentication
        if config.api_key:
            self.client.headers["Authorization"] = f"Bearer {config.api_key}"
        elif config.auth_header:
            self.client.headers["Authorization"] = config.auth_header

        self.results: List[TestResult] = []

    async def run_all_tests(self) -> bool:
        """Run all smoke tests."""
        logger.info(f"Starting smoke tests for {self.config.environment} environment")
        logger.info(f"Base URL: {self.config.base_url}")

        start_time = time.time()

        # Core functionality tests
        await self._run_basic_health_tests()
        await self._run_api_functionality_tests()

        if self.config.comprehensive:
            await self._run_comprehensive_tests()

        if self.config.validate_privacy:
            await self._run_privacy_tests()

        # Security tests
        await self._run_security_tests()

        # Performance tests
        await self._run_performance_tests()

        total_duration = time.time() - start_time

        # Generate report
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]

        logger.info(f"\n{'='*60}")
        logger.info(f"SMOKE TEST RESULTS - {self.config.environment.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {len(self.results)}")
        logger.info(f"Passed: {len(passed_tests)}")
        logger.info(f"Failed: {len(failed_tests)}")
        logger.info(f"Success Rate: {len(passed_tests)/len(self.results)*100:.1f}%")
        logger.info(f"Total Duration: {total_duration:.2f}s")

        if failed_tests:
            logger.error(f"\nFAILED TESTS:")
            for test in failed_tests:
                logger.error(f"  ❌ {test.name}: {test.error}")

        await self.client.aclose()

        return len(failed_tests) == 0

    async def _run_basic_health_tests(self) -> None:
        """Run basic health and readiness tests."""
        logger.info("Running basic health tests...")

        # Health check
        await self._test_endpoint(
            "health_check",
            "/health",
            expected_status=200,
            check_response=lambda r: r.get("status") == "healthy",
        )

        # Readiness check
        await self._test_endpoint(
            "readiness_check",
            "/health/ready",
            expected_status=200,
            check_response=lambda r: r.get("ready") == True,
        )

        # Liveness check
        await self._test_endpoint(
            "liveness_check",
            "/health/live",
            expected_status=200,
            check_response=lambda r: r.get("alive") == True,
        )

        # Startup check
        await self._test_endpoint("startup_check", "/health/startup", expected_status=200)

    async def _run_api_functionality_tests(self) -> None:
        """Run core API functionality tests."""
        logger.info("Running API functionality tests...")

        # API version check
        await self._test_endpoint(
            "api_version",
            "/api/v1/version",
            expected_status=200,
            check_response=lambda r: "version" in r and "build" in r,
        )

        # System status
        await self._test_endpoint(
            "system_status",
            "/api/v1/status",
            expected_status=200,
            check_response=self._validate_system_status,
        )

        # Hypervector encoding test
        await self._test_hypervector_encoding()

        # PIR query test (if not production)
        if self.config.environment != "prod":
            await self._test_pir_query()

        # ZK proof test
        await self._test_zk_proof()

    async def _test_hypervector_encoding(self) -> None:
        """Test hypervector encoding functionality."""
        test_data = {
            "variants": [
                {"chrom": "1", "pos": 12345, "ref": "A", "alt": "T"},
                {"chrom": "2", "pos": 67890, "ref": "C", "alt": "G"},
            ],
            "dimension": 1000,
        }

        start_time = time.time()
        try:
            response = await self.client.post(
                urljoin(self.config.base_url, "/api/v1/hv/encode"), json=test_data
            )

            duration_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                if "hypervector" in result and len(result["hypervector"]) == 1000:
                    self.results.append(
                        TestResult(
                            "hypervector_encoding",
                            True,
                            duration_ms,
                            {"dimension": len(result["hypervector"])},
                        )
                    )
                else:
                    self.results.append(
                        TestResult(
                            "hypervector_encoding",
                            False,
                            duration_ms,
                            error="Invalid hypervector response",
                        )
                    )
            else:
                self.results.append(
                    TestResult(
                        "hypervector_encoding",
                        False,
                        duration_ms,
                        error=f"HTTP {response.status_code}: {response.text}",
                    )
                )

        except Exception as e:
            self.results.append(
                TestResult(
                    "hypervector_encoding", False, (time.time() - start_time) * 1000, error=str(e)
                )
            )

    async def _test_pir_query(self) -> None:
        """Test PIR query functionality."""
        test_data = {
            "query_type": "test",
            "database_size": 100,
            "query_index": 42,
            "privacy_level": "standard",
        }

        start_time = time.time()
        try:
            response = await self.client.post(
                urljoin(self.config.base_url, "/api/v1/pir/query"), json=test_data
            )

            duration_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                if "result" in result and "privacy_guarantee" in result:
                    privacy_breach_prob = result.get("privacy_guarantee", {}).get(
                        "breach_probability", 1.0
                    )
                    if privacy_breach_prob < self.config.privacy_threshold:
                        self.results.append(
                            TestResult(
                                "pir_query",
                                True,
                                duration_ms,
                                {"privacy_breach_prob": privacy_breach_prob},
                            )
                        )
                    else:
                        self.results.append(
                            TestResult(
                                "pir_query",
                                False,
                                duration_ms,
                                error=f"Privacy breach probability too high: {privacy_breach_prob}",
                            )
                        )
                else:
                    self.results.append(
                        TestResult(
                            "pir_query", False, duration_ms, error="Invalid PIR response format"
                        )
                    )
            else:
                self.results.append(
                    TestResult(
                        "pir_query",
                        False,
                        duration_ms,
                        error=f"HTTP {response.status_code}: {response.text}",
                    )
                )

        except Exception as e:
            self.results.append(
                TestResult("pir_query", False, (time.time() - start_time) * 1000, error=str(e))
            )

    async def _test_zk_proof(self) -> None:
        """Test zero-knowledge proof functionality."""
        test_data = {"proof_type": "range", "public_input": [1, 2, 3], "circuit": "test_circuit"}

        start_time = time.time()
        try:
            # Generate proof
            response = await self.client.post(
                urljoin(self.config.base_url, "/api/v1/zk/prove"), json=test_data
            )

            if response.status_code == 200:
                proof_result = response.json()
                if "proof" in proof_result:
                    # Verify proof
                    verify_data = {
                        "proof": proof_result["proof"],
                        "public_input": test_data["public_input"],
                        "circuit": test_data["circuit"],
                    }

                    verify_response = await self.client.post(
                        urljoin(self.config.base_url, "/api/v1/zk/verify"), json=verify_data
                    )

                    duration_ms = (time.time() - start_time) * 1000

                    if verify_response.status_code == 200:
                        verify_result = verify_response.json()
                        if verify_result.get("valid") == True:
                            self.results.append(
                                TestResult("zk_proof", True, duration_ms, {"proof_verified": True})
                            )
                        else:
                            self.results.append(
                                TestResult(
                                    "zk_proof",
                                    False,
                                    duration_ms,
                                    error="Proof verification failed",
                                )
                            )
                    else:
                        self.results.append(
                            TestResult(
                                "zk_proof",
                                False,
                                duration_ms,
                                error=f"Verification failed: HTTP {verify_response.status_code}",
                            )
                        )
                else:
                    self.results.append(
                        TestResult(
                            "zk_proof",
                            False,
                            (time.time() - start_time) * 1000,
                            error="No proof in response",
                        )
                    )
            else:
                self.results.append(
                    TestResult(
                        "zk_proof",
                        False,
                        (time.time() - start_time) * 1000,
                        error=f"Proof generation failed: HTTP {response.status_code}",
                    )
                )

        except Exception as e:
            self.results.append(
                TestResult("zk_proof", False, (time.time() - start_time) * 1000, error=str(e))
            )

    async def _run_comprehensive_tests(self) -> None:
        """Run comprehensive tests for staging/production."""
        logger.info("Running comprehensive tests...")

        # Load test
        await self._run_load_test()

        # Data integrity test
        await self._test_data_integrity()

        # Failover test
        if self.config.environment == "staging":
            await self._test_failover()

    async def _run_load_test(self) -> None:
        """Run basic load test."""
        logger.info("Running load test with 10 concurrent requests...")

        async def make_request():
            try:
                response = await self.client.get(urljoin(self.config.base_url, "/api/v1/status"))
                return response.status_code == 200
            except Exception:
                return False

        start_time = time.time()

        # Run 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        duration_ms = (time.time() - start_time) * 1000
        success_rate = sum(results) / len(results)

        self.results.append(
            TestResult(
                "load_test",
                success_rate >= 0.9,  # 90% success rate required
                duration_ms,
                {
                    "concurrent_requests": 10,
                    "success_rate": success_rate,
                    "avg_response_time": duration_ms / len(results),
                },
                error=(
                    f"Success rate {success_rate*100:.1f}% below threshold"
                    if success_rate < 0.9
                    else None
                ),
            )
        )

    async def _test_data_integrity(self) -> None:
        """Test data integrity through encoding/decoding cycle."""
        test_variants = [
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "T"},
            {"chrom": "X", "pos": 200, "ref": "G", "alt": "C"},
        ]

        start_time = time.time()
        try:
            # Encode
            encode_response = await self.client.post(
                urljoin(self.config.base_url, "/api/v1/hv/encode"),
                json={"variants": test_variants, "dimension": 1000},
            )

            if encode_response.status_code == 200:
                encoded_data = encode_response.json()

                # Decode (if endpoint exists)
                decode_response = await self.client.post(
                    urljoin(self.config.base_url, "/api/v1/hv/decode"),
                    json={"hypervector": encoded_data["hypervector"]},
                )

                duration_ms = (time.time() - start_time) * 1000

                if decode_response.status_code == 200:
                    decoded_data = decode_response.json()
                    # Basic integrity check (would be more sophisticated)
                    integrity_ok = len(decoded_data.get("variants", [])) == len(test_variants)

                    self.results.append(
                        TestResult(
                            "data_integrity",
                            integrity_ok,
                            duration_ms,
                            {
                                "original_count": len(test_variants),
                                "decoded_count": len(decoded_data.get("variants", [])),
                            },
                            error="Data integrity check failed" if not integrity_ok else None,
                        )
                    )
                else:
                    self.results.append(
                        TestResult(
                            "data_integrity",
                            False,
                            duration_ms,
                            error=f"Decode failed: HTTP {decode_response.status_code}",
                        )
                    )
            else:
                self.results.append(
                    TestResult(
                        "data_integrity",
                        False,
                        (time.time() - start_time) * 1000,
                        error=f"Encode failed: HTTP {encode_response.status_code}",
                    )
                )

        except Exception as e:
            self.results.append(
                TestResult("data_integrity", False, (time.time() - start_time) * 1000, error=str(e))
            )

    async def _test_failover(self) -> None:
        """Test basic failover capabilities."""
        # This would test connection to backup PIR servers, etc.
        # Simplified for this implementation

        await self._test_endpoint(
            "failover_test",
            "/api/v1/system/failover-status",
            expected_status=200,
            check_response=lambda r: r.get("backup_systems_available", False),
        )

    async def _run_privacy_tests(self) -> None:
        """Run privacy guarantee validation tests."""
        logger.info("Running privacy validation tests...")

        # PIR privacy test
        await self._test_pir_privacy()

        # Differential privacy test
        await self._test_differential_privacy()

        # Data anonymization test
        await self._test_data_anonymization()

    async def _test_pir_privacy(self) -> None:
        """Test PIR privacy guarantees."""
        start_time = time.time()
        try:
            response = await self.client.get(
                urljoin(self.config.base_url, "/api/v1/pir/privacy-status")
            )

            duration_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                status = response.json()
                servers_available = status.get("servers_available", 0)
                privacy_guarantee = status.get("privacy_guarantee", {})
                breach_probability = privacy_guarantee.get("breach_probability", 1.0)

                privacy_ok = (
                    servers_available >= 2 and breach_probability < self.config.privacy_threshold
                )

                self.results.append(
                    TestResult(
                        "pir_privacy",
                        privacy_ok,
                        duration_ms,
                        {
                            "servers_available": servers_available,
                            "breach_probability": breach_probability,
                        },
                        error=(
                            f"Privacy guarantee breach probability {breach_probability} too high"
                            if not privacy_ok
                            else None
                        ),
                    )
                )
            else:
                self.results.append(
                    TestResult(
                        "pir_privacy",
                        False,
                        duration_ms,
                        error=f"HTTP {response.status_code}: {response.text}",
                    )
                )

        except Exception as e:
            self.results.append(
                TestResult("pir_privacy", False, (time.time() - start_time) * 1000, error=str(e))
            )

    async def _test_differential_privacy(self) -> None:
        """Test differential privacy implementation."""
        test_data = {"query": "test_query", "epsilon": 1.0, "delta": 1e-5}

        await self._test_endpoint(
            "differential_privacy",
            "/api/v1/dp/query",
            method="POST",
            json_data=test_data,
            expected_status=200,
            check_response=lambda r: (
                "result" in r
                and "privacy_cost" in r
                and r["privacy_cost"]["epsilon"] <= test_data["epsilon"]
            ),
        )

    async def _test_data_anonymization(self) -> None:
        """Test data anonymization capabilities."""
        test_data = {
            "data": [
                {"patient_id": "test123", "variant": "rs1234"},
                {"patient_id": "test456", "variant": "rs5678"},
            ],
            "anonymization_level": "k_anonymity",
            "k": 5,
        }

        await self._test_endpoint(
            "data_anonymization",
            "/api/v1/anonymize",
            method="POST",
            json_data=test_data,
            expected_status=200,
            check_response=lambda r: (
                "anonymized_data" in r
                and all("patient_id" not in item for item in r["anonymized_data"])
            ),
        )

    async def _run_security_tests(self) -> None:
        """Run security validation tests."""
        logger.info("Running security tests...")

        # HTTPS enforcement
        await self._test_https_enforcement()

        # Authentication test
        await self._test_authentication()

        # Rate limiting test
        await self._test_rate_limiting()

        # Security headers test
        await self._test_security_headers()

    async def _test_https_enforcement(self) -> None:
        """Test HTTPS enforcement."""
        if not self.config.base_url.startswith("https://"):
            self.results.append(
                TestResult("https_enforcement", False, 0, error="Base URL is not HTTPS")
            )
            return

        # Try HTTP version if HTTPS
        if self.config.validate_ssl:
            http_url = self.config.base_url.replace("https://", "http://")
            start_time = time.time()

            try:
                # Should redirect or fail
                response = await self.client.get(
                    urljoin(http_url, "/health"), follow_redirects=False
                )

                duration_ms = (time.time() - start_time) * 1000

                # Should get redirect (301/302) or connection error
                https_enforced = response.status_code in [301, 302, 308, 426]

                self.results.append(
                    TestResult(
                        "https_enforcement",
                        https_enforced,
                        duration_ms,
                        {"http_status": response.status_code},
                        error=(
                            f"HTTP not redirected: {response.status_code}"
                            if not https_enforced
                            else None
                        ),
                    )
                )

            except Exception:
                # Connection error is also acceptable (HTTPS-only)
                self.results.append(
                    TestResult(
                        "https_enforcement",
                        True,
                        (time.time() - start_time) * 1000,
                        {"http_blocked": True},
                    )
                )

    async def _test_authentication(self) -> None:
        """Test authentication requirements."""
        start_time = time.time()

        # Create client without auth
        no_auth_client = httpx.AsyncClient(timeout=self.config.timeout)

        try:
            response = await no_auth_client.get(urljoin(self.config.base_url, "/api/v1/status"))

            duration_ms = (time.time() - start_time) * 1000

            # Should require authentication (401/403)
            auth_required = response.status_code in [401, 403]

            self.results.append(
                TestResult(
                    "authentication_required",
                    auth_required,
                    duration_ms,
                    {"status_code": response.status_code},
                    error=(
                        f"No authentication required: {response.status_code}"
                        if not auth_required
                        else None
                    ),
                )
            )

        except Exception as e:
            self.results.append(
                TestResult(
                    "authentication_required",
                    False,
                    (time.time() - start_time) * 1000,
                    error=str(e),
                )
            )
        finally:
            await no_auth_client.aclose()

    async def _test_rate_limiting(self) -> None:
        """Test rate limiting implementation."""
        start_time = time.time()

        # Make rapid requests to trigger rate limiting
        tasks = []
        for _ in range(20):  # 20 rapid requests
            tasks.append(self.client.get(urljoin(self.config.base_url, "/health")))

        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            duration_ms = (time.time() - start_time) * 1000

            status_codes = []
            for response in responses:
                if isinstance(response, httpx.Response):
                    status_codes.append(response.status_code)
                elif isinstance(response, Exception):
                    status_codes.append(0)  # Connection error

            # Should have some rate limiting (429) or connection limits
            rate_limited = any(code == 429 for code in status_codes)

            self.results.append(
                TestResult(
                    "rate_limiting",
                    rate_limited,
                    duration_ms,
                    {
                        "status_codes": status_codes,
                        "rate_limited_requests": sum(1 for code in status_codes if code == 429),
                    },
                    error="No rate limiting detected" if not rate_limited else None,
                )
            )

        except Exception as e:
            self.results.append(
                TestResult("rate_limiting", False, (time.time() - start_time) * 1000, error=str(e))
            )

    async def _test_security_headers(self) -> None:
        """Test security headers presence."""
        start_time = time.time()

        try:
            response = await self.client.get(urljoin(self.config.base_url, "/health"))
            duration_ms = (time.time() - start_time) * 1000

            required_headers = {
                "strict-transport-security": "HSTS",
                "x-frame-options": "Clickjacking protection",
                "x-content-type-options": "MIME type sniffing protection",
                "x-xss-protection": "XSS protection",
            }

            missing_headers = []
            for header, description in required_headers.items():
                if header not in response.headers:
                    missing_headers.append(f"{header} ({description})")

            headers_ok = len(missing_headers) == 0

            self.results.append(
                TestResult(
                    "security_headers",
                    headers_ok,
                    duration_ms,
                    {
                        "present_headers": [
                            h for h in required_headers.keys() if h in response.headers
                        ],
                        "missing_headers": missing_headers,
                    },
                    error=(
                        f"Missing security headers: {', '.join(missing_headers)}"
                        if missing_headers
                        else None
                    ),
                )
            )

        except Exception as e:
            self.results.append(
                TestResult(
                    "security_headers", False, (time.time() - start_time) * 1000, error=str(e)
                )
            )

    async def _run_performance_tests(self) -> None:
        """Run basic performance tests."""
        logger.info("Running performance tests...")

        # Response time test
        await self._test_response_time()

        # Metrics availability test
        await self._test_metrics_endpoint()

    async def _test_response_time(self) -> None:
        """Test API response times."""
        endpoints = ["/health", "/api/v1/version", "/api/v1/status"]

        for endpoint in endpoints:
            start_time = time.time()
            try:
                response = await self.client.get(urljoin(self.config.base_url, endpoint))
                duration_ms = (time.time() - start_time) * 1000

                # Response should be under 1 second
                fast_enough = duration_ms < 1000

                self.results.append(
                    TestResult(
                        f"response_time_{endpoint.replace('/', '_')}",
                        fast_enough,
                        duration_ms,
                        {"response_time_ms": duration_ms},
                        error=f"Slow response: {duration_ms:.0f}ms" if not fast_enough else None,
                    )
                )

            except Exception as e:
                self.results.append(
                    TestResult(
                        f"response_time_{endpoint.replace('/', '_')}",
                        False,
                        (time.time() - start_time) * 1000,
                        error=str(e),
                    )
                )

    async def _test_metrics_endpoint(self) -> None:
        """Test Prometheus metrics endpoint."""
        await self._test_endpoint(
            "metrics_endpoint",
            "/metrics",
            expected_status=200,
            check_response=self._validate_prometheus_metrics,
            expect_json=False,
        )

    def _validate_prometheus_metrics(self, content: str) -> bool:
        """Validate Prometheus metrics format."""
        try:
            families = list(text_string_to_metric_families(content))

            # Should have some basic metrics
            metric_names = [family.name for family in families]

            required_metrics = [
                "genomevault_http_requests_total",
                "genomevault_pir_query_duration_seconds",
                "python_info",
            ]

            return any(metric in metric_names for metric in required_metrics)
        except Exception:
            return False

    def _validate_system_status(self, response: Dict[str, Any]) -> bool:
        """Validate system status response."""
        required_fields = ["status", "timestamp", "components"]
        if not all(field in response for field in required_fields):
            return False

        # Check component health
        components = response.get("components", {})
        for component, status in components.items():
            if status.get("status") != "healthy":
                logger.warning(f"Component {component} is not healthy: {status}")

        return response.get("status") == "healthy"

    async def _test_endpoint(
        self,
        test_name: str,
        endpoint: str,
        method: str = "GET",
        json_data: Optional[Dict] = None,
        expected_status: int = 200,
        check_response: Optional[callable] = None,
        expect_json: bool = True,
    ) -> None:
        """Generic endpoint test helper."""
        start_time = time.time()

        try:
            if method.upper() == "POST":
                response = await self.client.post(
                    urljoin(self.config.base_url, endpoint), json=json_data
                )
            else:
                response = await self.client.get(urljoin(self.config.base_url, endpoint))

            duration_ms = (time.time() - start_time) * 1000

            if response.status_code != expected_status:
                self.results.append(
                    TestResult(
                        test_name,
                        False,
                        duration_ms,
                        {"status_code": response.status_code},
                        error=f"Expected {expected_status}, got {response.status_code}",
                    )
                )
                return

            # Check response format and content
            if expect_json:
                try:
                    response_data = response.json()
                except json.JSONDecodeError:
                    self.results.append(
                        TestResult(
                            test_name, False, duration_ms, error="Response is not valid JSON"
                        )
                    )
                    return
            else:
                response_data = response.text

            # Custom response validation
            if check_response:
                validation_ok = check_response(response_data)
                self.results.append(
                    TestResult(
                        test_name,
                        validation_ok,
                        duration_ms,
                        {"response_validated": validation_ok},
                        error="Response validation failed" if not validation_ok else None,
                    )
                )
            else:
                self.results.append(
                    TestResult(test_name, True, duration_ms, {"status_code": response.status_code})
                )

        except Exception as e:
            self.results.append(
                TestResult(test_name, False, (time.time() - start_time) * 1000, error=str(e))
            )


def get_environment_config(environment: str) -> Dict[str, str]:
    """Get environment-specific configuration."""
    configs = {
        "dev": {"base_url": "https://dev.genomevault.io", "validate_ssl": False, "timeout": 30},
        "staging": {
            "base_url": "https://staging.genomevault.io",
            "validate_ssl": True,
            "timeout": 60,
        },
        "prod": {"base_url": "https://genomevault.io", "validate_ssl": True, "timeout": 45},
    }

    return configs.get(environment, configs["dev"])


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GenomeVault smoke tests")
    parser.add_argument(
        "--environment",
        choices=["dev", "staging", "prod"],
        default="dev",
        help="Target environment",
    )
    parser.add_argument("--base-url", help="Override base URL")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive test suite")
    parser.add_argument(
        "--validate-privacy", action="store_true", default=True, help="Validate privacy guarantees"
    )
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Maximum retries for failed tests"
    )

    args = parser.parse_args()

    # Get environment config
    env_config = get_environment_config(args.environment)

    # Override with command line args
    base_url = args.base_url or env_config["base_url"]

    # Get API key from environment if not provided
    api_key = args.api_key or os.getenv("GENOMEVAULT_API_KEY")

    config = SmokeTestConfig(
        base_url=base_url,
        timeout=args.timeout,
        comprehensive=args.comprehensive,
        validate_privacy=args.validate_privacy,
        environment=args.environment,
        api_key=api_key,
        max_retries=args.max_retries,
        validate_ssl=env_config.get("validate_ssl", True),
    )

    runner = SmokeTestRunner(config)

    try:
        success = await runner.run_all_tests()

        # Write results to file for CI/CD pipeline
        results_file = f"smoke-test-results-{args.environment}.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "environment": args.environment,
                    "timestamp": time.time(),
                    "success": success,
                    "total_tests": len(runner.results),
                    "passed_tests": len([r for r in runner.results if r.passed]),
                    "failed_tests": len([r for r in runner.results if not r.passed]),
                    "results": [
                        {
                            "name": r.name,
                            "passed": r.passed,
                            "duration_ms": r.duration_ms,
                            "details": r.details,
                            "error": r.error,
                        }
                        for r in runner.results
                    ],
                },
                f,
                indent=2,
            )

        logger.info(f"Results written to {results_file}")

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
