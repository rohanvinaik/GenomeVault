#!/usr/bin/env python3
"""
Synthetic monitoring for GenomeVault privacy guarantees.

Continuously monitors privacy-preserving functionality including:
- PIR query privacy validation
- Zero-knowledge proof verification 
- Differential privacy budget compliance
- Hypervector encoding integrity
- End-to-end genomic workflow testing
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import httpx
import numpy as np
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, push_to_gateway

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PrivacyTestResult:
    """Privacy test result with detailed metrics."""
    test_name: str
    success: bool
    privacy_breach_probability: float
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SyntheticMonitorConfig:
    """Configuration for synthetic monitoring."""
    base_url: str
    environment: str = "prod"
    api_key: Optional[str] = None
    
    # Test intervals (seconds)
    pir_test_interval: int = 300  # 5 minutes
    zk_test_interval: int = 600   # 10 minutes
    dp_test_interval: int = 180   # 3 minutes
    e2e_test_interval: int = 900  # 15 minutes
    
    # Privacy thresholds
    max_privacy_breach_prob: float = 0.01  # 1%
    max_dp_epsilon_per_user: float = 1.0   # Per query
    min_pir_servers: int = 2
    
    # Performance thresholds
    max_pir_latency_ms: int = 500
    max_zk_proof_time_ms: int = 2000
    max_hv_encoding_time_ms: int = 100
    
    # Monitoring
    pushgateway_url: Optional[str] = None
    alert_webhook: Optional[str] = None
    
    # Test parameters
    synthetic_genome_size: int = 1000  # Number of variants
    test_database_size: int = 10000    # PIR database size


class PrivacyGuaranteeMonitor:
    """Monitor privacy guarantees across GenomeVault services."""
    
    def __init__(self, config: SyntheticMonitorConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=30.0,
            verify=True
        )
        
        if config.api_key:
            self.client.headers["Authorization"] = f"Bearer {config.api_key}"
        
        self.test_results: List[PrivacyTestResult] = []
        self.registry = CollectorRegistry()
        
        # Metrics
        self.privacy_breach_probability = Gauge(
            'genomevault_synthetic_privacy_breach_probability',
            'Current privacy breach probability from synthetic tests',
            ['test_type', 'environment'],
            registry=self.registry
        )
        
        self.test_duration = Histogram(
            'genomevault_synthetic_test_duration_seconds',
            'Duration of synthetic privacy tests',
            ['test_type', 'environment'],
            registry=self.registry
        )
        
        self.test_success = Counter(
            'genomevault_synthetic_tests_total',
            'Total synthetic tests run',
            ['test_type', 'environment', 'status'],
            registry=self.registry
        )
        
        self.privacy_violations = Counter(
            'genomevault_synthetic_privacy_violations_total',
            'Privacy violations detected in synthetic tests',
            ['test_type', 'environment', 'violation_type'],
            registry=self.registry
        )
    
    async def run_continuous_monitoring(self, duration_hours: Optional[int] = None) -> None:
        """Run continuous monitoring for specified duration or indefinitely."""
        logger.info(f"Starting synthetic monitoring for {self.config.environment}")
        
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600) if duration_hours else float('inf')
        
        # Schedule initial tests
        tasks = [
            self._schedule_pir_tests(),
            self._schedule_zk_tests(),
            self._schedule_dp_tests(),
            self._schedule_e2e_tests(),
            self._schedule_metrics_reporting()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
            raise
        finally:
            await self.client.aclose()
    
    async def _schedule_pir_tests(self) -> None:
        """Schedule PIR privacy tests."""
        while True:
            try:
                await self._test_pir_privacy()
                await asyncio.sleep(self.config.pir_test_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"PIR test scheduling failed: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
    
    async def _schedule_zk_tests(self) -> None:
        """Schedule ZK proof tests."""
        while True:
            try:
                await self._test_zk_privacy()
                await asyncio.sleep(self.config.zk_test_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ZK test scheduling failed: {e}")
                await asyncio.sleep(60)
    
    async def _schedule_dp_tests(self) -> None:
        """Schedule differential privacy tests."""
        while True:
            try:
                await self._test_differential_privacy()
                await asyncio.sleep(self.config.dp_test_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"DP test scheduling failed: {e}")
                await asyncio.sleep(60)
    
    async def _schedule_e2e_tests(self) -> None:
        """Schedule end-to-end workflow tests."""
        while True:
            try:
                await self._test_e2e_workflow()
                await asyncio.sleep(self.config.e2e_test_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"E2E test scheduling failed: {e}")
                await asyncio.sleep(60)
    
    async def _schedule_metrics_reporting(self) -> None:
        """Schedule metrics reporting to Pushgateway."""
        while True:
            try:
                if self.config.pushgateway_url:
                    await self._push_metrics()
                await asyncio.sleep(60)  # Report every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics reporting failed: {e}")
                await asyncio.sleep(60)
    
    async def _test_pir_privacy(self) -> None:
        """Test PIR privacy guarantees."""
        logger.info("Testing PIR privacy guarantees...")
        
        start_time = time.time()
        
        try:
            # Test 1: Verify minimum server requirement
            server_status = await self._check_pir_servers()
            if not server_status["success"]:
                self._record_privacy_test(
                    "pir_servers",
                    False,
                    1.0,  # Maximum breach probability if servers insufficient
                    (time.time() - start_time) * 1000,
                    error=server_status["error"]
                )
                return
            
            # Test 2: Perform actual PIR query and validate privacy
            pir_result = await self._perform_pir_query_test()
            if not pir_result["success"]:
                self._record_privacy_test(
                    "pir_query",
                    False,
                    pir_result.get("breach_probability", 1.0),
                    (time.time() - start_time) * 1000,
                    error=pir_result["error"]
                )
                return
            
            # Test 3: Validate query pattern privacy
            pattern_result = await self._test_pir_query_patterns()
            
            overall_success = (
                server_status["success"] and
                pir_result["success"] and
                pattern_result["success"]
            )
            
            max_breach_prob = max(
                pir_result.get("breach_probability", 0),
                pattern_result.get("breach_probability", 0)
            )
            
            self._record_privacy_test(
                "pir_privacy_overall",
                overall_success,
                max_breach_prob,
                (time.time() - start_time) * 1000,
                details={
                    "servers_available": server_status.get("servers", 0),
                    "query_privacy_verified": pir_result["success"],
                    "pattern_privacy_verified": pattern_result["success"]
                }
            )
            
        except Exception as e:
            self._record_privacy_test(
                "pir_privacy_overall",
                False,
                1.0,
                (time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def _check_pir_servers(self) -> Dict[str, Any]:
        """Check PIR server availability and configuration."""
        try:
            response = await self.client.get(
                urljoin(self.config.base_url, "/api/v1/pir/status")
            )
            
            if response.status_code == 200:
                status = response.json()
                servers_available = status.get("servers_available", 0)
                
                if servers_available >= self.config.min_pir_servers:
                    return {"success": True, "servers": servers_available}
                else:
                    return {
                        "success": False,
                        "servers": servers_available,
                        "error": f"Only {servers_available} servers available, need {self.config.min_pir_servers}"
                    }
            else:
                return {
                    "success": False,
                    "error": f"PIR status check failed: HTTP {response.status_code}"
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _perform_pir_query_test(self) -> Dict[str, Any]:
        """Perform PIR query and validate privacy."""
        # Generate synthetic query
        query_data = {
            "database_size": self.config.test_database_size,
            "query_index": random.randint(0, self.config.test_database_size - 1),
            "privacy_level": "information_theoretic",
            "test_mode": True
        }
        
        try:
            response = await self.client.post(
                urljoin(self.config.base_url, "/api/v1/pir/query"),
                json=query_data
            )
            
            if response.status_code == 200:
                result = response.json()
                privacy_info = result.get("privacy_guarantee", {})
                breach_prob = privacy_info.get("breach_probability", 1.0)
                
                if breach_prob <= self.config.max_privacy_breach_prob:
                    return {
                        "success": True,
                        "breach_probability": breach_prob,
                        "query_time_ms": result.get("query_time_ms", 0)
                    }
                else:
                    return {
                        "success": False,
                        "breach_probability": breach_prob,
                        "error": f"Privacy breach probability {breach_prob} exceeds threshold {self.config.max_privacy_breach_prob}"
                    }
            else:
                return {
                    "success": False,
                    "error": f"PIR query failed: HTTP {response.status_code}"
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_pir_query_patterns(self) -> Dict[str, Any]:
        """Test that query patterns don't leak information."""
        # Perform multiple queries and validate they don't reveal patterns
        num_queries = 5
        query_times = []
        
        try:
            for i in range(num_queries):
                query_data = {
                    "database_size": self.config.test_database_size,
                    "query_index": random.randint(0, self.config.test_database_size - 1),
                    "privacy_level": "information_theoretic",
                    "test_mode": True
                }
                
                start_time = time.time()
                response = await self.client.post(
                    urljoin(self.config.base_url, "/api/v1/pir/query"),
                    json=query_data
                )
                query_time = (time.time() - start_time) * 1000
                query_times.append(query_time)
                
                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"Query {i} failed: HTTP {response.status_code}"
                    }
            
            # Validate constant-time execution (timing attack resistance)
            time_variance = np.var(query_times)
            max_allowed_variance = (self.config.max_pir_latency_ms * 0.1) ** 2  # 10% variance
            
            constant_time_ok = time_variance <= max_allowed_variance
            
            return {
                "success": constant_time_ok,
                "breach_probability": 0.0 if constant_time_ok else 0.5,  # Timing attacks
                "time_variance": time_variance,
                "query_times": query_times
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_zk_privacy(self) -> None:
        """Test zero-knowledge proof privacy."""
        logger.info("Testing ZK proof privacy guarantees...")
        
        start_time = time.time()
        
        try:
            # Test ZK proof generation and verification
            proof_result = await self._generate_and_verify_zk_proof()
            
            if proof_result["success"]:
                # Validate proof doesn't leak information
                privacy_result = await self._validate_zk_privacy(proof_result["proof"])
                
                self._record_privacy_test(
                    "zk_privacy",
                    privacy_result["success"],
                    privacy_result.get("breach_probability", 0.0),
                    (time.time() - start_time) * 1000,
                    details={
                        "proof_verified": proof_result["success"],
                        "zero_knowledge_validated": privacy_result["success"],
                        "proof_size_bytes": len(proof_result.get("proof", ""))
                    }
                )
            else:
                self._record_privacy_test(
                    "zk_privacy",
                    False,
                    1.0,
                    (time.time() - start_time) * 1000,
                    error=proof_result["error"]
                )
                
        except Exception as e:
            self._record_privacy_test(
                "zk_privacy",
                False,
                1.0,
                (time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def _generate_and_verify_zk_proof(self) -> Dict[str, Any]:
        """Generate and verify a test ZK proof."""
        # Create synthetic genomic data for proof
        proof_data = {
            "proof_type": "variant_presence",
            "public_input": {
                "variant_hash": "0x" + os.urandom(16).hex(),
                "population_frequency": random.uniform(0.01, 0.99)
            },
            "private_input": {
                "user_genotype": random.choice(["0/0", "0/1", "1/1"]),
                "confidence_score": random.uniform(0.8, 1.0)
            },
            "circuit": "variant_presence_v1"
        }
        
        try:
            # Generate proof
            prove_response = await self.client.post(
                urljoin(self.config.base_url, "/api/v1/zk/prove"),
                json=proof_data
            )
            
            if prove_response.status_code == 200:
                proof_result = prove_response.json()
                
                # Verify proof
                verify_data = {
                    "proof": proof_result["proof"],
                    "public_input": proof_data["public_input"],
                    "circuit": proof_data["circuit"]
                }
                
                verify_response = await self.client.post(
                    urljoin(self.config.base_url, "/api/v1/zk/verify"),
                    json=verify_data
                )
                
                if verify_response.status_code == 200:
                    verify_result = verify_response.json()
                    if verify_result.get("valid"):
                        return {
                            "success": True,
                            "proof": proof_result["proof"],
                            "proof_time_ms": proof_result.get("proof_time_ms", 0),
                            "verify_time_ms": verify_result.get("verify_time_ms", 0)
                        }
                    else:
                        return {"success": False, "error": "Proof verification failed"}
                else:
                    return {"success": False, "error": f"Verification failed: HTTP {verify_response.status_code}"}
            else:
                return {"success": False, "error": f"Proof generation failed: HTTP {prove_response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _validate_zk_privacy(self, proof: str) -> Dict[str, Any]:
        """Validate that ZK proof doesn't leak private information."""
        # Basic validation - proof should be opaque
        try:
            # Check proof format and size
            if len(proof) == 0:
                return {"success": False, "breach_probability": 1.0, "error": "Empty proof"}
            
            # Validate proof doesn't contain obvious private data patterns
            # (In practice, this would be more sophisticated)
            privacy_violations = []
            
            if any(pattern in proof.lower() for pattern in ["genotype", "patient", "variant"]):
                privacy_violations.append("Contains private data keywords")
            
            if len(privacy_violations) == 0:
                return {"success": True, "breach_probability": 0.0}
            else:
                return {
                    "success": False,
                    "breach_probability": 0.5,
                    "error": f"Privacy violations: {', '.join(privacy_violations)}"
                }
                
        except Exception as e:
            return {"success": False, "breach_probability": 1.0, "error": str(e)}
    
    async def _test_differential_privacy(self) -> None:
        """Test differential privacy implementation."""
        logger.info("Testing differential privacy guarantees...")
        
        start_time = time.time()
        
        try:
            # Test DP query
            dp_result = await self._perform_dp_query_test()
            
            if dp_result["success"]:
                # Validate epsilon budget tracking
                budget_result = await self._validate_dp_budget_tracking()
                
                overall_success = dp_result["success"] and budget_result["success"]
                
                self._record_privacy_test(
                    "differential_privacy",
                    overall_success,
                    dp_result.get("privacy_cost", 0.0),
                    (time.time() - start_time) * 1000,
                    details={
                        "epsilon_consumed": dp_result.get("epsilon_consumed", 0),
                        "budget_tracking_ok": budget_result["success"],
                        "noise_added": dp_result.get("noise_added", False)
                    }
                )
            else:
                self._record_privacy_test(
                    "differential_privacy",
                    False,
                    1.0,
                    (time.time() - start_time) * 1000,
                    error=dp_result["error"]
                )
                
        except Exception as e:
            self._record_privacy_test(
                "differential_privacy",
                False,
                1.0,
                (time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def _perform_dp_query_test(self) -> Dict[str, Any]:
        """Perform differential privacy query test."""
        dp_query = {
            "query_type": "frequency",
            "dataset": "synthetic_variants",
            "filter": {
                "chromosome": "1",
                "position_range": [100000, 200000]
            },
            "epsilon": self.config.max_dp_epsilon_per_user,
            "delta": 1e-6
        }
        
        try:
            response = await self.client.post(
                urljoin(self.config.base_url, "/api/v1/dp/query"),
                json=dp_query
            )
            
            if response.status_code == 200:
                result = response.json()
                privacy_cost = result.get("privacy_cost", {})
                epsilon_used = privacy_cost.get("epsilon", 0)
                
                if epsilon_used <= self.config.max_dp_epsilon_per_user:
                    return {
                        "success": True,
                        "epsilon_consumed": epsilon_used,
                        "privacy_cost": epsilon_used,
                        "noise_added": result.get("noise_added", False),
                        "result": result.get("result")
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Epsilon consumption {epsilon_used} exceeds limit {self.config.max_dp_epsilon_per_user}"
                    }
            else:
                return {"success": False, "error": f"DP query failed: HTTP {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _validate_dp_budget_tracking(self) -> Dict[str, Any]:
        """Validate differential privacy budget tracking."""
        try:
            response = await self.client.get(
                urljoin(self.config.base_url, "/api/v1/dp/budget/synthetic_user")
            )
            
            if response.status_code == 200:
                budget_info = response.json()
                remaining_budget = budget_info.get("remaining_epsilon", 0)
                
                # Budget tracking is working if we have remaining budget info
                return {
                    "success": remaining_budget >= 0,
                    "remaining_budget": remaining_budget,
                    "consumed_budget": budget_info.get("consumed_epsilon", 0)
                }
            else:
                return {"success": False, "error": f"Budget check failed: HTTP {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_e2e_workflow(self) -> None:
        """Test end-to-end genomic workflow with privacy preservation."""
        logger.info("Testing end-to-end privacy-preserving workflow...")
        
        start_time = time.time()
        
        try:
            # Step 1: Upload synthetic genomic data
            upload_result = await self._upload_synthetic_genome()
            if not upload_result["success"]:
                self._record_privacy_test(
                    "e2e_workflow",
                    False,
                    1.0,
                    (time.time() - start_time) * 1000,
                    error=f"Upload failed: {upload_result['error']}"
                )
                return
            
            # Step 2: Perform hypervector encoding
            encoding_result = await self._test_hypervector_encoding()
            if not encoding_result["success"]:
                self._record_privacy_test(
                    "e2e_workflow",
                    False,
                    0.5,
                    (time.time() - start_time) * 1000,
                    error=f"Encoding failed: {encoding_result['error']}"
                )
                return
            
            # Step 3: Perform privacy-preserving analysis
            analysis_result = await self._perform_private_analysis(upload_result["data_id"])
            
            # Step 4: Validate overall privacy
            privacy_validation = await self._validate_e2e_privacy(
                upload_result, encoding_result, analysis_result
            )
            
            overall_success = all([
                upload_result["success"],
                encoding_result["success"], 
                analysis_result["success"],
                privacy_validation["success"]
            ])
            
            self._record_privacy_test(
                "e2e_workflow",
                overall_success,
                privacy_validation.get("breach_probability", 0.0),
                (time.time() - start_time) * 1000,
                details={
                    "upload_success": upload_result["success"],
                    "encoding_success": encoding_result["success"],
                    "analysis_success": analysis_result["success"],
                    "privacy_validated": privacy_validation["success"],
                    "total_variants": upload_result.get("variant_count", 0)
                }
            )
            
        except Exception as e:
            self._record_privacy_test(
                "e2e_workflow",
                False,
                1.0,
                (time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def _upload_synthetic_genome(self) -> Dict[str, Any]:
        """Upload synthetic genomic data for testing."""
        # Generate synthetic variants
        variants = []
        for i in range(self.config.synthetic_genome_size):
            variants.append({
                "chromosome": str(random.randint(1, 22)),
                "position": random.randint(1000000, 249000000),
                "reference": random.choice(["A", "T", "G", "C"]),
                "alternate": random.choice(["A", "T", "G", "C"]),
                "genotype": random.choice(["0/0", "0/1", "1/1"]),
                "quality": random.uniform(30, 99)
            })
        
        upload_data = {
            "user_id": "synthetic_test_user",
            "variants": variants,
            "metadata": {
                "source": "synthetic_monitor",
                "timestamp": time.time()
            }
        }
        
        try:
            response = await self.client.post(
                urljoin(self.config.base_url, "/api/v1/genomic/upload"),
                json=upload_data
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "data_id": result.get("data_id"),
                    "variant_count": len(variants)
                }
            else:
                return {"success": False, "error": f"Upload failed: HTTP {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_hypervector_encoding(self) -> Dict[str, Any]:
        """Test hypervector encoding functionality."""
        test_variants = [
            {"chrom": "1", "pos": 12345, "ref": "A", "alt": "T"},
            {"chrom": "2", "pos": 67890, "ref": "C", "alt": "G"}
        ]
        
        encoding_data = {
            "variants": test_variants,
            "dimension": 10000,
            "encoding_type": "privacy_preserving"
        }
        
        try:
            start_time = time.time()
            response = await self.client.post(
                urljoin(self.config.base_url, "/api/v1/hv/encode"),
                json=encoding_data
            )
            encoding_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                hypervector = result.get("hypervector", [])
                
                # Validate encoding properties
                if len(hypervector) == 10000 and encoding_time <= self.config.max_hv_encoding_time_ms:
                    return {
                        "success": True,
                        "hypervector_dimension": len(hypervector),
                        "encoding_time_ms": encoding_time
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Invalid encoding: dim={len(hypervector)}, time={encoding_time}ms"
                    }
            else:
                return {"success": False, "error": f"Encoding failed: HTTP {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _perform_private_analysis(self, data_id: str) -> Dict[str, Any]:
        """Perform privacy-preserving analysis."""
        analysis_request = {
            "data_id": data_id,
            "analysis_type": "similarity_search",
            "privacy_level": "maximum",
            "parameters": {
                "threshold": 0.8,
                "max_results": 10
            }
        }
        
        try:
            response = await self.client.post(
                urljoin(self.config.base_url, "/api/v1/analysis/private"),
                json=analysis_request
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "results_count": len(result.get("results", [])),
                    "privacy_guarantees": result.get("privacy_guarantees", {})
                }
            else:
                return {"success": False, "error": f"Analysis failed: HTTP {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _validate_e2e_privacy(self, *results) -> Dict[str, Any]:
        """Validate overall privacy of end-to-end workflow."""
        # Aggregate privacy guarantees from all steps
        total_breach_prob = 0.0
        
        for result in results:
            if isinstance(result, dict):
                privacy_info = result.get("privacy_guarantees", {})
                breach_prob = privacy_info.get("breach_probability", 0.0)
                total_breach_prob = max(total_breach_prob, breach_prob)
        
        # Privacy composition - simplified
        if total_breach_prob <= self.config.max_privacy_breach_prob:
            return {"success": True, "breach_probability": total_breach_prob}
        else:
            return {
                "success": False,
                "breach_probability": total_breach_prob,
                "error": f"Composed privacy breach probability {total_breach_prob} exceeds threshold"
            }
    
    def _record_privacy_test(
        self,
        test_name: str,
        success: bool,
        breach_probability: float,
        duration_ms: float,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Record privacy test result."""
        result = PrivacyTestResult(
            test_name=test_name,
            success=success,
            privacy_breach_probability=breach_probability,
            duration_ms=duration_ms,
            details=details or {},
            error=error
        )
        
        self.test_results.append(result)
        
        # Update metrics
        status = "success" if success else "failure"
        self.test_success.labels(
            test_type=test_name,
            environment=self.config.environment,
            status=status
        ).inc()
        
        self.privacy_breach_probability.labels(
            test_type=test_name,
            environment=self.config.environment
        ).set(breach_probability)
        
        self.test_duration.labels(
            test_type=test_name,
            environment=self.config.environment
        ).observe(duration_ms / 1000.0)
        
        if not success or breach_probability > self.config.max_privacy_breach_prob:
            violation_type = "failure" if not success else "privacy_breach"
            self.privacy_violations.labels(
                test_type=test_name,
                environment=self.config.environment,
                violation_type=violation_type
            ).inc()
            
            # Send alert
            asyncio.create_task(self._send_alert(result))
        
        # Log result
        status_str = "✅ PASS" if success else "❌ FAIL"
        logger.info(
            f"{status_str} {test_name}: breach_prob={breach_probability:.6f}, "
            f"duration={duration_ms:.1f}ms"
        )
        
        if error:
            logger.error(f"  Error: {error}")
    
    async def _send_alert(self, result: PrivacyTestResult) -> None:
        """Send alert for privacy violation or test failure."""
        if not self.config.alert_webhook:
            return
        
        alert_data = {
            "alert_type": "privacy_test_failure",
            "test_name": result.test_name,
            "environment": self.config.environment,
            "success": result.success,
            "breach_probability": result.privacy_breach_probability,
            "error": result.error,
            "timestamp": result.timestamp,
            "details": result.details
        }
        
        try:
            async with httpx.AsyncClient() as client:
                await client.post(self.config.alert_webhook, json=alert_data)
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    async def _push_metrics(self) -> None:
        """Push metrics to Pushgateway."""
        if not self.config.pushgateway_url:
            return
        
        try:
            push_to_gateway(
                self.config.pushgateway_url,
                job="genomevault-synthetic-monitoring",
                registry=self.registry,
                grouping_key={"environment": self.config.environment}
            )
        except Exception as e:
            logger.error(f"Failed to push metrics: {e}")
    
    async def run_single_validation(self, test_types: Optional[List[str]] = None) -> bool:
        """Run single validation round."""
        logger.info("Running single validation round...")
        
        test_types = test_types or ["pir", "zk", "dp", "e2e"]
        
        tasks = []
        if "pir" in test_types:
            tasks.append(self._test_pir_privacy())
        if "zk" in test_types:
            tasks.append(self._test_zk_privacy())
        if "dp" in test_types:
            tasks.append(self._test_differential_privacy())
        if "e2e" in test_types:
            tasks.append(self._test_e2e_workflow())
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        failed_tests = [r for r in self.test_results if not r.success]
        privacy_violations = [
            r for r in self.test_results
            if r.privacy_breach_probability > self.config.max_privacy_breach_prob
        ]
        
        success = len(failed_tests) == 0 and len(privacy_violations) == 0
        
        logger.info(f"Validation complete: {len(self.test_results)} tests, {len(failed_tests)} failures, {len(privacy_violations)} privacy violations")
        
        return success


def get_environment_config(environment: str) -> Dict[str, Any]:
    """Get environment-specific configuration."""
    configs = {
        "dev": {
            "base_url": "https://dev.genomevault.io",
            "pir_test_interval": 600,
            "max_privacy_breach_prob": 0.05  # More lenient for dev
        },
        "staging": {
            "base_url": "https://staging.genomevault.io",
            "pir_test_interval": 300,
            "max_privacy_breach_prob": 0.01
        },
        "prod": {
            "base_url": "https://genomevault.io",
            "pir_test_interval": 180,
            "max_privacy_breach_prob": 0.005  # Strictest for prod
        }
    }
    
    return configs.get(environment, configs["prod"])


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GenomeVault synthetic monitoring")
    parser.add_argument(
        "--environment",
        choices=["dev", "staging", "prod"],
        default="prod",
        help="Target environment"
    )
    parser.add_argument(
        "--base-url",
        help="Override base URL"
    )
    parser.add_argument(
        "--duration-hours",
        type=int,
        help="Run for specified hours (default: continuous)"
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run once and exit"
    )
    parser.add_argument(
        "--validate-privacy",
        action="store_true",
        default=True,
        help="Run privacy validation tests"
    )
    parser.add_argument(
        "--validate-pir",
        action="store_true",
        help="Run PIR-specific validation"
    )
    parser.add_argument(
        "--validate-zk",
        action="store_true",
        help="Run ZK proof validation"
    )
    parser.add_argument(
        "--canary",
        action="store_true",
        help="Monitor canary deployment"
    )
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Run comprehensive validation"
    )
    parser.add_argument(
        "--api-key",
        help="API key for authentication"
    )
    parser.add_argument(
        "--pushgateway-url",
        help="Pushgateway URL for metrics"
    )
    parser.add_argument(
        "--alert-webhook",
        help="Webhook URL for alerts"
    )
    
    args = parser.parse_args()
    
    # Get environment configuration
    env_config = get_environment_config(args.environment)
    
    # Override with command line arguments
    base_url = args.base_url or env_config["base_url"]
    api_key = args.api_key or os.getenv("GENOMEVAULT_API_KEY")
    
    # Adjust for canary
    if args.canary:
        base_url = base_url.replace("://", "://canary.")
    
    config = SyntheticMonitorConfig(
        base_url=base_url,
        environment=args.environment,
        api_key=api_key,
        pushgateway_url=args.pushgateway_url or os.getenv("PUSHGATEWAY_URL"),
        alert_webhook=args.alert_webhook or os.getenv("ALERT_WEBHOOK_URL"),
        **env_config
    )
    
    monitor = PrivacyGuaranteeMonitor(config)
    
    try:
        if args.single_run:
            # Determine test types
            test_types = []
            if args.validate_pir or args.full_validation:
                test_types.append("pir")
            if args.validate_zk or args.full_validation:
                test_types.append("zk")
            if args.full_validation:
                test_types.extend(["dp", "e2e"])
            
            if not test_types:
                test_types = ["pir", "zk", "dp", "e2e"]  # Default all
            
            success = await monitor.run_single_validation(test_types)
            sys.exit(0 if success else 1)
        else:
            await monitor.run_continuous_monitoring(args.duration_hours)
            
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())