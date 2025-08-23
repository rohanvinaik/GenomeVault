#!/usr/bin/env python3
"""
Load Testing Suite for GenomeVault using Locust.

Simulates HIPAA-compliant access patterns with realistic workloads,
tests differential privacy under load, and validates system performance.
"""

import os
import json
import time
import random
import secrets
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np

from locust import HttpUser, task, between, events, LoadTestShape
from locust.env import Environment
from locust.stats import StatsEntry
import gevent


# Configuration from environment
API_BASE_URL = os.getenv('GENOMEVAULT_API_URL', 'http://localhost:8000')
ENABLE_AUTH = os.getenv('ENABLE_AUTH', 'true').lower() == 'true'
MAX_GENOME_SIZE = int(os.getenv('MAX_GENOME_SIZE', '3221225472'))  # 3GB
COMPRESSION_TIER = os.getenv('COMPRESSION_TIER', 'CLINICAL')
DIFFERENTIAL_PRIVACY_EPSILON = float(os.getenv('DP_EPSILON', '1.0'))


class HIPAACompliantUser(HttpUser):
    """
    Simulates a HIPAA-compliant user accessing GenomeVault.
    
    Implements realistic access patterns with:
    - Authentication and authorization
    - Rate limiting compliance
    - PHI access auditing
    - Differential privacy budget tracking
    """
    
    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks
    host = API_BASE_URL
    
    def __init__(self, *args, **kwargs):
        """Initialize user with authentication and privacy budget."""
        super().__init__(*args, **kwargs)
        self.user_id = f"test_user_{secrets.token_hex(8)}"
        self.organization = random.choice(['hospital_a', 'research_b', 'clinic_c'])
        self.role = random.choice(['clinician', 'researcher', 'analyst'])
        self.access_token = None
        self.privacy_budget = DIFFERENTIAL_PRIVACY_EPSILON
        self.privacy_consumed = 0.0
        self.query_count = 0
        self.cache_hits = 0
    
    def on_start(self):
        """Authenticate user and setup session."""
        # Authenticate if enabled
        if ENABLE_AUTH:
            self.authenticate()
        
        # Initialize privacy budget
        self.initialize_privacy_budget()
        
        # Set headers
        self.client.headers.update({
            'X-Request-ID': secrets.token_hex(16),
            'X-User-ID': self.user_id,
            'X-Organization': self.organization,
            'X-Role': self.role,
        })
    
    def authenticate(self):
        """Authenticate and obtain access token."""
        response = self.client.post('/auth/token', json={
            'username': f'{self.user_id}@genomevault.test',
            'password': 'test_password',
            'grant_type': 'password',
            'scope': 'genomic:read clinical:read'
        })
        
        if response.status_code == 200:
            data = response.json()
            self.access_token = data['access_token']
            self.client.headers['Authorization'] = f'Bearer {self.access_token}'
        else:
            print(f"Authentication failed for {self.user_id}")
    
    def initialize_privacy_budget(self):
        """Initialize differential privacy budget."""
        response = self.client.post('/privacy/budget/init', json={
            'user_id': self.user_id,
            'epsilon': DIFFERENTIAL_PRIVACY_EPSILON,
            'delta': 1e-5
        }, catch_response=True)
        
        if response.status_code != 200:
            response.failure(f"Failed to initialize privacy budget")
    
    def consume_privacy_budget(self, amount: float) -> bool:
        """
        Consume privacy budget for a query.
        
        Args:
            amount: Epsilon to consume
            
        Returns:
            True if budget available
        """
        if self.privacy_consumed + amount > self.privacy_budget:
            return False
        
        self.privacy_consumed += amount
        return True
    
    @task(10)
    def query_genomic_variant(self):
        """Query for a specific genomic variant (common operation)."""
        # Check privacy budget
        epsilon_cost = 0.01  # Small epsilon for single variant
        if not self.consume_privacy_budget(epsilon_cost):
            print(f"User {self.user_id} exhausted privacy budget")
            return
        
        # Generate realistic variant query
        chromosome = random.choice(['1', '2', '3', '4', '5', 'X', 'Y'])
        position = random.randint(1000000, 100000000)
        variant_id = f"chr{chromosome}:{position}"
        
        with self.client.get(
            f'/genomic/variant/{variant_id}',
            name='/genomic/variant/[id]',
            catch_response=True
        ) as response:
            if response.status_code == 200:
                self.query_count += 1
                
                # Check if response was from cache
                if response.headers.get('X-Cache-Hit') == 'true':
                    self.cache_hits += 1
                
                # Validate response has fixed-size padding (1KB blocks)
                content_length = len(response.content)
                if content_length % 1024 != 0:
                    response.failure(f"Response not padded to 1KB blocks: {content_length}")
                else:
                    response.success()
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(5)
    def hypervector_encoding(self):
        """Encode genomic data as hypervector."""
        # Generate test genomic data
        num_variants = random.randint(10, 100)
        variants = []
        for _ in range(num_variants):
            variants.append({
                'chromosome': random.choice(['1', '2', '3', '4', '5']),
                'position': random.randint(1000000, 100000000),
                'ref': random.choice(['A', 'C', 'G', 'T']),
                'alt': random.choice(['A', 'C', 'G', 'T']),
                'quality': random.uniform(20, 100)
            })
        
        start_time = time.time()
        
        with self.client.post(
            '/hv/encode',
            json={
                'variants': variants,
                'dimension': 10000,
                'compression_tier': COMPRESSION_TIER
            },
            name='/hv/encode',
            catch_response=True
        ) as response:
            encoding_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate encoding throughput (target: 1000 variants/sec)
                throughput = num_variants / encoding_time
                if throughput < 1000:
                    response.failure(f"Encoding too slow: {throughput:.0f} variants/sec")
                else:
                    response.success()
                    
                # Track compression ratio
                if 'compression_ratio' in data:
                    events.request.fire(
                        request_type='METRIC',
                        name='compression_ratio',
                        response_time=data['compression_ratio'] * 1000,  # Convert to ms for Locust
                        response_length=0,
                        exception=None,
                        context={}
                    )
            else:
                response.failure(f"Encoding failed: {response.status_code}")
    
    @task(3)
    def pir_query(self):
        """Execute PIR query for genomic data."""
        # Check privacy budget (PIR queries consume more epsilon)
        epsilon_cost = 0.1
        if not self.consume_privacy_budget(epsilon_cost):
            return
        
        # Generate PIR query
        database_size = random.choice([1000, 10000, 100000])
        index = random.randint(0, database_size - 1)
        
        with self.client.post(
            '/pir/query',
            json={
                'database_id': 'genomic_variants',
                'index': index,
                'database_size': database_size,
                'use_byzantine_protection': True
            },
            name='/pir/query',
            catch_response=True,
            timeout=10  # 10 second timeout for PIR
        ) as response:
            if response.status_code == 200:
                data = response.json()
                
                # Validate response is 1KB padded
                if 'response_size' in data and data['response_size'] % 1024 != 0:
                    response.failure("PIR response not 1KB padded")
                else:
                    response.success()
            elif response.status_code == 504:
                response.failure("PIR query timeout")
            else:
                response.failure(f"PIR query failed: {response.status_code}")
    
    @task(2)
    def clinical_data_query(self):
        """Query clinical data with anonymization."""
        # Higher epsilon cost for clinical data
        epsilon_cost = 0.2
        if not self.consume_privacy_budget(epsilon_cost):
            return
        
        # Anonymized query
        query_params = {
            'condition': random.choice(['diabetes', 'hypertension', 'cancer']),
            'age_range': random.choice(['0-18', '18-35', '35-50', '50-65', '65+']),
            'anonymize': True,
            'k_anonymity': 5
        }
        
        with self.client.get(
            '/clinical/cohort',
            params=query_params,
            name='/clinical/cohort',
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                
                # Validate k-anonymity
                if 'cohort_size' in data and data['cohort_size'] < 5:
                    response.failure(f"K-anonymity violated: cohort_size={data['cohort_size']}")
                else:
                    response.success()
            else:
                response.failure(f"Clinical query failed: {response.status_code}")
    
    @task(1)
    def zk_proof_verification(self):
        """Verify zero-knowledge proof."""
        # Generate test proof
        proof_data = {
            'proof_type': 'genomic_computation',
            'proof': secrets.token_hex(256),  # Simulated proof
            'public_inputs': {
                'result_hash': secrets.token_hex(32),
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        with self.client.post(
            '/zk/verify',
            json=proof_data,
            name='/zk/verify',
            catch_response=True,
            timeout=5
        ) as response:
            if response.status_code == 200:
                data = response.json()
                
                # Check verification time (should be fast)
                if response.elapsed.total_seconds() > 0.5:
                    response.failure(f"ZK verification too slow: {response.elapsed.total_seconds()}s")
                else:
                    response.success()
            else:
                response.failure(f"ZK verification failed: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Check system health."""
        with self.client.get('/health', name='/health') as response:
            if response.status_code != 200:
                print(f"Health check failed: {response.status_code}")
    
    def on_stop(self):
        """Clean up and report user statistics."""
        # Report privacy budget usage
        if self.privacy_consumed > 0:
            events.request.fire(
                request_type='METRIC',
                name='privacy_budget_consumed',
                response_time=self.privacy_consumed * 1000,
                response_length=0,
                exception=None,
                context={'user_id': self.user_id}
            )
        
        # Report cache hit rate
        if self.query_count > 0:
            cache_hit_rate = self.cache_hits / self.query_count
            events.request.fire(
                request_type='METRIC',
                name='cache_hit_rate',
                response_time=cache_hit_rate * 1000,
                response_length=0,
                exception=None,
                context={'user_id': self.user_id}
            )


class SteppedLoadShape(LoadTestShape):
    """
    Stepped load test shape for gradual load increase.
    
    Simulates realistic traffic patterns:
    1. Warm-up phase (10 users, 2 min)
    2. Normal load (50 users, 5 min)
    3. Peak load (200 users, 5 min)
    4. Stress test (500 users, 3 min)
    5. Cool down (10 users, 2 min)
    """
    
    stages = [
        {'duration': 120, 'users': 10, 'spawn_rate': 2},     # Warm-up
        {'duration': 420, 'users': 50, 'spawn_rate': 5},     # Normal (total 7 min)
        {'duration': 720, 'users': 200, 'spawn_rate': 10},   # Peak (total 12 min)
        {'duration': 900, 'users': 500, 'spawn_rate': 20},   # Stress (total 15 min)
        {'duration': 1020, 'users': 10, 'spawn_rate': 10},   # Cool down (total 17 min)
    ]
    
    def tick(self):
        """Return current load test shape."""
        run_time = self.get_run_time()
        
        for stage in self.stages:
            if run_time < stage['duration']:
                return stage['users'], stage['spawn_rate']
        
        return None  # Test complete


class PrivacyBudgetMonitor:
    """
    Monitor differential privacy budget consumption during load test.
    """
    
    def __init__(self, epsilon_total: float = 10.0):
        """
        Initialize privacy monitor.
        
        Args:
            epsilon_total: Total privacy budget
        """
        self.epsilon_total = epsilon_total
        self.epsilon_consumed = 0.0
        self.user_budgets: Dict[str, float] = {}
        self.violations = []
    
    def consume(self, user_id: str, epsilon: float) -> bool:
        """
        Consume privacy budget.
        
        Args:
            user_id: User ID
            epsilon: Epsilon to consume
            
        Returns:
            True if budget available
        """
        if user_id not in self.user_budgets:
            self.user_budgets[user_id] = 0.0
        
        if self.user_budgets[user_id] + epsilon > DIFFERENTIAL_PRIVACY_EPSILON:
            self.violations.append({
                'user_id': user_id,
                'requested': epsilon,
                'available': DIFFERENTIAL_PRIVACY_EPSILON - self.user_budgets[user_id],
                'timestamp': datetime.utcnow().isoformat()
            })
            return False
        
        self.user_budgets[user_id] += epsilon
        self.epsilon_consumed += epsilon
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get privacy budget statistics."""
        return {
            'total_consumed': self.epsilon_consumed,
            'num_users': len(self.user_budgets),
            'avg_per_user': self.epsilon_consumed / max(len(self.user_budgets), 1),
            'max_per_user': max(self.user_budgets.values()) if self.user_budgets else 0,
            'violations': len(self.violations),
            'violation_rate': len(self.violations) / max(len(self.user_budgets), 1)
        }


# Global privacy monitor
privacy_monitor = PrivacyBudgetMonitor()


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test environment."""
    print("\n" + "="*80)
    print("GenomeVault Load Test Starting")
    print(f"Target Host: {API_BASE_URL}")
    print(f"Max Genome Size: {MAX_GENOME_SIZE / 1e9:.1f} GB")
    print(f"Compression Tier: {COMPRESSION_TIER}")
    print(f"Differential Privacy ε: {DIFFERENTIAL_PRIVACY_EPSILON}")
    print("="*80 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate test report."""
    print("\n" + "="*80)
    print("Load Test Complete")
    print("-"*80)
    
    # Print statistics
    stats = environment.stats
    
    print(f"Total Requests: {stats.total.num_requests}")
    print(f"Total Failures: {stats.total.num_failures}")
    print(f"Failure Rate: {stats.total.fail_ratio:.2%}")
    print(f"Average Response Time: {stats.total.avg_response_time:.2f} ms")
    print(f"Median Response Time: {stats.total.median_response_time:.2f} ms")
    print(f"95% Response Time: {stats.total.get_response_time_percentile(0.95):.2f} ms")
    print(f"99% Response Time: {stats.total.get_response_time_percentile(0.99):.2f} ms")
    
    # Privacy budget statistics
    privacy_stats = privacy_monitor.get_statistics()
    print("\nPrivacy Budget Statistics:")
    print(f"  Total Consumed: {privacy_stats['total_consumed']:.2f}")
    print(f"  Average per User: {privacy_stats['avg_per_user']:.4f}")
    print(f"  Max per User: {privacy_stats['max_per_user']:.4f}")
    print(f"  Violations: {privacy_stats['violations']}")
    
    # SLO validation
    print("\nSLO Validation:")
    p95_target = 500  # ms
    p99_target = 2000  # ms
    availability_target = 0.999
    
    p95_actual = stats.total.get_response_time_percentile(0.95)
    p99_actual = stats.total.get_response_time_percentile(0.99)
    availability_actual = 1 - stats.total.fail_ratio
    
    slo_p95 = p95_actual <= p95_target
    slo_p99 = p99_actual <= p99_target
    slo_availability = availability_actual >= availability_target
    
    print(f"  P95 ≤ {p95_target}ms: {'✓ PASS' if slo_p95 else '✗ FAIL'} ({p95_actual:.2f}ms)")
    print(f"  P99 ≤ {p99_target}ms: {'✓ PASS' if slo_p99 else '✗ FAIL'} ({p99_actual:.2f}ms)")
    print(f"  Availability ≥ {availability_target:.1%}: {'✓ PASS' if slo_availability else '✗ FAIL'} ({availability_actual:.3%})")
    
    # Save detailed results
    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'total_requests': stats.total.num_requests,
        'total_failures': stats.total.num_failures,
        'failure_rate': stats.total.fail_ratio,
        'response_times': {
            'avg': stats.total.avg_response_time,
            'median': stats.total.median_response_time,
            'p95': p95_actual,
            'p99': p99_actual,
        },
        'privacy_budget': privacy_stats,
        'slo_validation': {
            'p95_passed': slo_p95,
            'p99_passed': slo_p99,
            'availability_passed': slo_availability,
        }
    }
    
    with open('load_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to load_test_results.json")
    print("="*80)


if __name__ == '__main__':
    # This file is meant to be run with locust command:
    # locust -f load_test_suite.py --host http://localhost:8000
    print("Run this file with: locust -f load_test_suite.py --host <API_URL>")
    print("Or for headless: locust -f load_test_suite.py --headless -u 100 -r 10 -t 10m")