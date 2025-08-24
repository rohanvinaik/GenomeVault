"""
Prometheus metrics for GenomeVault API.

Comprehensive metrics collection for all API endpoints including:
- Request counts and duration
- HDC encoding metrics
- ZK proof metrics
- PIR query metrics
- Error rates and performance
"""

import time
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter, Histogram, Gauge, Info, Enum,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)

# Create custom registry to avoid conflicts
GENOMEVAULT_REGISTRY = CollectorRegistry()

# HTTP Request Metrics
http_requests_total = Counter(
    'genomevault_http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code', 'component'],
    registry=GENOMEVAULT_REGISTRY
)

http_request_duration_seconds = Histogram(
    'genomevault_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint', 'component'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=GENOMEVAULT_REGISTRY
)

http_request_size_bytes = Histogram(
    'genomevault_http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint'],
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000),
    registry=GENOMEVAULT_REGISTRY
)

http_response_size_bytes = Histogram(
    'genomevault_http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint', 'status_code'],
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000),
    registry=GENOMEVAULT_REGISTRY
)

# HDC (Hyperdimensional Computing) Metrics
hdc_encodings_total = Counter(
    'genomevault_hdc_encodings_total',
    'Total number of HDC encodings created',
    ['dimension_range', 'status'],
    registry=GENOMEVAULT_REGISTRY
)

hdc_encoding_duration_seconds = Histogram(
    'genomevault_hdc_encoding_duration_seconds',
    'Time spent encoding variants to hypervectors',
    ['dimension_range', 'variant_count_range'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=GENOMEVAULT_REGISTRY
)

hdc_comparisons_total = Counter(
    'genomevault_hdc_comparisons_total',
    'Total number of HDC comparisons performed',
    ['metric_type', 'status'],
    registry=GENOMEVAULT_REGISTRY
)

hdc_active_encodings = Gauge(
    'genomevault_hdc_active_encodings',
    'Number of active HDC encodings in memory',
    registry=GENOMEVAULT_REGISTRY
)

# Zero-Knowledge Proof Metrics
zk_proofs_total = Counter(
    'genomevault_zk_proofs_total',
    'Total number of ZK proofs generated',
    ['circuit_type', 'status'],
    registry=GENOMEVAULT_REGISTRY
)

zk_proof_generation_duration_seconds = Histogram(
    'genomevault_zk_proof_generation_duration_seconds',
    'Time spent generating ZK proofs',
    ['circuit_type', 'fallback_used'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    registry=GENOMEVAULT_REGISTRY
)

zk_verifications_total = Counter(
    'genomevault_zk_verifications_total',
    'Total number of ZK proof verifications',
    ['circuit_type', 'result'],
    registry=GENOMEVAULT_REGISTRY
)

zk_verification_duration_seconds = Histogram(
    'genomevault_zk_verification_duration_seconds',
    'Time spent verifying ZK proofs',
    ['circuit_type'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0),
    registry=GENOMEVAULT_REGISTRY
)

# PIR (Private Information Retrieval) Metrics
pir_queries_total = Counter(
    'genomevault_pir_queries_total',
    'Total number of PIR queries executed',
    ['database_size_range', 'servers_used', 'status'],
    registry=GENOMEVAULT_REGISTRY
)

pir_query_duration_seconds = Histogram(
    'genomevault_pir_query_duration_seconds',
    'Time spent executing PIR queries',
    ['database_size_range', 'servers_used'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=GENOMEVAULT_REGISTRY
)

pir_byzantine_failures_total = Counter(
    'genomevault_pir_byzantine_failures_total',
    'Total number of Byzantine failures detected in PIR',
    ['failure_type'],
    registry=GENOMEVAULT_REGISTRY
)

pir_active_databases = Gauge(
    'genomevault_pir_active_databases',
    'Number of active PIR databases',
    registry=GENOMEVAULT_REGISTRY
)

# Database Metrics
database_connections_active = Gauge(
    'genomevault_database_connections_active',
    'Number of active database connections',
    ['database_type'],
    registry=GENOMEVAULT_REGISTRY
)

database_queries_total = Counter(
    'genomevault_database_queries_total',
    'Total number of database queries',
    ['query_type', 'table', 'status'],
    registry=GENOMEVAULT_REGISTRY
)

database_query_duration_seconds = Histogram(
    'genomevault_database_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type', 'table'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=GENOMEVAULT_REGISTRY
)

# Cache Metrics (Redis)
cache_operations_total = Counter(
    'genomevault_cache_operations_total',
    'Total number of cache operations',
    ['operation', 'result'],
    registry=GENOMEVAULT_REGISTRY
)

cache_hit_ratio = Gauge(
    'genomevault_cache_hit_ratio',
    'Cache hit ratio (0-1)',
    registry=GENOMEVAULT_REGISTRY
)

# Rate Limiting Metrics
rate_limit_hits_total = Counter(
    'genomevault_rate_limit_hits_total',
    'Total number of rate limit hits',
    ['limit_type', 'client_type'],
    registry=GENOMEVAULT_REGISTRY
)

rate_limit_current_usage = Gauge(
    'genomevault_rate_limit_current_usage',
    'Current rate limit usage',
    ['limit_type', 'client_id'],
    registry=GENOMEVAULT_REGISTRY
)

# System Metrics
system_info = Info(
    'genomevault_system_info',
    'System information',
    registry=GENOMEVAULT_REGISTRY
)

application_start_time_seconds = Gauge(
    'genomevault_application_start_time_seconds',
    'Application start time in seconds since epoch',
    registry=GENOMEVAULT_REGISTRY
)

active_users = Gauge(
    'genomevault_active_users',
    'Number of active users',
    registry=GENOMEVAULT_REGISTRY
)

# Error Metrics
errors_total = Counter(
    'genomevault_errors_total',
    'Total number of errors',
    ['error_type', 'component', 'severity'],
    registry=GENOMEVAULT_REGISTRY
)

# Business Logic Metrics
genomic_variants_processed_total = Counter(
    'genomevault_genomic_variants_processed_total',
    'Total number of genomic variants processed',
    ['variant_type', 'status'],
    registry=GENOMEVAULT_REGISTRY
)

privacy_operations_total = Counter(
    'genomevault_privacy_operations_total',
    'Total number of privacy-preserving operations',
    ['operation_type', 'privacy_level'],
    registry=GENOMEVAULT_REGISTRY
)


class MetricsCollector:
    """Centralized metrics collector for GenomeVault."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.start_time = time.time()
        application_start_time_seconds.set(self.start_time)
        
        # Initialize system info
        system_info.info({
            'version': '1.0.0',
            'component': 'genomevault-api',
            'environment': 'development',  # Should be configurable
        })
    
    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None,
        component: str = 'api'
    ):
        """Record HTTP request metrics."""
        http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
            component=component
        ).inc()
        
        http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint,
            component=component
        ).observe(duration)
        
        if request_size is not None:
            http_request_size_bytes.labels(
                method=method,
                endpoint=endpoint
            ).observe(request_size)
        
        if response_size is not None:
            http_response_size_bytes.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code)
            ).observe(response_size)
    
    def record_hdc_encoding(
        self,
        dimension: int,
        variant_count: int,
        duration: float,
        status: str = 'success'
    ):
        """Record HDC encoding metrics."""
        dimension_range = self._get_dimension_range(dimension)
        variant_range = self._get_variant_count_range(variant_count)
        
        hdc_encodings_total.labels(
            dimension_range=dimension_range,
            status=status
        ).inc()
        
        hdc_encoding_duration_seconds.labels(
            dimension_range=dimension_range,
            variant_count_range=variant_range
        ).observe(duration)
        
        # Track genomic variants processed
        genomic_variants_processed_total.labels(
            variant_type='mixed',
            status=status
        ).inc(variant_count)
    
    def record_hdc_comparison(
        self,
        metric_type: str,
        duration: float,
        status: str = 'success'
    ):
        """Record HDC comparison metrics."""
        hdc_comparisons_total.labels(
            metric_type=metric_type,
            status=status
        ).inc()
    
    def record_zk_proof_generation(
        self,
        circuit_type: str,
        duration: float,
        fallback_used: bool = False,
        status: str = 'success'
    ):
        """Record ZK proof generation metrics."""
        zk_proofs_total.labels(
            circuit_type=circuit_type,
            status=status
        ).inc()
        
        zk_proof_generation_duration_seconds.labels(
            circuit_type=circuit_type,
            fallback_used=str(fallback_used)
        ).observe(duration)
        
        # Track privacy operations
        privacy_operations_total.labels(
            operation_type='zk_proof_generation',
            privacy_level='high'
        ).inc()
    
    def record_zk_verification(
        self,
        circuit_type: str,
        duration: float,
        result: str = 'valid'
    ):
        """Record ZK proof verification metrics."""
        zk_verifications_total.labels(
            circuit_type=circuit_type,
            result=result
        ).inc()
        
        zk_verification_duration_seconds.labels(
            circuit_type=circuit_type
        ).observe(duration)
    
    def record_pir_query(
        self,
        database_size: int,
        servers_used: int,
        duration: float,
        status: str = 'success'
    ):
        """Record PIR query metrics."""
        db_size_range = self._get_database_size_range(database_size)
        
        pir_queries_total.labels(
            database_size_range=db_size_range,
            servers_used=str(servers_used),
            status=status
        ).inc()
        
        pir_query_duration_seconds.labels(
            database_size_range=db_size_range,
            servers_used=str(servers_used)
        ).observe(duration)
        
        # Track privacy operations
        privacy_operations_total.labels(
            operation_type='pir_query',
            privacy_level='high'
        ).inc()
    
    def record_pir_byzantine_failure(self, failure_type: str):
        """Record PIR Byzantine failure."""
        pir_byzantine_failures_total.labels(
            failure_type=failure_type
        ).inc()
    
    def record_database_query(
        self,
        query_type: str,
        table: str,
        duration: float,
        status: str = 'success'
    ):
        """Record database query metrics."""
        database_queries_total.labels(
            query_type=query_type,
            table=table,
            status=status
        ).inc()
        
        database_query_duration_seconds.labels(
            query_type=query_type,
            table=table
        ).observe(duration)
    
    def record_cache_operation(
        self,
        operation: str,
        result: str = 'hit'
    ):
        """Record cache operation metrics."""
        cache_operations_total.labels(
            operation=operation,
            result=result
        ).inc()
    
    def record_rate_limit_hit(
        self,
        limit_type: str,
        client_type: str = 'api'
    ):
        """Record rate limit hit."""
        rate_limit_hits_total.labels(
            limit_type=limit_type,
            client_type=client_type
        ).inc()
    
    def record_error(
        self,
        error_type: str,
        component: str,
        severity: str = 'error'
    ):
        """Record error metrics."""
        errors_total.labels(
            error_type=error_type,
            component=component,
            severity=severity
        ).inc()
    
    def update_active_encodings(self, count: int):
        """Update active HDC encodings count."""
        hdc_active_encodings.set(count)
    
    def update_active_databases(self, count: int):
        """Update active PIR databases count."""
        pir_active_databases.set(count)
    
    def update_database_connections(self, database_type: str, count: int):
        """Update database connections count."""
        database_connections_active.labels(
            database_type=database_type
        ).set(count)
    
    def update_cache_hit_ratio(self, ratio: float):
        """Update cache hit ratio."""
        cache_hit_ratio.set(ratio)
    
    def update_active_users(self, count: int):
        """Update active users count."""
        active_users.set(count)
    
    @staticmethod
    def _get_dimension_range(dimension: int) -> str:
        """Get dimension range label."""
        if dimension < 1000:
            return "0-1k"
        elif dimension < 5000:
            return "1k-5k"
        elif dimension < 10000:
            return "5k-10k"
        elif dimension < 50000:
            return "10k-50k"
        else:
            return "50k+"
    
    @staticmethod
    def _get_variant_count_range(count: int) -> str:
        """Get variant count range label."""
        if count < 10:
            return "0-10"
        elif count < 100:
            return "10-100"
        elif count < 1000:
            return "100-1k"
        elif count < 10000:
            return "1k-10k"
        else:
            return "10k+"
    
    @staticmethod
    def _get_database_size_range(size: int) -> str:
        """Get database size range label."""
        if size < 100:
            return "0-100"
        elif size < 1000:
            return "100-1k"
        elif size < 10000:
            return "1k-10k"
        elif size < 100000:
            return "10k-100k"
        else:
            return "100k+"


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_prometheus_metrics() -> str:
    """Get Prometheus metrics in text format."""
    return generate_latest(GENOMEVAULT_REGISTRY).decode('utf-8')


def get_metrics_content_type() -> str:
    """Get the content type for Prometheus metrics."""
    return CONTENT_TYPE_LATEST