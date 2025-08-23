"""Pact configuration for contract testing."""

from __future__ import annotations

import os
from typing import Dict, Any
from pact import Consumer, Provider
from pact.broker_client import PactBroker


# Pact configuration
PACT_CONFIG = {
    "consumer_name": "genomevault-client",
    "provider_name": "genomevault-api",
    "pact_dir": os.path.join(os.path.dirname(__file__), "pacts"),
    "log_dir": os.path.join(os.path.dirname(__file__), "logs"),
    "log_level": "INFO",
}

# Pact Broker configuration
PACT_BROKER_CONFIG = {
    "broker_base_url": os.getenv("PACT_BROKER_URL", "http://localhost:9292"),
    "broker_username": os.getenv("PACT_BROKER_USERNAME"),
    "broker_password": os.getenv("PACT_BROKER_PASSWORD"),
    "consumer_version": os.getenv("CONSUMER_VERSION", "1.0.0"),
    "provider_version": os.getenv("PROVIDER_VERSION", "1.0.0"),
}

# Base URL for the provider
PROVIDER_BASE_URL = os.getenv("PROVIDER_BASE_URL", "http://localhost:8000")


def create_pact_consumer() -> Consumer:
    """Create a Pact consumer for contract testing."""
    return Consumer(
        consumer=PACT_CONFIG["consumer_name"],
        provider=PACT_CONFIG["provider_name"],
        pact_dir=PACT_CONFIG["pact_dir"],
        log_dir=PACT_CONFIG["log_dir"],
        log_level=PACT_CONFIG["log_level"],
    )


def create_pact_broker() -> PactBroker:
    """Create a Pact broker client for publishing contracts."""
    return PactBroker(
        broker_base_url=PACT_BROKER_CONFIG["broker_base_url"],
        broker_username=PACT_BROKER_CONFIG["broker_username"],
        broker_password=PACT_BROKER_CONFIG["broker_password"],
    )


# Common request/response matchers
COMMON_MATCHERS = {
    "uuid": {
        "match": "regex",
        "regex": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    },
    "iso_datetime": {"match": "regex", "regex": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z?$"},
    "genomic_coordinate": {"match": "regex", "regex": r"^(chr)?([1-9]|1[0-9]|2[0-2]|X|Y|M):\d+$"},
    "chromosome": {"match": "regex", "regex": r"^(chr)?([1-9]|1[0-9]|2[0-2]|X|Y|M)$"},
    "nucleotide_sequence": {"match": "regex", "regex": r"^[ATCGN]+$"},
    "base64": {"match": "regex", "regex": r"^[A-Za-z0-9+/]*={0,2}$"},
    "sha256": {"match": "regex", "regex": r"^[a-f0-9]{64}$"},
    "positive_integer": {"match": "type", "value": 1},
    "positive_number": {"match": "type", "value": 1.0},
    "boolean": {"match": "type", "value": True},
    "array": {"match": "type", "value": []},
    "object": {"match": "type", "value": {}},
}

# Standard headers for all requests
STANDARD_HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": "test-api-key",
    "User-Agent": "genomevault-client/1.0.0",
}

# Rate limit response headers
RATE_LIMIT_HEADERS = {
    "X-RateLimit-Limit": {"match": "type", "value": "1000"},
    "X-RateLimit-Remaining": {"match": "type", "value": "999"},
    "X-RateLimit-Reset": {"match": "type", "value": "1642248600"},
}

# Error response template
ERROR_RESPONSE_TEMPLATE = {
    "type": {"match": "type", "value": "ValidationError"},
    "code": {"match": "regex", "regex": r"^GV_[A-Z_]+$"},
    "message": {"match": "type", "value": "Error message"},
    "details": COMMON_MATCHERS["object"],
    "request_id": COMMON_MATCHERS["uuid"],
    "timestamp": COMMON_MATCHERS["iso_datetime"],
}


def create_genomic_variant_matcher() -> Dict[str, Any]:
    """Create matcher for genomic variant objects."""
    return {
        "chrom": COMMON_MATCHERS["chromosome"],
        "pos": COMMON_MATCHERS["positive_integer"],
        "ref": COMMON_MATCHERS["nucleotide_sequence"],
        "alt": COMMON_MATCHERS["nucleotide_sequence"],
        "impact": {
            "match": "regex",
            "regex": r"^(missense|nonsense|synonymous|frameshift|splice_site|intron|intergenic)$",
        },
        "quality": {"match": "decimal", "value": 99.5},
    }


def create_hypervector_response_matcher() -> Dict[str, Any]:
    """Create matcher for hypervector encoding responses."""
    return {
        "dim": COMMON_MATCHERS["positive_integer"],
        "binary": COMMON_MATCHERS["boolean"],
        "vector": {
            "match": "type",
            "value": [0.1, -0.2, 0.3],  # Array of numbers
        },
        "privacy_level": {
            "match": "regex",
            "regex": r"^(k-anonymous|differential_private|information_theoretic)$",
        },
        "compression_ratio": COMMON_MATCHERS["positive_number"],
    }


def create_pir_response_matcher() -> Dict[str, Any]:
    """Create matcher for PIR query responses."""
    return {
        "index": COMMON_MATCHERS["positive_integer"],
        "item_base64": COMMON_MATCHERS["base64"],
        "privacy_proof": {"match": "type", "value": "proof_string"},
        "query_time_ms": COMMON_MATCHERS["positive_integer"],
    }


def create_clinical_response_matcher() -> Dict[str, Any]:
    """Create matcher for clinical analysis responses."""
    return {
        "analysis_id": COMMON_MATCHERS["uuid"],
        "risk_score": {"match": "decimal", "value": 0.85, "min": 0.0, "max": 1.0},
        "confidence_interval": {"match": "type", "value": [0.78, 0.92]},
        "recommendations": {"match": "type", "value": ["recommendation1", "recommendation2"]},
        "audit_trail_hash": COMMON_MATCHERS["sha256"],
        "differential_privacy_epsilon": COMMON_MATCHERS["positive_number"],
    }
