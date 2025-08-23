"""Pact contract tests for Hypervector API."""

from __future__ import annotations

import pytest
import requests
from pact import Consumer, Provider

from tests.contract.pact_config import (
    create_pact_consumer,
    STANDARD_HEADERS,
    RATE_LIMIT_HEADERS,
    ERROR_RESPONSE_TEMPLATE,
    create_genomic_variant_matcher,
    create_hypervector_response_matcher,
)


class TestHypervectorContract:
    """Contract tests for Hypervector encoding endpoints."""

    @pytest.fixture(scope="class")
    def pact(self):
        """Set up Pact consumer for hypervector tests."""
        consumer = create_pact_consumer()
        yield consumer
        consumer.stop()

    def test_encode_variants_success(self, pact):
        """Test successful encoding of genomic variants."""
        # Expected request
        request_body = {
            "variants": [
                {
                    "chrom": "1",
                    "pos": 1234567,
                    "ref": "A",
                    "alt": "T",
                    "impact": "missense",
                    "quality": 99.5,
                }
            ],
            "dim": 8192,
            "binary": False,
        }

        # Expected response
        response_matcher = create_hypervector_response_matcher()

        (
            pact.given("Valid genomic variants are provided")
            .upon_receiving("A request to encode variants")
            .with_request(
                method="POST", path="/v1/hv/encode", headers=STANDARD_HEADERS, body=request_body
            )
            .will_respond_with(
                status=200,
                headers={"Content-Type": "application/json", **RATE_LIMIT_HEADERS},
                body=response_matcher,
            )
        )

        with pact:
            response = requests.post(
                f"{pact.uri}/v1/hv/encode", json=request_body, headers=STANDARD_HEADERS
            )

            assert response.status_code == 200
            data = response.json()
            assert "dim" in data
            assert "vector" in data
            assert "privacy_level" in data

    def test_encode_numeric_success(self, pact):
        """Test successful encoding of numeric features."""
        request_body = {"numeric": [0.1, 0.8, 0.3, 0.9, 0.2], "dim": 4096, "binary": True}

        response_matcher = create_hypervector_response_matcher()
        response_matcher["binary"] = {"match": "type", "value": True}

        (
            pact.given("Valid numeric features are provided")
            .upon_receiving("A request to encode numeric features")
            .with_request(
                method="POST", path="/v1/hv/encode", headers=STANDARD_HEADERS, body=request_body
            )
            .will_respond_with(
                status=200,
                headers={"Content-Type": "application/json", **RATE_LIMIT_HEADERS},
                body=response_matcher,
            )
        )

        with pact:
            response = requests.post(
                f"{pact.uri}/v1/hv/encode", json=request_body, headers=STANDARD_HEADERS
            )

            assert response.status_code == 200
            data = response.json()
            assert data["binary"] is True

    def test_encode_validation_error(self, pact):
        """Test validation error when no data is provided."""
        request_body = {
            "dim": 8192,
            "binary": False,
            # Missing both 'numeric' and 'variants'
        }

        (
            pact.given("No encoding data is provided")
            .upon_receiving("A request with missing encoding data")
            .with_request(
                method="POST", path="/v1/hv/encode", headers=STANDARD_HEADERS, body=request_body
            )
            .will_respond_with(
                status=400,
                headers={"Content-Type": "application/json", **RATE_LIMIT_HEADERS},
                body={
                    **ERROR_RESPONSE_TEMPLATE,
                    "code": {"match": "type", "value": "GV_VALIDATION_ERROR"},
                    "message": {"match": "type", "value": "Provide either 'numeric' or 'variants'"},
                },
            )
        )

        with pact:
            response = requests.post(
                f"{pact.uri}/v1/hv/encode", json=request_body, headers=STANDARD_HEADERS
            )

            assert response.status_code == 400
            data = response.json()
            assert data["code"] == "GV_VALIDATION_ERROR"

    def test_encode_invalid_variant_format(self, pact):
        """Test error for invalid variant format."""
        request_body = {
            "variants": [
                {
                    "chrom": "invalid_chromosome",
                    "pos": -1,  # Invalid position
                    "ref": "123",  # Invalid nucleotide
                    "alt": "XYZ",  # Invalid nucleotide
                    "impact": "unknown_impact",
                }
            ],
            "dim": 8192,
        }

        (
            pact.given("Invalid variant format is provided")
            .upon_receiving("A request with invalid variant data")
            .with_request(
                method="POST", path="/v1/hv/encode", headers=STANDARD_HEADERS, body=request_body
            )
            .will_respond_with(
                status=422,
                headers={"Content-Type": "application/json", **RATE_LIMIT_HEADERS},
                body={
                    **ERROR_RESPONSE_TEMPLATE,
                    "type": {"match": "type", "value": "ValidationError"},
                    "code": {"match": "type", "value": "GV_VALIDATION_ERROR"},
                    "errors": {
                        "match": "type",
                        "value": [
                            {
                                "field": "variants.0.chrom",
                                "message": "Invalid genomic coordinate format",
                                "code": "GV_VALIDATION_ERROR",
                            }
                        ],
                    },
                },
            )
        )

        with pact:
            response = requests.post(
                f"{pact.uri}/v1/hv/encode", json=request_body, headers=STANDARD_HEADERS
            )

            assert response.status_code == 422
            data = response.json()
            assert "errors" in data
            assert len(data["errors"]) > 0

    def test_encode_rate_limit_exceeded(self, pact):
        """Test rate limit exceeded response."""
        request_body = {"numeric": [0.1, 0.2, 0.3], "dim": 1024}

        (
            pact.given("Rate limit has been exceeded")
            .upon_receiving("A request when rate limited")
            .with_request(
                method="POST", path="/v1/hv/encode", headers=STANDARD_HEADERS, body=request_body
            )
            .will_respond_with(
                status=429,
                headers={
                    "Content-Type": "application/json",
                    "Retry-After": "3600",
                    **RATE_LIMIT_HEADERS,
                },
                body={
                    **ERROR_RESPONSE_TEMPLATE,
                    "type": {"match": "type", "value": "RateLimitError"},
                    "code": {"match": "type", "value": "GV_RATE_LIMITED"},
                    "details": {"match": "type", "value": {"retry_after": "3600", "limit": "1000"}},
                },
            )
        )

        with pact:
            response = requests.post(
                f"{pact.uri}/v1/hv/encode", json=request_body, headers=STANDARD_HEADERS
            )

            assert response.status_code == 429
            assert "Retry-After" in response.headers
            data = response.json()
            assert data["code"] == "GV_RATE_LIMITED"
