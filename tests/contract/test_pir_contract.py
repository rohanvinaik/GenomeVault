"""Pact contract tests for PIR API."""

from __future__ import annotations

import pytest
import requests

from tests.contract.pact_config import (
    create_pact_consumer,
    STANDARD_HEADERS,
    RATE_LIMIT_HEADERS,
    ERROR_RESPONSE_TEMPLATE,
    create_pir_response_matcher,
    COMMON_MATCHERS,
)


class TestPIRContract:
    """Contract tests for PIR query endpoints."""

    @pytest.fixture(scope="class")
    def pact(self):
        """Set up Pact consumer for PIR tests."""
        consumer = create_pact_consumer()
        yield consumer
        consumer.stop()

    def test_pir_query_success(self, pact):
        """Test successful PIR query execution."""
        request_body = {
            "index": 42,
            "query_id": "550e8400-e29b-41d4-a716-446655440000",
            "timeout_seconds": 30,
        }

        response_matcher = create_pir_response_matcher()

        (
            pact.given("PIR database is available")
            .upon_receiving("A valid PIR query request")
            .with_request(
                method="POST", path="/v1/pir/query", headers=STANDARD_HEADERS, body=request_body
            )
            .will_respond_with(
                status=200,
                headers={"Content-Type": "application/json", **RATE_LIMIT_HEADERS},
                body=response_matcher,
            )
        )

        with pact:
            response = requests.post(
                f"{pact.uri}/v1/pir/query", json=request_body, headers=STANDARD_HEADERS
            )

            assert response.status_code == 200
            data = response.json()
            assert "index" in data
            assert "item_base64" in data
            assert "privacy_proof" in data
            assert data["index"] == request_body["index"]

    def test_pir_query_invalid_index(self, pact):
        """Test PIR query with invalid index."""
        request_body = {
            "index": -1,  # Invalid negative index
            "query_id": "550e8400-e29b-41d4-a716-446655440000",
        }

        (
            pact.given("Invalid PIR query index is provided")
            .upon_receiving("A PIR query with negative index")
            .with_request(
                method="POST", path="/v1/pir/query", headers=STANDARD_HEADERS, body=request_body
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
                                "field": "index",
                                "message": "Index must be non-negative",
                                "code": "GV_VALIDATION_ERROR",
                            }
                        ],
                    },
                },
            )
        )

        with pact:
            response = requests.post(
                f"{pact.uri}/v1/pir/query", json=request_body, headers=STANDARD_HEADERS
            )

            assert response.status_code == 422
            data = response.json()
            assert "errors" in data

    def test_pir_query_index_out_of_range(self, pact):
        """Test PIR query with index out of database range."""
        request_body = {
            "index": 999999,  # Index beyond database size
            "query_id": "550e8400-e29b-41d4-a716-446655440000",
        }

        (
            pact.given("PIR query index is out of range")
            .upon_receiving("A PIR query with out-of-range index")
            .with_request(
                method="POST", path="/v1/pir/query", headers=STANDARD_HEADERS, body=request_body
            )
            .will_respond_with(
                status=400,
                headers={"Content-Type": "application/json", **RATE_LIMIT_HEADERS},
                body={
                    **ERROR_RESPONSE_TEMPLATE,
                    "type": {"match": "type", "value": "BadRequestError"},
                    "code": {"match": "type", "value": "GV_PIR_QUERY_FAILED"},
                    "message": {"match": "type", "value": "Query index out of database range"},
                },
            )
        )

        with pact:
            response = requests.post(
                f"{pact.uri}/v1/pir/query", json=request_body, headers=STANDARD_HEADERS
            )

            assert response.status_code == 400
            data = response.json()
            assert data["code"] == "GV_PIR_QUERY_FAILED"

    def test_pir_query_timeout(self, pact):
        """Test PIR query timeout handling."""
        request_body = {
            "index": 10,
            "query_id": "550e8400-e29b-41d4-a716-446655440000",
            "timeout_seconds": 1,  # Very short timeout
        }

        (
            pact.given("PIR query will timeout")
            .upon_receiving("A PIR query that times out")
            .with_request(
                method="POST", path="/v1/pir/query", headers=STANDARD_HEADERS, body=request_body
            )
            .will_respond_with(
                status=408,
                headers={"Content-Type": "application/json", **RATE_LIMIT_HEADERS},
                body={
                    **ERROR_RESPONSE_TEMPLATE,
                    "type": {"match": "type", "value": "TimeoutError"},
                    "code": {"match": "type", "value": "GV_PIR_QUERY_FAILED"},
                    "message": {"match": "type", "value": "PIR query timed out"},
                    "details": {
                        "match": "type",
                        "value": {"timeout_seconds": 1, "suggested_timeout": 30},
                    },
                },
            )
        )

        with pact:
            response = requests.post(
                f"{pact.uri}/v1/pir/query", json=request_body, headers=STANDARD_HEADERS
            )

            assert response.status_code == 408
            data = response.json()
            assert "timeout" in data["message"].lower()

    def test_pir_query_missing_required_field(self, pact):
        """Test PIR query with missing required field."""
        request_body = {
            # Missing required 'index' field
            "query_id": "550e8400-e29b-41d4-a716-446655440000"
        }

        (
            pact.given("PIR query is missing required field")
            .upon_receiving("A PIR query without index")
            .with_request(
                method="POST", path="/v1/pir/query", headers=STANDARD_HEADERS, body=request_body
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
                                "field": "index",
                                "message": "Field required",
                                "code": "GV_VALIDATION_ERROR",
                            }
                        ],
                    },
                },
            )
        )

        with pact:
            response = requests.post(
                f"{pact.uri}/v1/pir/query", json=request_body, headers=STANDARD_HEADERS
            )

            assert response.status_code == 422
            data = response.json()
            assert any(error["field"] == "index" for error in data["errors"])

    def test_pir_service_unavailable(self, pact):
        """Test PIR service unavailable scenario."""
        request_body = {"index": 5, "query_id": "550e8400-e29b-41d4-a716-446655440000"}

        (
            pact.given("PIR service is unavailable")
            .upon_receiving("A PIR query when service is down")
            .with_request(
                method="POST", path="/v1/pir/query", headers=STANDARD_HEADERS, body=request_body
            )
            .will_respond_with(
                status=503,
                headers={
                    "Content-Type": "application/json",
                    "Retry-After": "60",
                    **RATE_LIMIT_HEADERS,
                },
                body={
                    **ERROR_RESPONSE_TEMPLATE,
                    "type": {"match": "type", "value": "ServiceUnavailableError"},
                    "code": {"match": "type", "value": "GV_SERVICE_UNAVAILABLE"},
                    "message": {"match": "type", "value": "PIR service temporarily unavailable"},
                    "details": {
                        "match": "type",
                        "value": {"retry_after": "60", "service": "pir_engine"},
                    },
                },
            )
        )

        with pact:
            response = requests.post(
                f"{pact.uri}/v1/pir/query", json=request_body, headers=STANDARD_HEADERS
            )

            assert response.status_code == 503
            assert "Retry-After" in response.headers
            data = response.json()
            assert data["code"] == "GV_SERVICE_UNAVAILABLE"
