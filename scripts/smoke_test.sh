#!/usr/bin/env bash

################################################################################
# Smoke Test Script for GenomeVault
#
# Quick validation that all services are running and basic operations work.
# Should complete in under 30 seconds.
#
# Exit codes:
#   0 - All tests passed
#   1 - One or more tests failed
################################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
TIMEOUT="${TIMEOUT:-5}"
MAX_RETRIES="${MAX_RETRIES:-3}"

# Test results
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Log file for capturing errors
LOG_FILE="${TMPDIR:-${TMPDIR:-/tmp}}/genomevault_smoke_test_$(date +%Y%m%d_%H%M%S).log"

# Timer
START_TIME=$(date +%s)

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

test_passed() {
    local test_name="$1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    echo -e "${GREEN}✓${NC} $test_name" | tee -a "$LOG_FILE"
}

test_failed() {
    local test_name="$1"
    local error_msg="${2:-Unknown error}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    echo -e "${RED}✗${NC} $test_name: $error_msg" | tee -a "$LOG_FILE"
}

run_test() {
    local test_name="$1"
    TESTS_RUN=$((TESTS_RUN + 1))
    echo -n "Testing $test_name... "
}

check_service() {
    local service_name="$1"
    local port="$2"

    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

make_request() {
    local method="$1"
    local endpoint="$2"
    local data="${3:-}"
    local expected_status="${4:-200}"

    local curl_opts="-s -w \n%{http_code} -X $method"
    curl_opts="$curl_opts --connect-timeout $TIMEOUT --max-time $((TIMEOUT * 2))"

    if [[ -n "$data" ]]; then
        curl_opts="$curl_opts -H 'Content-Type: application/json' -d '$data'"
    fi

    local response
    response=$(eval "curl $curl_opts '$API_BASE_URL$endpoint'" 2>&1)
    local status_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | head -n-1)

    if [[ "$status_code" == "$expected_status" ]]; then
        echo "$body"
        return 0
    else
        echo "Expected $expected_status, got $status_code" >&2
        echo "$body" >&2
        return 1
    fi
}

check_logs_for_errors() {
    local log_file="$1"
    local error_count=0

    if [[ -f "$log_file" ]]; then
        # Check for common error patterns
        error_count=$(grep -Ei "(error|exception|fatal|critical)" "$log_file" 2>/dev/null | \
                     grep -Ev "(0 errors|no error|error_count.*0)" | \
                     wc -l)
    fi

    echo "$error_count"
}

################################################################################
# Tests
################################################################################

# Header
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧬 GenomeVault Smoke Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "API URL: $API_BASE_URL"
echo "Timeout: ${TIMEOUT}s"
echo "Log: $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Test 1: Check services
echo "1. CHECKING SERVICES"
echo "────────────────────────────────"

run_test "API Server (port 8000)"
if check_service "API" 8000; then
    test_passed "API Server"
else
    test_failed "API Server" "Not listening on port 8000"
fi

run_test "Redis (port 6379)"
if check_service "Redis" 6379 || redis-cli ping >/dev/null 2>&1; then
    test_passed "Redis"
else
    log_warning "Redis not available (rate limiting disabled)"
fi

run_test "PostgreSQL (port 5432)"
if check_service "PostgreSQL" 5432; then
    test_passed "PostgreSQL"
else
    log_warning "PostgreSQL not available (using SQLite fallback)"
fi

echo

# Test 2: Health check endpoint
echo "2. HEALTH CHECK"
echo "────────────────────────────────"

run_test "/api/healthz endpoint"
if health_response=$(make_request GET /api/healthz); then
    if echo "$health_response" | grep -q "healthy\|ok\|UP"; then
        test_passed "Health check"
    else
        test_failed "Health check" "Unexpected response: $health_response"
    fi
else
    test_failed "Health check" "Request failed"
fi

echo

# Test 3: API documentation
echo "3. API DOCUMENTATION"
echo "────────────────────────────────"

run_test "/api/docs accessibility"
if docs_response=$(make_request GET /api/docs "" 200 2>/dev/null || make_request GET /docs "" 200 2>/dev/null); then
    if echo "$docs_response" | grep -q "swagger-ui\|openapi\|GenomeVault"; then
        test_passed "API documentation"
    else
        test_failed "API documentation" "Page not found or invalid"
    fi
else
    test_failed "API documentation" "Cannot access /api/docs"
fi

echo

# Test 4: HDC Encoding
echo "4. HDC ENCODING TEST"
echo "────────────────────────────────"

run_test "HDC variant encoding"
hdc_data='{
    "variants": [{
        "chromosome": "1",
        "position": 12345,
        "ref": "A",
        "alt": "G",
        "quality": 30
    }],
    "dimension": 5000,
    "normalize": true
}'

if hdc_response=$(make_request POST /api/hdc/encode "$hdc_data"); then
    if echo "$hdc_response" | grep -q "encoding_id"; then
        encoding_id=$(echo "$hdc_response" | grep -o '"encoding_id":"[^"]*"' | cut -d'"' -f4)
        test_passed "HDC encoding (ID: ${encoding_id:0:12}...)"
    else
        test_failed "HDC encoding" "No encoding_id in response"
    fi
else
    test_failed "HDC encoding" "Request failed"
fi

echo

# Test 5: ZK Proof
echo "5. ZERO-KNOWLEDGE PROOF TEST"
echo "────────────────────────────────"

run_test "ZK proof generation"
zk_data='{
    "circuit_name": "sum64",
    "inputs": [
        {"name": "a", "value": 10, "is_public": false},
        {"name": "b", "value": 20, "is_public": false},
        {"name": "c", "value": 30, "is_public": true}
    ],
    "store_proof": false
}'

if zk_response=$(make_request POST /api/zk/prove "$zk_data"); then
    if echo "$zk_response" | grep -q "proof_id"; then
        proof_id=$(echo "$zk_response" | grep -o '"proof_id":"[^"]*"' | cut -d'"' -f4)
        test_passed "ZK proof generation (ID: ${proof_id:0:12}...)"
    else
        test_failed "ZK proof" "No proof_id in response"
    fi
else
    test_failed "ZK proof" "Request failed"
fi

echo

# Test 6: PIR Status
echo "6. PIR SYSTEM TEST"
echo "────────────────────────────────"

run_test "PIR system status"
if pir_response=$(make_request GET /api/pir/status); then
    if echo "$pir_response" | grep -q "status"; then
        test_passed "PIR system status"
    else
        test_failed "PIR status" "Invalid response format"
    fi
else
    test_failed "PIR status" "Request failed"
fi

echo

# Test 7: Check circuit list
echo "7. ZK CIRCUITS LIST"
echo "────────────────────────────────"

run_test "ZK circuits listing"
if circuits_response=$(make_request GET /api/zk/circuits); then
    circuit_count=$(echo "$circuits_response" | grep -o '"name"' | wc -l)
    if [[ $circuit_count -gt 0 ]]; then
        test_passed "ZK circuits ($circuit_count available)"
    else
        test_failed "ZK circuits" "No circuits found"
    fi
else
    test_failed "ZK circuits" "Request failed"
fi

echo

# Test 8: Check logs for errors
echo "8. LOG ANALYSIS"
echo "────────────────────────────────"

run_test "Application logs"

# Check for GenomeVault log files
app_log="${GENOMEVAULT_LOGS:-/var/log/genomevault}/app.log"
if [[ ! -f "$app_log" ]]; then
    app_log="./genomevault.log"
fi

if [[ -f "$app_log" ]]; then
    error_count=$(check_logs_for_errors "$app_log")
    if [[ $error_count -eq 0 ]]; then
        test_passed "No errors in logs"
    else
        test_failed "Log errors" "$error_count errors found"
        log_warning "Recent errors:"
        tail -n 20 "$app_log" | grep -Ei "(error|exception)" | head -5 | while read line; do
            echo "  $line" | tee -a "$LOG_FILE"
        done
    fi
else
    log_warning "Application log not found"
fi

echo

# Test 9: Response times
echo "9. PERFORMANCE CHECK"
echo "────────────────────────────────"

run_test "API response times"

total_time=0
endpoints=("/api/healthz" "/api/zk/circuits" "/api/pir/status")
slow_count=0

for endpoint in "${endpoints[@]}"; do
    start=$(date +%s%N)
    if make_request GET "$endpoint" >/dev/null 2>&1; then
        end=$(date +%s%N)
        elapsed=$((($end - $start) / 1000000))  # Convert to milliseconds

        if [[ $elapsed -gt 1000 ]]; then
            slow_count=$((slow_count + 1))
            log_warning "$endpoint: ${elapsed}ms (slow)"
        else
            log_info "$endpoint: ${elapsed}ms"
        fi

        total_time=$((total_time + elapsed))
    fi
done

avg_time=$((total_time / ${#endpoints[@]}))
if [[ $slow_count -eq 0 ]]; then
    test_passed "Response times OK (avg: ${avg_time}ms)"
else
    test_failed "Response times" "$slow_count slow endpoints"
fi

echo

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Tests run:    $TESTS_RUN"
echo "Tests passed: $TESTS_PASSED"
echo "Tests failed: $TESTS_FAILED"
echo "Time elapsed: ${ELAPSED}s"
echo "Log file:     $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Final result
if [[ $TESTS_FAILED -eq 0 ]]; then
    echo -e "\n${GREEN}✅ All smoke tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}❌ $TESTS_FAILED test(s) failed${NC}"
    echo -e "\nCheck the log file for details: $LOG_FILE"
    exit 1
fi
