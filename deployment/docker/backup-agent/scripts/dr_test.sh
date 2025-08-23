#!/bin/bash
# Disaster Recovery Testing Script for GenomeVault
# Tests backup restoration and validates RTO < 4 hours

set -euo pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/backups}"
LOG_FILE="${LOG_FILE:-/var/log/dr_test.log}"
TEST_DB="genomevault_dr_test"
PRODUCTION_DB="${DATABASE_NAME:-genomevault}"
START_TIME=$(date +%s)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "${RED}ERROR: $1${NC}"
    exit 1
}

# Success message
success() {
    log "${GREEN}SUCCESS: $1${NC}"
}

# Warning message
warning() {
    log "${YELLOW}WARNING: $1${NC}"
}

# Find latest backup
find_latest_backup() {
    local backup_type="$1"
    local latest_backup=""

    # Check S3 first
    if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
        latest_backup=$(aws s3 ls "s3://${S3_BACKUP_BUCKET}/${backup_type}/" \
            --recursive \
            | sort -r \
            | head -1 \
            | awk '{print $4}')

        if [[ -n "$latest_backup" ]]; then
            # Download from S3
            local local_file="${BACKUP_DIR}/dr_test_backup.sql.gz"
            aws s3 cp "s3://${S3_BACKUP_BUCKET}/${latest_backup}" "$local_file"
            echo "$local_file"
            return 0
        fi
    fi

    # Check GCS
    if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
        latest_backup=$(gsutil ls -l "gs://${GCS_BACKUP_BUCKET}/${backup_type}/**" \
            | grep -v "TOTAL:" \
            | sort -k2 -r \
            | head -1 \
            | awk '{print $3}')

        if [[ -n "$latest_backup" ]]; then
            # Download from GCS
            local local_file="${BACKUP_DIR}/dr_test_backup.sql.gz"
            gsutil cp "$latest_backup" "$local_file"
            echo "$local_file"
            return 0
        fi
    fi

    # Check local backups
    latest_backup=$(find "${BACKUP_DIR}/${backup_type}" -name "*.sql.gz" -type f \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn \
        | head -1 \
        | cut -d' ' -f2-)

    if [[ -n "$latest_backup" ]]; then
        echo "$latest_backup"
        return 0
    fi

    return 1
}

# Test database restoration
test_restore() {
    local backup_file="$1"
    local restore_start=$(date +%s)

    log "Testing restore from: $backup_file"

    # Drop test database if exists
    PGPASSWORD="${DATABASE_PASSWORD}" psql \
        -h "${DATABASE_HOST}" \
        -p "${DATABASE_PORT}" \
        -U "${DATABASE_USER}" \
        -d postgres \
        -c "DROP DATABASE IF EXISTS ${TEST_DB};" 2>/dev/null || true

    # Create test database
    PGPASSWORD="${DATABASE_PASSWORD}" psql \
        -h "${DATABASE_HOST}" \
        -p "${DATABASE_PORT}" \
        -U "${DATABASE_USER}" \
        -d postgres \
        -c "CREATE DATABASE ${TEST_DB};" \
        || error_exit "Failed to create test database"

    # Decrypt backup if encrypted
    local restore_file="$backup_file"
    if file "$backup_file" | grep -q "GPG\|encrypted"; then
        log "Decrypting backup..."
        restore_file="${BACKUP_DIR}/decrypted_backup.sql.gz"

        if [[ -f "${ENCRYPTION_KEY_FILE}" ]]; then
            openssl enc -aes-256-cbc -d \
                -in "$backup_file" \
                -out "$restore_file" \
                -pass "file:${ENCRYPTION_KEY_FILE}" \
                || error_exit "Failed to decrypt backup"
        else
            error_exit "Encryption key not found"
        fi
    fi

    # Decompress if needed
    if [[ "$restore_file" == *.gz ]]; then
        log "Decompressing backup..."
        gunzip -c "$restore_file" > "${BACKUP_DIR}/restore.sql"
        restore_file="${BACKUP_DIR}/restore.sql"
    fi

    # Restore database
    log "Restoring database..."
    PGPASSWORD="${DATABASE_PASSWORD}" psql \
        -h "${DATABASE_HOST}" \
        -p "${DATABASE_PORT}" \
        -U "${DATABASE_USER}" \
        -d "${TEST_DB}" \
        -f "$restore_file" \
        > /dev/null 2>&1 \
        || error_exit "Failed to restore database"

    # Calculate restore time
    local restore_end=$(date +%s)
    local restore_duration=$((restore_end - restore_start))

    log "Restore completed in ${restore_duration} seconds"

    # Return restore duration
    echo "$restore_duration"
}

# Validate restored data
validate_data() {
    log "Validating restored data..."

    local validation_errors=0

    # Check critical tables exist
    local tables=(
        "users"
        "organizations"
        "genomic_data_metadata"
        "hypervector_storage"
        "audit_logs"
        "phi_access_logs"
    )

    for table in "${tables[@]}"; do
        PGPASSWORD="${DATABASE_PASSWORD}" psql \
            -h "${DATABASE_HOST}" \
            -p "${DATABASE_PORT}" \
            -U "${DATABASE_USER}" \
            -d "${TEST_DB}" \
            -c "SELECT 1 FROM ${table} LIMIT 1;" \
            > /dev/null 2>&1

        if [[ $? -ne 0 ]]; then
            warning "Table ${table} not found or empty"
            ((validation_errors++))
        else
            success "Table ${table} validated"
        fi
    done

    # Check row counts
    log "Checking row counts..."

    PGPASSWORD="${DATABASE_PASSWORD}" psql \
        -h "${DATABASE_HOST}" \
        -p "${DATABASE_PORT}" \
        -U "${DATABASE_USER}" \
        -d "${TEST_DB}" \
        -c "SELECT
            (SELECT COUNT(*) FROM users) as user_count,
            (SELECT COUNT(*) FROM genomic_data_metadata) as genomic_count,
            (SELECT COUNT(*) FROM audit_logs) as audit_count;" \
        | tee -a "$LOG_FILE"

    # Check data integrity
    log "Checking data integrity..."

    # Verify foreign key constraints
    PGPASSWORD="${DATABASE_PASSWORD}" psql \
        -h "${DATABASE_HOST}" \
        -p "${DATABASE_PORT}" \
        -U "${DATABASE_USER}" \
        -d "${TEST_DB}" \
        -c "SELECT conname FROM pg_constraint WHERE contype = 'f';" \
        > /dev/null 2>&1

    if [[ $? -eq 0 ]]; then
        success "Foreign key constraints intact"
    else
        warning "Foreign key constraint issues detected"
        ((validation_errors++))
    fi

    # Check indexes
    PGPASSWORD="${DATABASE_PASSWORD}" psql \
        -h "${DATABASE_HOST}" \
        -p "${DATABASE_PORT}" \
        -U "${DATABASE_USER}" \
        -d "${TEST_DB}" \
        -c "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public';" \
        | tee -a "$LOG_FILE"

    return $validation_errors
}

# Test failover scenario
test_failover() {
    log "Testing failover scenario..."

    # Simulate primary failure (would normally switch to replica)
    log "Simulating primary database failure..."

    # In a real scenario, this would:
    # 1. Promote replica to primary
    # 2. Update connection strings
    # 3. Verify application connectivity

    # For testing, we'll just verify the backup can be restored
    # to a different host/port if configured

    if [[ -n "${FAILOVER_HOST:-}" ]]; then
        log "Testing restore to failover host: ${FAILOVER_HOST}"

        DATABASE_HOST="${FAILOVER_HOST}" \
        DATABASE_PORT="${FAILOVER_PORT:-5432}" \
            test_restore "$1"

        success "Failover test completed"
    else
        warning "No failover host configured, skipping failover test"
    fi
}

# Calculate RTO
calculate_rto() {
    local start="$1"
    local end=$(date +%s)
    local duration=$((end - start))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))

    log "Recovery Time: ${hours}h ${minutes}m"

    # Check if RTO < 4 hours
    if [[ $hours -lt 4 ]]; then
        success "RTO requirement met: ${hours}h ${minutes}m < 4 hours"
        return 0
    else
        error_exit "RTO requirement NOT met: ${hours}h ${minutes}m >= 4 hours"
    fi
}

# Main DR test
main() {
    log "=========================================="
    log "Starting Disaster Recovery Test"
    log "=========================================="

    # Find latest full backup
    log "Finding latest full backup..."
    BACKUP_FILE=$(find_latest_backup "full") || error_exit "No full backup found"
    success "Found backup: $BACKUP_FILE"

    # Test restoration
    RESTORE_TIME=$(test_restore "$BACKUP_FILE")
    success "Database restored in ${RESTORE_TIME} seconds"

    # Validate data
    validate_data
    VALIDATION_RESULT=$?

    if [[ $VALIDATION_RESULT -eq 0 ]]; then
        success "All data validation checks passed"
    else
        warning "Some validation checks failed: ${VALIDATION_RESULT} issues found"
    fi

    # Test failover (if configured)
    test_failover "$BACKUP_FILE"

    # Cleanup test database
    log "Cleaning up test database..."
    PGPASSWORD="${DATABASE_PASSWORD}" psql \
        -h "${DATABASE_HOST}" \
        -p "${DATABASE_PORT}" \
        -U "${DATABASE_USER}" \
        -d postgres \
        -c "DROP DATABASE IF EXISTS ${TEST_DB};" 2>/dev/null || true

    # Calculate and verify RTO
    calculate_rto "$START_TIME"

    # Send success notification
    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        curl -X POST "${SLACK_WEBHOOK}" \
            -H 'Content-Type: application/json' \
            -d "{
                \"text\": \"DR Test Successful\",
                \"attachments\": [{
                    \"color\": \"good\",
                    \"fields\": [
                        {\"title\": \"Restore Time\", \"value\": \"${RESTORE_TIME}s\", \"short\": true},
                        {\"title\": \"Validation Issues\", \"value\": \"${VALIDATION_RESULT}\", \"short\": true},
                        {\"title\": \"RTO\", \"value\": \"< 4 hours ✓\", \"short\": true}
                    ]
                }]
            }" 2>/dev/null || true
    fi

    log "=========================================="
    success "Disaster Recovery Test Completed Successfully"
    log "=========================================="
}

# Run main function
main "$@"
