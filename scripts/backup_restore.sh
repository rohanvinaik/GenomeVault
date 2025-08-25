#!/bin/bash

# GenomeVault Database Backup and Restore Script
# Implements encrypted backups with RPO < 1 hour and RTO < 4 hours
# HIPAA-compliant with audit logging and encryption

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/genomevault}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-90}"
ENCRYPTION_KEY_FILE="${ENCRYPTION_KEY_FILE:-/etc/genomevault/backup-key}"
LOG_FILE="${LOG_FILE:-${GENOMEVAULT_LOGS:-${GENOMEVAULT_LOGS:-/var/log/genomevault}}/backup.log}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"

# Database configuration from environment
DB_HOST="${DATABASE_HOST:-localhost}"
DB_PORT="${DATABASE_PORT:-5432}"
DB_NAME="${DATABASE_NAME:-genomevault}"
DB_USER="${DATABASE_USER:-genomevault}"
DB_PASSWORD="${DATABASE_PASSWORD:-}"

# AWS S3 configuration for offsite backup
S3_BUCKET="${S3_BACKUP_BUCKET:-}"
S3_PREFIX="${S3_BACKUP_PREFIX:-genomevault-backups}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# GCS configuration for offsite backup
GCS_BUCKET="${GCS_BACKUP_BUCKET:-}"
GCS_PREFIX="${GCS_BACKUP_PREFIX:-genomevault-backups}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"

    # Send critical alerts to Slack
    if [[ "${level}" == "ERROR" ]] && [[ -n "${SLACK_WEBHOOK}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"GenomeVault Backup ${level}: ${message}\"}" \
            "${SLACK_WEBHOOK}" 2>/dev/null || true
    fi
}

# Check prerequisites
check_prerequisites() {
    local missing_tools=()

    # Check required tools
    for tool in pg_dump pg_restore psql openssl jq; do
        if ! command -v ${tool} &> /dev/null; then
            missing_tools+=("${tool}")
        fi
    done

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log "ERROR" "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi

    # Check encryption key
    if [[ ! -f "${ENCRYPTION_KEY_FILE}" ]]; then
        log "ERROR" "Encryption key file not found: ${ENCRYPTION_KEY_FILE}"
        exit 1
    fi

    # Create backup directory if it doesn't exist
    mkdir -p "${BACKUP_DIR}"
    mkdir -p "$(dirname "${LOG_FILE}")"
}

# Generate backup filename with timestamp
generate_backup_filename() {
    local backup_type="${1:-full}"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    echo "genomevault_${backup_type}_${timestamp}.sql"
}

# Encrypt backup file
encrypt_backup() {
    local input_file=$1
    local output_file="${input_file}.enc"

    log "INFO" "Encrypting backup file..."
    openssl enc -aes-256-cbc -salt -pbkdf2 \
        -in "${input_file}" \
        -out "${output_file}" \
        -pass file:"${ENCRYPTION_KEY_FILE}"

    # Generate checksum
    local checksum=$(sha256sum "${output_file}" | cut -d' ' -f1)
    echo "${checksum}" > "${output_file}.sha256"

    # Remove unencrypted file
    rm -f "${input_file}"

    echo "${output_file}"
}

# Decrypt backup file
decrypt_backup() {
    local input_file=$1
    local output_file="${input_file%.enc}"

    log "INFO" "Decrypting backup file..."

    # Verify checksum
    if [[ -f "${input_file}.sha256" ]]; then
        local expected_checksum=$(cat "${input_file}.sha256")
        local actual_checksum=$(sha256sum "${input_file}" | cut -d' ' -f1)

        if [[ "${expected_checksum}" != "${actual_checksum}" ]]; then
            log "ERROR" "Checksum verification failed for ${input_file}"
            exit 1
        fi
        log "INFO" "Checksum verified successfully"
    fi

    openssl enc -aes-256-cbc -d -pbkdf2 \
        -in "${input_file}" \
        -out "${output_file}" \
        -pass file:"${ENCRYPTION_KEY_FILE}"

    echo "${output_file}"
}

# Perform full backup
perform_full_backup() {
    local start_time=$(date +%s)
    local backup_file=$(generate_backup_filename "full")
    local backup_path="${BACKUP_DIR}/${backup_file}"

    log "INFO" "Starting full backup to ${backup_path}"

    # Set PGPASSWORD for authentication
    export PGPASSWORD="${DB_PASSWORD}"

    # Perform backup with custom options for HIPAA compliance
    pg_dump \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        -f "${backup_path}" \
        --verbose \
        --format=custom \
        --compress=9 \
        --no-privileges \
        --no-owner \
        --exclude-table-data='audit_logs_*' \
        --exclude-table-data='phi_access_logs_*' \
        --exclude-table-data='query_history_*' \
        2>&1 | tee -a "${LOG_FILE}"

    # Backup audit tables separately (they're partitioned)
    log "INFO" "Backing up audit tables..."
    pg_dump \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        -f "${backup_path}.audit" \
        --verbose \
        --format=custom \
        --compress=9 \
        --table='audit_logs*' \
        --table='phi_access_logs*' \
        --table='query_history*' \
        2>&1 | tee -a "${LOG_FILE}"

    unset PGPASSWORD

    # Encrypt backups
    local encrypted_backup=$(encrypt_backup "${backup_path}")
    local encrypted_audit=$(encrypt_backup "${backup_path}.audit")

    # Calculate backup size and duration
    local backup_size=$(du -h "${encrypted_backup}" | cut -f1)
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Update backup history in database
    update_backup_history "full" "completed" "${encrypted_backup}" "${backup_size}" "${duration}"

    log "INFO" "Full backup completed: ${encrypted_backup} (${backup_size}, ${duration}s)"

    # Upload to cloud storage
    upload_to_cloud "${encrypted_backup}"
    upload_to_cloud "${encrypted_audit}"

    echo "${encrypted_backup}"
}

# Perform incremental backup using WAL archiving
perform_incremental_backup() {
    local start_time=$(date +%s)
    local backup_file=$(generate_backup_filename "incremental")
    local backup_path="${BACKUP_DIR}/${backup_file}"

    log "INFO" "Starting incremental backup using pg_basebackup"

    export PGPASSWORD="${DB_PASSWORD}"

    # Use pg_basebackup for incremental backup
    pg_basebackup \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -D "${backup_path}.tmp" \
        -Ft \
        -z \
        -Xs \
        -P \
        -v \
        2>&1 | tee -a "${LOG_FILE}"

    # Tar the backup directory
    tar -czf "${backup_path}" -C "${backup_path}.tmp" .
    rm -rf "${backup_path}.tmp"

    unset PGPASSWORD

    # Encrypt backup
    local encrypted_backup=$(encrypt_backup "${backup_path}")

    # Calculate backup size and duration
    local backup_size=$(du -h "${encrypted_backup}" | cut -f1)
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log "INFO" "Incremental backup completed: ${encrypted_backup} (${backup_size}, ${duration}s)"

    # Upload to cloud storage
    upload_to_cloud "${encrypted_backup}"

    echo "${encrypted_backup}"
}

# Upload backup to cloud storage
upload_to_cloud() {
    local backup_file=$1
    local filename=$(basename "${backup_file}")

    # Upload to AWS S3
    if [[ -n "${S3_BUCKET}" ]]; then
        log "INFO" "Uploading to S3: s3://${S3_BUCKET}/${S3_PREFIX}/${filename}"
        aws s3 cp "${backup_file}" "s3://${S3_BUCKET}/${S3_PREFIX}/${filename}" \
            --storage-class GLACIER_IR \
            --server-side-encryption AES256 \
            --metadata "backup-date=$(date -Iseconds)" \
            2>&1 | tee -a "${LOG_FILE}"

        # Also upload checksum
        if [[ -f "${backup_file}.sha256" ]]; then
            aws s3 cp "${backup_file}.sha256" "s3://${S3_BUCKET}/${S3_PREFIX}/${filename}.sha256" \
                2>&1 | tee -a "${LOG_FILE}"
        fi
    fi

    # Upload to Google Cloud Storage
    if [[ -n "${GCS_BUCKET}" ]]; then
        log "INFO" "Uploading to GCS: gs://${GCS_BUCKET}/${GCS_PREFIX}/${filename}"
        gsutil -o GSUtil:parallel_composite_upload_threshold=150M \
            cp "${backup_file}" "gs://${GCS_BUCKET}/${GCS_PREFIX}/${filename}" \
            2>&1 | tee -a "${LOG_FILE}"

        # Set lifecycle for automatic transition to coldline
        gsutil lifecycle set -r <(cat <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {"age": 30}
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "ARCHIVE"},
        "condition": {"age": 365}
      }
    ]
  }
}
EOF
        ) "gs://${GCS_BUCKET}"
    fi
}

# Restore from backup
restore_backup() {
    local backup_file=$1
    local target_db="${2:-${DB_NAME}_restored}"

    log "INFO" "Starting restore from ${backup_file} to database ${target_db}"

    # Decrypt backup if encrypted
    local restore_file="${backup_file}"
    if [[ "${backup_file}" == *.enc ]]; then
        restore_file=$(decrypt_backup "${backup_file}")
    fi

    export PGPASSWORD="${DB_PASSWORD}"

    # Create target database if it doesn't exist
    psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d postgres \
        -c "CREATE DATABASE ${target_db} WITH TEMPLATE template0 ENCODING 'UTF8';" \
        2>/dev/null || true

    # Restore backup
    pg_restore \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${target_db}" \
        "${restore_file}" \
        --verbose \
        --no-owner \
        --no-privileges \
        --if-exists \
        --clean \
        2>&1 | tee -a "${LOG_FILE}"

    unset PGPASSWORD

    # Cleanup decrypted file
    if [[ "${restore_file}" != "${backup_file}" ]]; then
        rm -f "${restore_file}"
    fi

    log "INFO" "Restore completed to database ${target_db}"

    # Verify restore
    verify_restore "${target_db}"
}

# Verify restore integrity
verify_restore() {
    local target_db=$1

    log "INFO" "Verifying restore integrity for ${target_db}"

    export PGPASSWORD="${DB_PASSWORD}"

    # Check table counts
    local tables=("users" "organizations" "genomic_data_metadata" "hypervector_storage")
    for table in "${tables[@]}"; do
        local count=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${target_db}" \
            -t -c "SELECT COUNT(*) FROM ${table};" 2>/dev/null || echo "0")
        log "INFO" "Table ${table}: ${count} rows"
    done

    # Check for required extensions
    local extensions=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${target_db}" \
        -t -c "SELECT extname FROM pg_extension;" 2>/dev/null)
    log "INFO" "Installed extensions: ${extensions}"

    unset PGPASSWORD
}

# Update backup history in database
update_backup_history() {
    local backup_type=$1
    local status=$2
    local location=$3
    local size=$4
    local duration=$5

    export PGPASSWORD="${DB_PASSWORD}"

    psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" <<EOF
INSERT INTO backup_history (
    backup_type, backup_status, backup_location,
    backup_size_bytes, encrypted, started_at,
    completed_at, duration_seconds
) VALUES (
    '${backup_type}', '${status}', '${location}',
    (SELECT ${size%%[A-Z]*} *
        CASE
            WHEN '${size}' LIKE '%G' THEN 1073741824
            WHEN '${size}' LIKE '%M' THEN 1048576
            WHEN '${size}' LIKE '%K' THEN 1024
            ELSE 1
        END),
    true,
    NOW() - interval '${duration} seconds',
    NOW(),
    ${duration}
);
EOF

    unset PGPASSWORD
}

# Clean up old backups
cleanup_old_backups() {
    log "INFO" "Cleaning up backups older than ${BACKUP_RETENTION_DAYS} days"

    # Clean local backups
    find "${BACKUP_DIR}" -name "genomevault_*.enc" -mtime +${BACKUP_RETENTION_DAYS} -delete
    find "${BACKUP_DIR}" -name "genomevault_*.sha256" -mtime +${BACKUP_RETENTION_DAYS} -delete

    # Clean S3 backups (lifecycle policies handle this, but we can force it)
    if [[ -n "${S3_BUCKET}" ]]; then
        local cutoff_date=$(date -d "${BACKUP_RETENTION_DAYS} days ago" +%Y-%m-%d)
        aws s3api list-objects-v2 \
            --bucket "${S3_BUCKET}" \
            --prefix "${S3_PREFIX}/" \
            --query "Contents[?LastModified<'${cutoff_date}'].Key" \
            --output text | \
        xargs -I {} aws s3 rm "s3://${S3_BUCKET}/{}" 2>/dev/null || true
    fi
}

# Disaster recovery test
test_disaster_recovery() {
    log "INFO" "Starting disaster recovery test"

    local start_time=$(date +%s)

    # 1. Create test backup
    local test_backup=$(perform_full_backup)

    # 2. Restore to test database
    local test_db="genomevault_dr_test_$(date +%Y%m%d_%H%M%S)"
    restore_backup "${test_backup}" "${test_db}"

    # 3. Verify restore
    verify_restore "${test_db}"

    # 4. Clean up test database
    export PGPASSWORD="${DB_PASSWORD}"
    psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d postgres \
        -c "DROP DATABASE IF EXISTS ${test_db};"
    unset PGPASSWORD

    local end_time=$(date +%s)
    local rto=$((end_time - start_time))

    log "INFO" "Disaster recovery test completed. RTO: ${rto} seconds"

    # Check if RTO meets requirement (< 4 hours = 14400 seconds)
    if [[ ${rto} -gt 14400 ]]; then
        log "ERROR" "RTO exceeds 4 hours requirement: ${rto} seconds"
        return 1
    fi

    return 0
}

# Monitor backup status
monitor_backups() {
    export PGPASSWORD="${DB_PASSWORD}"

    # Get recent backup history
    psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" <<EOF
SELECT
    backup_type,
    backup_status,
    backup_size_bytes / 1073741824.0 AS size_gb,
    duration_seconds / 60.0 AS duration_minutes,
    started_at,
    completed_at
FROM backup_history
WHERE started_at > NOW() - interval '7 days'
ORDER BY started_at DESC
LIMIT 10;
EOF

    # Check last successful backup
    local last_backup=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
        -t -c "SELECT EXTRACT(EPOCH FROM (NOW() - MAX(completed_at)))/3600
               FROM backup_history
               WHERE backup_status = 'completed';" 2>/dev/null)

    unset PGPASSWORD

    # Alert if no backup in last hour (RPO requirement)
    if (( $(echo "${last_backup} > 1" | bc -l) )); then
        log "ERROR" "RPO violation: Last successful backup was ${last_backup} hours ago"
        return 1
    fi

    log "INFO" "Backup monitoring: Last backup ${last_backup} hours ago"
}

# Main function
main() {
    case "${1:-}" in
        backup-full)
            check_prerequisites
            perform_full_backup
            cleanup_old_backups
            ;;
        backup-incremental)
            check_prerequisites
            perform_incremental_backup
            cleanup_old_backups
            ;;
        restore)
            check_prerequisites
            restore_backup "${2:-}" "${3:-}"
            ;;
        test-dr)
            check_prerequisites
            test_disaster_recovery
            ;;
        monitor)
            monitor_backups
            ;;
        cleanup)
            cleanup_old_backups
            ;;
        *)
            echo "Usage: $0 {backup-full|backup-incremental|restore <backup_file> [target_db]|test-dr|monitor|cleanup}"
            echo ""
            echo "Environment variables:"
            echo "  DATABASE_HOST     Database host (default: localhost)"
            echo "  DATABASE_PORT     Database port (default: 5432)"
            echo "  DATABASE_NAME     Database name (default: genomevault)"
            echo "  DATABASE_USER     Database user (default: genomevault)"
            echo "  DATABASE_PASSWORD Database password"
            echo "  BACKUP_DIR        Backup directory (default: /var/backups/genomevault)"
            echo "  S3_BACKUP_BUCKET  S3 bucket for offsite backups"
            echo "  GCS_BACKUP_BUCKET GCS bucket for offsite backups"
            exit 1
            ;;
    esac
}

main "$@"
