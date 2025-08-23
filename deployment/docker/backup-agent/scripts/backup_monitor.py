#!/usr/bin/env python3
"""
Backup monitoring and alerting for GenomeVault.

Monitors backup status, validates integrity, and sends alerts for failures.
Implements HIPAA-compliant audit logging for backup operations.
"""

import os
import sys
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import psycopg2
from google.cloud import storage as gcs
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from cryptography.fernet import Fernet
from retrying import retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/backup_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
backup_attempts = Counter('genomevault_backup_attempts_total', 'Total backup attempts', ['type'])
backup_successes = Counter('genomevault_backup_successes_total', 'Successful backups', ['type'])
backup_failures = Counter('genomevault_backup_failures_total', 'Failed backups', ['type'])
backup_size_bytes = Gauge('genomevault_backup_size_bytes', 'Size of last backup in bytes', ['type'])
backup_duration_seconds = Histogram('genomevault_backup_duration_seconds', 'Backup duration', ['type'])
restore_test_results = Gauge('genomevault_restore_test_success', 'Restore test success (1) or failure (0)')
data_integrity_checks = Counter('genomevault_data_integrity_checks_total', 'Data integrity checks', ['status'])


class BackupMonitor:
    """Monitor and validate backup operations."""
    
    def __init__(self):
        """Initialize backup monitor with cloud storage clients."""
        self.s3_client = self._init_s3()
        self.gcs_client = self._init_gcs()
        self.slack_client = self._init_slack()
        self.encryption_key = self._load_encryption_key()
        
        # Configuration
        self.s3_bucket = os.getenv('S3_BACKUP_BUCKET', 'genomevault-backups-prod')
        self.gcs_bucket = os.getenv('GCS_BACKUP_BUCKET', 'genomevault-backups-prod')
        self.retention_days = {
            'incremental': int(os.getenv('INCREMENTAL_RETENTION_DAYS', '7')),
            'full': int(os.getenv('FULL_RETENTION_DAYS', '90')),
            'archive': int(os.getenv('ARCHIVE_RETENTION_DAYS', '2555'))  # 7 years for HIPAA
        }
    
    def _init_s3(self) -> Optional[boto3.client]:
        """Initialize S3 client."""
        if os.getenv('AWS_ACCESS_KEY_ID'):
            return boto3.client(
                's3',
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
        return None
    
    def _init_gcs(self) -> Optional[gcs.Client]:
        """Initialize GCS client."""
        if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            return gcs.Client()
        return None
    
    def _init_slack(self) -> Optional[WebClient]:
        """Initialize Slack client for alerts."""
        webhook_url = os.getenv('SLACK_WEBHOOK')
        if webhook_url:
            return WebClient(token=os.getenv('SLACK_BOT_TOKEN'))
        return None
    
    def _load_encryption_key(self) -> bytes:
        """Load encryption key for backup validation."""
        key_file = os.getenv('ENCRYPTION_KEY_FILE', '/etc/encryption/backup-key')
        try:
            with open(key_file, 'rb') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Encryption key not found at {key_file}")
            return b''
    
    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def monitor_backup(self, backup_type: str, backup_path: str) -> Dict:
        """
        Monitor a backup operation and collect metrics.
        
        Args:
            backup_type: Type of backup (incremental, full, archive)
            backup_path: Path to the backup file
            
        Returns:
            Monitoring results dictionary
        """
        start_time = time.time()
        results = {
            'type': backup_type,
            'path': backup_path,
            'timestamp': datetime.utcnow().isoformat(),
            'success': False,
            'size_bytes': 0,
            'checksum': '',
            'encrypted': False,
            'replicated': False,
            'alerts': []
        }
        
        try:
            # Update attempt counter
            backup_attempts.labels(type=backup_type).inc()
            
            # Validate backup file exists and get size
            if os.path.exists(backup_path):
                results['size_bytes'] = os.path.getsize(backup_path)
                backup_size_bytes.labels(type=backup_type).set(results['size_bytes'])
                
                # Calculate checksum
                results['checksum'] = self._calculate_checksum(backup_path)
                
                # Verify encryption
                results['encrypted'] = self._verify_encryption(backup_path)
                
                # Upload to cloud storage
                results['replicated'] = self._replicate_backup(backup_type, backup_path)
                
                # Validate backup integrity
                if self._validate_backup(backup_path):
                    results['success'] = True
                    backup_successes.labels(type=backup_type).inc()
                    logger.info(f"Backup {backup_type} completed successfully: {backup_path}")
                else:
                    raise ValueError("Backup validation failed")
            else:
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
                
        except Exception as e:
            backup_failures.labels(type=backup_type).inc()
            logger.error(f"Backup {backup_type} failed: {str(e)}")
            results['alerts'].append({
                'level': 'ERROR',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Send alert
            self._send_alert(
                level='ERROR',
                title=f"Backup {backup_type} Failed",
                message=str(e),
                details=results
            )
        
        finally:
            # Record duration
            duration = time.time() - start_time
            backup_duration_seconds.labels(type=backup_type).observe(duration)
            results['duration_seconds'] = duration
            
            # Log to audit trail
            self._audit_log(results)
        
        return results
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _verify_encryption(self, file_path: str) -> bool:
        """Verify that backup is encrypted."""
        try:
            # Check file header for encryption signature
            with open(file_path, 'rb') as f:
                header = f.read(32)
                # Check for GPG or OpenSSL encryption headers
                if header.startswith(b'-----BEGIN PGP MESSAGE-----') or \
                   header.startswith(b'Salted__'):
                    return True
            
            # Try to decrypt a small portion to verify
            if self.encryption_key:
                fernet = Fernet(self.encryption_key[:32].ljust(32, b'0'))
                with open(file_path, 'rb') as f:
                    test_data = f.read(1024)
                    try:
                        fernet.decrypt(test_data)
                        return True
                    except:
                        pass
            
            return False
        except Exception as e:
            logger.warning(f"Could not verify encryption: {e}")
            return False
    
    def _replicate_backup(self, backup_type: str, local_path: str) -> bool:
        """
        Replicate backup to cloud storage.
        
        Args:
            backup_type: Type of backup
            local_path: Local backup file path
            
        Returns:
            True if successfully replicated to at least one cloud
        """
        success = False
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Upload to S3
        if self.s3_client:
            try:
                s3_key = f"{backup_type}/{timestamp}/{os.path.basename(local_path)}"
                self.s3_client.upload_file(
                    local_path,
                    self.s3_bucket,
                    s3_key,
                    ExtraArgs={
                        'ServerSideEncryption': 'AES256',
                        'StorageClass': 'STANDARD_IA' if backup_type == 'full' else 'STANDARD',
                        'Metadata': {
                            'backup-type': backup_type,
                            'timestamp': timestamp,
                            'checksum': self._calculate_checksum(local_path)
                        }
                    }
                )
                logger.info(f"Backup replicated to S3: s3://{self.s3_bucket}/{s3_key}")
                success = True
            except Exception as e:
                logger.error(f"S3 replication failed: {e}")
        
        # Upload to GCS
        if self.gcs_client:
            try:
                bucket = self.gcs_client.bucket(self.gcs_bucket)
                blob_name = f"{backup_type}/{timestamp}/{os.path.basename(local_path)}"
                blob = bucket.blob(blob_name)
                
                # Set storage class based on backup type
                if backup_type == 'archive':
                    blob.storage_class = 'ARCHIVE'
                elif backup_type == 'full':
                    blob.storage_class = 'NEARLINE'
                else:
                    blob.storage_class = 'STANDARD'
                
                blob.upload_from_filename(local_path)
                blob.metadata = {
                    'backup-type': backup_type,
                    'timestamp': timestamp,
                    'checksum': self._calculate_checksum(local_path)
                }
                blob.patch()
                
                logger.info(f"Backup replicated to GCS: gs://{self.gcs_bucket}/{blob_name}")
                success = True
            except Exception as e:
                logger.error(f"GCS replication failed: {e}")
        
        return success
    
    def _validate_backup(self, backup_path: str) -> bool:
        """
        Validate backup integrity.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if backup is valid
        """
        try:
            # Basic file validation
            if not os.path.exists(backup_path):
                return False
            
            if os.path.getsize(backup_path) == 0:
                return False
            
            # For PostgreSQL dumps, try to validate format
            if backup_path.endswith('.sql') or backup_path.endswith('.sql.gz'):
                # Check for PostgreSQL dump header
                with open(backup_path, 'rb') as f:
                    header = f.read(100)
                    if b'PostgreSQL' in header or b'PGDMP' in header:
                        data_integrity_checks.labels(status='success').inc()
                        return True
            
            # For encrypted files, just check they're not corrupted
            if self._verify_encryption(backup_path):
                data_integrity_checks.labels(status='success').inc()
                return True
            
            data_integrity_checks.labels(status='failure').inc()
            return False
            
        except Exception as e:
            logger.error(f"Backup validation failed: {e}")
            data_integrity_checks.labels(status='error').inc()
            return False
    
    def test_restore(self, backup_path: str, test_database: str = 'genomevault_test') -> bool:
        """
        Test restore process with a backup.
        
        Args:
            backup_path: Path to backup to test
            test_database: Test database name
            
        Returns:
            True if restore test successful
        """
        success = False
        
        try:
            # Create test database
            conn = psycopg2.connect(
                host=os.getenv('DATABASE_HOST', 'localhost'),
                port=os.getenv('DATABASE_PORT', '5432'),
                user=os.getenv('DATABASE_USER', 'genomevault'),
                password=os.getenv('DATABASE_PASSWORD'),
                database='postgres'
            )
            conn.autocommit = True
            cur = conn.cursor()
            
            # Drop test database if exists
            cur.execute(f"DROP DATABASE IF EXISTS {test_database}")
            cur.execute(f"CREATE DATABASE {test_database}")
            
            # Restore backup to test database
            restore_cmd = f"pg_restore -h {os.getenv('DATABASE_HOST')} -p {os.getenv('DATABASE_PORT')} " \
                         f"-U {os.getenv('DATABASE_USER')} -d {test_database} {backup_path}"
            
            result = os.system(restore_cmd)
            
            if result == 0:
                # Verify restored data
                test_conn = psycopg2.connect(
                    host=os.getenv('DATABASE_HOST'),
                    port=os.getenv('DATABASE_PORT'),
                    user=os.getenv('DATABASE_USER'),
                    password=os.getenv('DATABASE_PASSWORD'),
                    database=test_database
                )
                test_cur = test_conn.cursor()
                
                # Check critical tables exist
                test_cur.execute("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('users', 'genomic_data_metadata', 'audit_logs')
                """)
                table_count = test_cur.fetchone()[0]
                
                if table_count >= 3:
                    success = True
                    restore_test_results.set(1)
                    logger.info("Restore test passed")
                else:
                    restore_test_results.set(0)
                    logger.error("Restore test failed: missing critical tables")
                
                test_conn.close()
            else:
                restore_test_results.set(0)
                logger.error(f"Restore command failed with code {result}")
            
            # Cleanup test database
            cur.execute(f"DROP DATABASE IF EXISTS {test_database}")
            conn.close()
            
        except Exception as e:
            restore_test_results.set(0)
            logger.error(f"Restore test failed: {e}")
        
        return success
    
    def cleanup_old_backups(self):
        """Clean up old backups based on retention policy."""
        try:
            cutoff_dates = {
                'incremental': datetime.utcnow() - timedelta(days=self.retention_days['incremental']),
                'full': datetime.utcnow() - timedelta(days=self.retention_days['full']),
                'archive': datetime.utcnow() - timedelta(days=self.retention_days['archive'])
            }
            
            # Cleanup S3
            if self.s3_client:
                self._cleanup_s3_backups(cutoff_dates)
            
            # Cleanup GCS
            if self.gcs_client:
                self._cleanup_gcs_backups(cutoff_dates)
            
            # Cleanup local backups
            self._cleanup_local_backups(cutoff_dates)
            
            logger.info("Backup cleanup completed")
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            self._send_alert(
                level='WARNING',
                title='Backup Cleanup Failed',
                message=str(e)
            )
    
    def _cleanup_s3_backups(self, cutoff_dates: Dict[str, datetime]):
        """Clean up old S3 backups."""
        try:
            for backup_type, cutoff_date in cutoff_dates.items():
                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(
                    Bucket=self.s3_bucket,
                    Prefix=f"{backup_type}/"
                )
                
                objects_to_delete = []
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                                objects_to_delete.append({'Key': obj['Key']})
                
                if objects_to_delete:
                    self.s3_client.delete_objects(
                        Bucket=self.s3_bucket,
                        Delete={'Objects': objects_to_delete}
                    )
                    logger.info(f"Deleted {len(objects_to_delete)} old {backup_type} backups from S3")
                    
        except Exception as e:
            logger.error(f"S3 cleanup failed: {e}")
    
    def _cleanup_gcs_backups(self, cutoff_dates: Dict[str, datetime]):
        """Clean up old GCS backups."""
        try:
            bucket = self.gcs_client.bucket(self.gcs_bucket)
            
            for backup_type, cutoff_date in cutoff_dates.items():
                blobs = bucket.list_blobs(prefix=f"{backup_type}/")
                deleted_count = 0
                
                for blob in blobs:
                    if blob.time_created.replace(tzinfo=None) < cutoff_date:
                        blob.delete()
                        deleted_count += 1
                
                if deleted_count > 0:
                    logger.info(f"Deleted {deleted_count} old {backup_type} backups from GCS")
                    
        except Exception as e:
            logger.error(f"GCS cleanup failed: {e}")
    
    def _cleanup_local_backups(self, cutoff_dates: Dict[str, datetime]):
        """Clean up old local backups."""
        backup_dir = Path(os.getenv('BACKUP_DIR', '/backups'))
        
        for backup_type, cutoff_date in cutoff_dates.items():
            type_dir = backup_dir / backup_type
            if type_dir.exists():
                deleted_count = 0
                for backup_file in type_dir.glob('*'):
                    if backup_file.is_file():
                        file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                        if file_time < cutoff_date:
                            backup_file.unlink()
                            deleted_count += 1
                
                if deleted_count > 0:
                    logger.info(f"Deleted {deleted_count} old {backup_type} backups from local storage")
    
    def _send_alert(self, level: str, title: str, message: str, details: Optional[Dict] = None):
        """Send alert via Slack."""
        if not self.slack_client:
            return
        
        try:
            color = {
                'ERROR': 'danger',
                'WARNING': 'warning',
                'INFO': 'good'
            }.get(level, 'default')
            
            attachment = {
                'color': color,
                'title': title,
                'text': message,
                'fields': [],
                'footer': 'GenomeVault Backup Monitor',
                'ts': int(time.time())
            }
            
            if details:
                for key, value in details.items():
                    if key != 'alerts':  # Skip nested alerts
                        attachment['fields'].append({
                            'title': key.replace('_', ' ').title(),
                            'value': str(value),
                            'short': True
                        })
            
            self.slack_client.chat_postMessage(
                channel=os.getenv('SLACK_CHANNEL', '#genomevault-alerts'),
                attachments=[attachment]
            )
            
        except SlackApiError as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _audit_log(self, results: Dict):
        """Log backup operation to audit trail."""
        try:
            audit_entry = {
                'timestamp': results['timestamp'],
                'operation': f"backup_{results['type']}",
                'success': results['success'],
                'details': {
                    'path': results['path'],
                    'size_bytes': results['size_bytes'],
                    'checksum': results['checksum'],
                    'encrypted': results['encrypted'],
                    'replicated': results['replicated'],
                    'duration_seconds': results.get('duration_seconds', 0)
                }
            }
            
            # Log to file (would normally go to database)
            with open('/var/log/backup_audit.jsonl', 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
            
            # Also log to standard audit log
            if results['success']:
                logger.info(f"AUDIT: Backup {results['type']} completed successfully")
            else:
                logger.error(f"AUDIT: Backup {results['type']} failed")
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def export_metrics(self) -> str:
        """Export Prometheus metrics."""
        return generate_latest().decode('utf-8')


def main():
    """Main monitoring loop."""
    monitor = BackupMonitor()
    
    # Get backup type from environment
    backup_type = os.getenv('BACKUP_TYPE', 'incremental')
    backup_path = os.getenv('BACKUP_PATH')
    
    if not backup_path:
        logger.error("BACKUP_PATH environment variable not set")
        sys.exit(1)
    
    # Monitor the backup
    results = monitor.monitor_backup(backup_type, backup_path)
    
    # Run restore test for weekly backups
    if backup_type == 'weekly':
        monitor.test_restore(backup_path)
    
    # Cleanup old backups after daily backup
    if backup_type == 'daily':
        monitor.cleanup_old_backups()
    
    # Export metrics
    metrics = monitor.export_metrics()
    with open('/var/log/backup_metrics.txt', 'w') as f:
        f.write(metrics)
    
    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)


if __name__ == '__main__':
    main()