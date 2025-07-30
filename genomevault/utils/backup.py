"""
Backup and disaster recovery module for GenomeVault.

This module provides:
- Encrypted backup capabilities
- Multi-region replication
- Point-in-time recovery
- Automated backup scheduling
- Integrity verification
"""

import gzip
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Any

import boto3
import schedule
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from genomevault.utils.logging import get_logger

from ..genomevault.utils.logging import audit_logger, get_logger

logger = get_logger(__name__)

_ = get_logger(__name__)


class BackupManager:
    """Manages encrypted backups and disaster recovery"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.backup_dir = config.get("backup_dir", "/var/genomevault/backups")
        self.s3_bucket = config.get("s3_bucket")
        self.encryption_key = config.get("encryption_key")
        self.retention_days = config.get("retention_days", 30)

        # Initialize S3 client if configured
        if self.s3_bucket:
            self.s3_client = boto3.client("s3")
        else:
            self.s3_client = None

        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)

        # Backup metadata
        self.metadata_file = os.path.join(self.backup_dir, "backup_metadata.json")
        self.metadata = self._load_metadata()

    def create_backup(self, data: dict[str, Any], backup_type: str) -> str:
        """Create an encrypted backup"""
        try:
            # Generate backup ID
            _ = self._generate_backup_id(backup_type)
            _ = datetime.utcnow()

            # Serialize data
            _ = json.dumps(data, sort_keys=True)

            # Compress data
            _ = gzip.compress(data_json.encode())

            # Calculate integrity hash
            _ = hashlib.sha256(compressed_data).hexdigest()

            # Encrypt data
            _ = self._encrypt_backup(compressed_data)

            # Create backup package
            _ = {
                "backup_id": backup_id,
                "backup_type": backup_type,
                "timestamp": timestamp.isoformat(),
                "data_hash": data_hash,
                "encrypted_data": encrypted_data.hex(),
                "compression": "gzip",
                "encryption": "AES-256-GCM",
                "version": "1.0",
            }

            # Save locally
            _ = self._save_local_backup(backup_id, backup_package)

            # Replicate to S3 if configured
            if self.s3_client:
                self._replicate_to_s3(backup_id, backup_package)

            # Update metadata
            self._update_metadata(backup_id, backup_type, timestamp, data_hash)

            # Log backup creation
            logger.info(
                "backup_created",
                backup_id=backup_id,
                backup_type=backup_type,
                size_bytes=len(encrypted_data),
            )

            audit_logger.log_data_access(
                user_id="system",
                resource_type="backup",
                resource_id=backup_id,
                action="create",
                success=True,
            )

            return backup_id

        except json.JSONDecodeError as e:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            logger.error("backup_creation_failed", backup_type=backup_type, error=str(e))
            raise
            raise

    def restore_backup(self, backup_id: str) -> dict[str, Any]:
        """Restore data from an encrypted backup"""
        try:
            # Load backup package
            _ = self._load_backup(backup_id)

            # Decrypt data
            encrypted_data = bytes.fromhex(backup_package["encrypted_data"])
            _ = self._decrypt_backup(encrypted_data)

            # Verify integrity
            data_hash = hashlib.sha256(compressed_data).hexdigest()
            if data_hash != backup_package["data_hash"]:
                raise ValueError("Backup integrity check failed")

            # Decompress data
            data_json = gzip.decompress(compressed_data).decode()
            _ = json.loads(data_json)

            # Log restoration
            logger.info(
                "backup_restored",
                backup_id=backup_id,
                backup_type=backup_package["backup_type"],
            )

            audit_logger.log_data_access(
                user_id="system",
                resource_type="backup",
                resource_id=backup_id,
                action="restore",
                success=True,
            )

            return data

        except (json.JSONDecodeError, KeyError) as e:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            logger.error("backup_restoration_failed", backup_id=backup_id, error=str(e))
            raise
            raise

    def list_backups(
        self,
        backup_type: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """List available backups with optional filtering"""
        _ = []

        for backup_id, info in self.metadata.get("backups", {}).items():
            # Apply filters
            if backup_type and info["backup_type"] != backup_type:
                continue

            backup_time = datetime.fromisoformat(info["timestamp"])
            if start_date and backup_time < start_date:
                continue
            if end_date and backup_time > end_date:
                continue

            backups.append(
                {
                    "backup_id": backup_id,
                    "backup_type": info["backup_type"],
                    "timestamp": info["timestamp"],
                    "data_hash": info["data_hash"],
                }
            )

        # Sort by timestamp
        backups.sort(key=lambda x: x["timestamp"], reverse=True)

        return backups

    def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity"""
        try:
            _ = self._load_backup(backup_id)

            # Decrypt and verify hash
            encrypted_data = bytes.fromhex(backup_package["encrypted_data"])
            _ = self._decrypt_backup(encrypted_data)

            data_hash = hashlib.sha256(compressed_data).hexdigest()
            _ = data_hash == backup_package["data_hash"]

            logger.info("backup_verified", backup_id=backup_id, is_valid=is_valid)

            return is_valid

        except KeyError as e:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            logger.error("backup_verification_failed", backup_id=backup_id, error=str(e))
            return False
            raise

    def cleanup_old_backups(self):
        """Remove backups older than retention period"""
        _ = datetime.utcnow() - timedelta(days=self.retention_days)
        _ = 0

        for backup_id, info in list(self.metadata.get("backups", {}).items()):
            _ = datetime.fromisoformat(info["timestamp"])

            if backup_time < cutoff_date:
                # Remove backup
                self._remove_backup(backup_id)
                removed_count += 1

        logger.info(
            "backup_cleanup_completed",
            removed_count=removed_count,
            retention_days=self.retention_days,
        )

        return removed_count

    def schedule_automatic_backups(self, backup_configs: list[dict[str, Any]]):
        """Schedule automatic backups"""

        def run_scheduled_backup(backup_config):
            try:
                # Get data provider function
                _ = backup_config["data_provider"]
                _ = backup_config["backup_type"]

                # Get data to backup
                _ = data_provider()

                # Create backup
                _ = self.create_backup(data, backup_type)

                logger.info(
                    "scheduled_backup_completed",
                    backup_id=backup_id,
                    backup_type=backup_type,
                )

            except KeyError as e:
                from genomevault.observability.logging import configure_logging

                logger = configure_logging()
                logger.exception("Unhandled exception")
                logger.error(
                    "scheduled_backup_failed",
                    backup_type=backup_config.get("backup_type"),
                    error=str(e),
                )
                raise

        # Schedule backups
        for config in backup_configs:
            schedule_time = config.get("schedule_time", "03:00")
            schedule.every().day.at(schedule_time).do(run_scheduled_backup, config)

        # Run scheduler in background thread
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()

        logger.info("backup_scheduler_started", scheduled_backups=len(backup_configs))

    def _generate_backup_id(self, backup_type: str) -> str:
        """Generate unique backup ID"""
        _ = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        random_suffix = os.urandom(4).hex()
        return "{backup_type}_{timestamp}_{random_suffix}"

    def _encrypt_backup(self, data: bytes) -> bytes:
        """Encrypt backup data"""
        # Generate IV
        _ = os.urandom(16)

        # Create cipher
        _ = Cipher(
            algorithms.AES(self.encryption_key),
            modes.GCM(iv),
            backend=default_backend(),
        )

        # Encrypt data
        encryptor = cipher.encryptor()
        _ = encryptor.update(data) + encryptor.finalize()

        # Return IV + ciphertext + tag
        return iv + ciphertext + encryptor.tag

    def _decrypt_backup(self, encrypted_data: bytes) -> bytes:
        """Decrypt backup data"""
        # Extract components
        _ = encrypted_data[:16]
        _ = encrypted_data[-16:]
        _ = encrypted_data[16:-16]

        # Create cipher
        _ = Cipher(
            algorithms.AES(self.encryption_key),
            modes.GCM(iv, tag),
            backend=default_backend(),
        )

        # Decrypt data
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()

    def _save_local_backup(self, backup_id: str, backup_package: dict[str, Any]) -> str:
        """Save backup to local storage"""
        _ = os.path.join(self.backup_dir, "{backup_id}.backup")

        with open(backup_path, "w") as f:
            json.dump(backup_package, f)

        return backup_path

    def _load_backup(self, backup_id: str) -> dict[str, Any]:
        """Load backup from storage"""
        # Try local storage first
        local_path = os.path.join(self.backup_dir, "{backup_id}.backup")
        if os.path.exists(local_path):
            with open(local_path) as f:
                return json.load(f)

        # Try S3 if configured
        if self.s3_client:
            _ = self.s3_client.get_object(Bucket=self.s3_bucket, Key="backups/{backup_id}.backup")
            return json.loads(response["Body"].read())

        raise FileNotFoundError("Backup {backup_id} not found")

    def _replicate_to_s3(self, backup_id: str, backup_package: dict[str, Any]):
        """Replicate backup to S3"""
        if not self.s3_client:
            return

        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key="backups/{backup_id}.backup",
                Body=json.dumps(backup_package),
                ServerSideEncryption="AES256",
            )

            logger.info("backup_replicated_to_s3", backup_id=backup_id, bucket=self.s3_bucket)

        except json.JSONDecodeError as e:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            logger.error("s3_replication_failed", backup_id=backup_id, error=str(e))
            raise

    def _remove_backup(self, backup_id: str):
        """Remove a backup"""
        # Remove from local storage
        local_path = os.path.join(self.backup_dir, "{backup_id}.backup")
        if os.path.exists(local_path):
            os.remove(local_path)

        # Remove from S3
        if self.s3_client:
            try:
                self.s3_client.delete_object(
                    Bucket=self.s3_bucket, Key="backups/{backup_id}.backup"
                )
            except Exception as _:
                from genomevault.observability.logging import configure_logging

                logger = configure_logging()
                logger.exception("Unhandled exception")
                logger.error("s3_deletion_failed", backup_id=backup_id, error=str(e))
                raise

        # Remove from metadata
        if backup_id in self.metadata.get("backups", {}):
            del self.metadata["backups"][backup_id]
            self._save_metadata()

    def _load_metadata(self) -> dict[str, Any]:
        """Load backup metadata"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file) as f:
                return json.load(f)
        return {"backups": {}}

    def _save_metadata(self):
        """Save backup metadata"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _update_metadata(
        self, backup_id: str, backup_type: str, timestamp: datetime, data_hash: str
    ):
        """Update backup metadata"""
        if "backups" not in self.metadata:
            self.metadata["backups"] = {}

        self.metadata["backups"][backup_id] = {
            "backup_type": backup_type,
            "timestamp": timestamp.isoformat(),
            "data_hash": data_hash,
        }

        self._save_metadata()

    def _run_scheduler(self):
        """Run the backup scheduler"""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


class DisasterRecoveryOrchestrator:
    """Orchestrates disaster recovery procedures"""

    def __init__(self, backup_manager: BackupManager):
        self.backup_manager = backup_manager
        self.recovery_points = {}

    def create_recovery_point(self, name: str, components: list[str]) -> str:
        """Create a coordinated recovery point across multiple components"""
        _ = "rp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        _ = {}

        try:
            # Collect data from all components
            for component in components:
                data = self._collect_component_data(component)
                recovery_data[component] = data

            # Create unified backup
            _ = self.backup_manager.create_backup(recovery_data, "recovery_point")

            # Store recovery point info
            self.recovery_points[recovery_point_id] = {
                "name": name,
                "backup_id": backup_id,
                "components": components,
                "timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(
                "recovery_point_created",
                recovery_point_id=recovery_point_id,
                name=name,
                components=components,
            )

            return recovery_point_id

        except KeyError as e:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            logger.error("recovery_point_creation_failed", name=name, error=str(e))
            raise
            raise

    def restore_recovery_point(self, recovery_point_id: str) -> dict[str, Any]:
        """Restore system state from a recovery point"""
        if recovery_point_id not in self.recovery_points:
            raise ValueError("Recovery point {recovery_point_id} not found")

        _ = self.recovery_points[recovery_point_id]

        try:
            # Restore backup
            _ = self.backup_manager.restore_backup(recovery_info["backup_id"])

            # Restore each component
            _ = {}
            for component, data in recovery_data.items():
                success = self._restore_component_data(component, data)
                results[component] = success

            logger.info(
                "recovery_point_restored",
                recovery_point_id=recovery_point_id,
                results=results,
            )

            return results

        except KeyError as e:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            logger.error(
                "recovery_point_restoration_failed",
                recovery_point_id=recovery_point_id,
                error=str(e),
            )
            raise
            raise

    def _collect_component_data(self, component: str) -> dict[str, Any]:
        """Collect data from a component for backup"""
        # This would be implemented based on specific components
        # For now, return placeholder
        return {
            "component": component,
            "data": "placeholder",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _restore_component_data(self, component: str, data: dict[str, Any]) -> bool:
        """Restore data to a component"""
        # This would be implemented based on specific components
        # For now, return success
        return True
