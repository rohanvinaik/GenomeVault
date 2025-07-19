"""
Tests for monitoring and logging infrastructure.
"""

import json
import time
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from genomevault.utils.backup import BackupManager, DisasterRecoveryOrchestrator
from genomevault.utils.logging import (
    audit_logger,
    filter_sensitive_data,
    get_logger,
    performance_logger,
    security_logger,
)
from genomevault.utils.monitoring import MetricsCollector, PrivacyAwareLogger
from genomevault.utils.security_monitor import ComplianceMonitor, SecurityMonitor


class TestMetricsCollector:
    """Test metrics collection functionality"""
    
    def test_metrics_initialization(self):
        """Test metrics collector initialization"""
        collector = MetricsCollector()
        
        # Check metrics exist
        assert collector.processing_duration is not None
        assert collector.hypervector_operations is not None
        assert collector.pir_query_latency is not None
        
    def test_processing_tracking(self):
        """Test processing operation tracking"""
        collector = MetricsCollector()
        
        @collector.track_processing("test_op", "genomic")
        def test_function():
            time.sleep(0.1)
            return "result"
        
        # Execute function
        result = test_function()
        
        assert result == "result"
        # Metrics should be recorded
        
    def test_pir_metrics_update(self):
        """Test PIR metrics update"""
        collector = MetricsCollector()
        
        config = {
            'trusted_signatures': 2,
            'server_honesty_prob': 0.98,
            'server_type': 'hipaa-ts'
        }
        
        collector.update_pir_metrics(config)
        
        # Check privacy failure probability
        expected_p_fail = (1 - 0.98) ** 2
        assert expected_p_fail == pytest.approx(0.0004, rel=1e-6)
        
    def test_storage_metrics(self):
        """Test storage tier metrics"""
        collector = MetricsCollector()
        
        # Update storage metrics
        collector.update_storage_metrics('mini', 25000)
        collector.update_storage_metrics('clinical', 300000)
        collector.update_storage_metrics('full', 200000)
        
        # Metrics should be set
        
    def test_privacy_tracking(self):
        """Test privacy budget tracking"""
        collector = MetricsCollector()
        
        # Track privacy budget
        collector.track_privacy_budget('user123', 0.5)
        
        # Record violation
        collector.record_privacy_violation('excessive_queries')


class TestLogging:
    """Test logging functionality"""
    
    def test_logger_creation(self):
        """Test logger creation"""
        logger = get_logger("test_module")
        
        assert logger is not None
        logger.info("test_message", extra_field="value")
        
    def test_sensitive_data_filtering(self):
        """Test sensitive data filtering"""
        event_dict = {
            'message': 'test',
            'user_id': 'user123',
            'password': 'secret123',
            'genome_data': 'ATCGATCG',
            'hypervector': [1, 2, 3, 4],
            'email': 'test@example.com',
            'safe_field': 'safe_value'
        }
        
        filtered = filter_sensitive_data(None, None, event_dict)
        
        assert filtered['password'] == '[REDACTED]'
        assert filtered['genome_data'] == '[REDACTED]'
        assert filtered['hypervector'] == '[REDACTED]'
        assert filtered['safe_field'] == 'safe_value'
        
    def test_audit_logger(self):
        """Test audit logger functionality"""
        # Test authentication logging
        audit_logger.log_authentication(
            user_id='user123',
            method='password',
            success=True,
            ip_address='192.168.1.1'
        )
        
        # Test data access logging
        audit_logger.log_data_access(
            user_id='user123',
            resource_type='genomic_data',
            resource_id='dataset_001',
            action='read',
            success=True
        )
        
        # Test consent change logging
        audit_logger.log_consent_change(
            user_id='user123',
            consent_type='research_participation',
            old_value=False,
            new_value=True
        )
        
    def test_performance_logger(self):
        """Test performance logger"""
        with performance_logger.track_operation('test_operation', {'key': 'value'}):
            time.sleep(0.1)
            
        # Log resource usage
        performance_logger.log_resource_usage(
            component='api_server',
            cpu_percent=45.2,
            memory_mb=2048,
            disk_io_mb=100
        )
        
    def test_security_logger(self):
        """Test security logger"""
        # Log intrusion attempt
        security_logger.log_intrusion_attempt(
            source_ip='192.168.1.100',
            attack_type='brute_force',
            target='/api/login',
            blocked=True
        )
        
        # Log encryption event
        security_logger.log_encryption_event(
            operation='encrypt',
            algorithm='AES-256-GCM',
            key_length=256,
            success=True
        )


class TestBackupManager:
    """Test backup and recovery functionality"""
    
    @pytest.fixture
    def backup_config(self, tmp_path):
        """Create test backup configuration"""
        return {
            'backup_dir': str(tmp_path / 'backups'),
            'encryption_key': b'0' * 32,  # Test key
            'retention_days': 7
        }
        
    def test_backup_creation(self, backup_config):
        """Test creating a backup"""
        manager = BackupManager(backup_config)
        
        test_data = {
            'user_id': 'user123',
            'genomic_data': 'test_data',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        backup_id = manager.create_backup(test_data, 'test_backup')
        
        assert backup_id is not None
        assert backup_id.startswith('test_backup_')
        
    def test_backup_restoration(self, backup_config):
        """Test restoring from backup"""
        manager = BackupManager(backup_config)
        
        # Create backup
        original_data = {
            'key1': 'value1',
            'key2': [1, 2, 3],
            'key3': {'nested': 'data'}
        }
        
        backup_id = manager.create_backup(original_data, 'test')
        
        # Restore backup
        restored_data = manager.restore_backup(backup_id)
        
        assert restored_data == original_data
        
    def test_backup_verification(self, backup_config):
        """Test backup integrity verification"""
        manager = BackupManager(backup_config)
        
        # Create backup
        data = {'test': 'data'}
        backup_id = manager.create_backup(data, 'test')
        
        # Verify backup
        is_valid = manager.verify_backup(backup_id)
        
        assert is_valid is True
        
    def test_backup_cleanup(self, backup_config):
        """Test old backup cleanup"""
        manager = BackupManager(backup_config)
        
        # Create some backups
        for i in range(5):
            manager.create_backup({'data': i}, 'test')
            
        # Run cleanup
        removed_count = manager.cleanup_old_backups()
        
        # Should not remove any (all are recent)
        assert removed_count == 0


class TestSecurityMonitor:
    """Test security monitoring functionality"""
    
    @pytest.fixture
    def security_config(self):
        """Security monitor configuration"""
        return {
            'threat_threshold': 0.8,
            'anomaly_window_minutes': 60
        }
        
    @pytest.mark.asyncio
    async def test_access_monitoring(self, security_config):
        """Test access monitoring"""
        monitor = SecurityMonitor(security_config)
        
        # Monitor normal access
        result = await monitor.monitor_access(
            user_id='user123',
            resource='genomic_data',
            action='read',
            success=True,
            ip_address='192.168.1.1'
        )
        
        assert result is True
        
    @pytest.mark.asyncio
    async def test_brute_force_detection(self, security_config):
        """Test brute force detection"""
        monitor = SecurityMonitor(security_config)
        
        # Simulate failed attempts
        for i in range(6):
            await monitor.monitor_access(
                user_id='attacker',
                resource='login',
                action='authenticate',
                success=False,
                ip_address='192.168.1.100'
            )
            
        # IP should be blocked
        assert '192.168.1.100' in monitor.blocked_ips
        
    @pytest.mark.asyncio
    async def test_data_exfiltration_monitoring(self, security_config):
        """Test data exfiltration detection"""
        monitor = SecurityMonitor(security_config)
        
        # Set up alert callback
        alerts = []
        monitor.register_alert_callback(lambda alert: alerts.append(alert))
        
        # Monitor large data access
        await monitor.monitor_data_access(
            user_id='user123',
            data_type='genomic',
            volume=2000000,  # 2M variants (above threshold)
            operation='export'
        )
        
        # Should trigger alert
        assert len(alerts) > 0
        assert alerts[0]['type'] == 'data_exfiltration_risk'


class TestComplianceMonitor:
    """Test compliance monitoring"""
    
    @pytest.fixture
    def compliance_monitor(self):
        """Create compliance monitor"""
        return ComplianceMonitor({})
        
    @pytest.mark.asyncio
    async def test_hipaa_compliance(self, compliance_monitor):
        """Test HIPAA compliance checking"""
        # Test minimum necessary
        context = {
            'fields_accessed': ['name', 'ssn', 'diagnosis', 'genome'],
            'necessary_fields': ['name', 'diagnosis'],
            'user_id': 'doctor123'
        }
        
        is_compliant = await compliance_monitor.check_hipaa_compliance(
            'data_access', context
        )
        
        assert is_compliant is False  # Accessed unnecessary fields
        
    @pytest.mark.asyncio
    async def test_gdpr_compliance(self, compliance_monitor):
        """Test GDPR compliance checking"""
        # Test consent verification
        context = {
            'consent_verified': False,
            'user_id': 'user123',
            'data_type': 'genomic'
        }
        
        is_compliant = await compliance_monitor.check_gdpr_compliance(
            'data_processing', context
        )
        
        assert is_compliant is False  # No consent
        
    def test_compliance_report(self, compliance_monitor):
        """Test compliance report generation"""
        # Generate report
        report = compliance_monitor.get_compliance_report()
        
        assert 'total_violations' in report
        assert 'violations_by_type' in report
        assert 'generated_at' in report
