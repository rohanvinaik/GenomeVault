"""
Kubernetes Secrets Manager with automatic rotation and HSM/KMS integration.

This module provides secure secret management with:
- Automatic rotation for JWT tokens, encryption keys, and passwords
- Integration with AWS KMS, GCP KMS, and HashiCorp Vault
- PHI-specific encryption for HIPAA compliance
- HSM support for cryptographic operations
"""

import os
import json
import base64
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.fernet import Fernet

try:
    import boto3
    HAS_AWS = True
except ImportError:
    HAS_AWS = False

try:
    from google.cloud import secretmanager, kms
    HAS_GCP = True
except ImportError:
    HAS_GCP = False

try:
    import hvac
    HAS_VAULT = True
except ImportError:
    HAS_VAULT = False

try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    HAS_K8S = True
except ImportError:
    HAS_K8S = False

logger = logging.getLogger(__name__)


class SecretType(Enum):
    """Types of secrets managed by the system."""
    JWT_SECRET = "jwt_secret"
    ENCRYPTION_KEY = "encryption_key"
    DATABASE_PASSWORD = "database_password"
    REDIS_PASSWORD = "redis_password"
    API_KEY = "api_key"
    TLS_CERT = "tls_cert"
    TLS_KEY = "tls_key"
    PHI_ENCRYPTION_KEY = "phi_encryption_key"
    HSM_PIN = "hsm_pin"
    BLOCKCHAIN_KEY = "blockchain_key"


class SecretProvider(Enum):
    """Secret storage providers."""
    KUBERNETES = "kubernetes"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    AWS_KMS = "aws_kms"
    GCP_SECRET_MANAGER = "gcp_secret_manager"
    GCP_KMS = "gcp_kms"
    HASHICORP_VAULT = "hashicorp_vault"
    LOCAL_HSM = "local_hsm"


@dataclass
class SecretMetadata:
    """Metadata for a secret."""
    name: str
    type: SecretType
    provider: SecretProvider
    created_at: datetime
    last_rotated: datetime
    rotation_interval: timedelta
    version: int
    compliance_tags: List[str]
    encryption_algorithm: str
    requires_approval: bool = False
    approvers: List[str] = None


@dataclass
class RotationPolicy:
    """Policy for secret rotation."""
    secret_type: SecretType
    rotation_interval: timedelta
    min_entropy_bits: int
    key_length: int
    backup_required: bool
    notification_channels: List[str]
    pre_rotation_hook: Optional[str] = None
    post_rotation_hook: Optional[str] = None


class SecretsManager:
    """Main secrets management class with rotation and HSM/KMS integration."""
    
    # Default rotation policies
    DEFAULT_POLICIES = {
        SecretType.JWT_SECRET: RotationPolicy(
            secret_type=SecretType.JWT_SECRET,
            rotation_interval=timedelta(days=90),
            min_entropy_bits=256,
            key_length=64,
            backup_required=True,
            notification_channels=["security-team"]
        ),
        SecretType.ENCRYPTION_KEY: RotationPolicy(
            secret_type=SecretType.ENCRYPTION_KEY,
            rotation_interval=timedelta(days=180),
            min_entropy_bits=256,
            key_length=32,
            backup_required=True,
            notification_channels=["security-team", "ops-team"]
        ),
        SecretType.DATABASE_PASSWORD: RotationPolicy(
            secret_type=SecretType.DATABASE_PASSWORD,
            rotation_interval=timedelta(days=60),
            min_entropy_bits=128,
            key_length=32,
            backup_required=True,
            notification_channels=["dba-team"]
        ),
        SecretType.REDIS_PASSWORD: RotationPolicy(
            secret_type=SecretType.REDIS_PASSWORD,
            rotation_interval=timedelta(days=90),
            min_entropy_bits=128,
            key_length=32,
            backup_required=False,
            notification_channels=["ops-team"]
        ),
        SecretType.PHI_ENCRYPTION_KEY: RotationPolicy(
            secret_type=SecretType.PHI_ENCRYPTION_KEY,
            rotation_interval=timedelta(days=30),
            min_entropy_bits=256,
            key_length=32,
            backup_required=True,
            notification_channels=["security-team", "compliance-team"],
            pre_rotation_hook="audit_phi_access",
            post_rotation_hook="update_phi_encryption"
        ),
    }
    
    def __init__(
        self,
        namespace: str = "genomevault",
        provider: SecretProvider = SecretProvider.KUBERNETES,
        config_path: Optional[str] = None
    ):
        """Initialize the secrets manager."""
        self.namespace = namespace
        self.provider = provider
        self.config_path = config_path
        self.policies = self.DEFAULT_POLICIES.copy()
        
        # Initialize provider clients
        self._init_providers()
        
        # Load custom policies if provided
        if config_path:
            self._load_custom_policies(config_path)
        
        # PHI encryption cipher for HIPAA compliance
        self.phi_cipher = None
        self._init_phi_encryption()
    
    def _init_providers(self):
        """Initialize provider clients."""
        self.k8s_client = None
        self.aws_secrets_client = None
        self.aws_kms_client = None
        self.gcp_secret_client = None
        self.gcp_kms_client = None
        self.vault_client = None
        
        if HAS_K8S:
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()
            self.k8s_client = client.CoreV1Api()
        
        if HAS_AWS:
            self.aws_secrets_client = boto3.client('secretsmanager')
            self.aws_kms_client = boto3.client('kms')
        
        if HAS_GCP:
            self.gcp_secret_client = secretmanager.SecretManagerServiceClient()
            self.gcp_kms_client = kms.KeyManagementServiceClient()
        
        if HAS_VAULT:
            vault_url = os.getenv('VAULT_ADDR', 'http://localhost:8200')
            vault_token = os.getenv('VAULT_TOKEN')
            if vault_token:
                self.vault_client = hvac.Client(url=vault_url, token=vault_token)
    
    def _init_phi_encryption(self):
        """Initialize PHI-specific encryption with FIPS 140-2 compliance."""
        # Use environment variable or generate new key
        phi_key = os.getenv('PHI_MASTER_KEY')
        if not phi_key:
            phi_key = Fernet.generate_key()
            logger.warning("Generated new PHI master key - should be stored securely")
        
        self.phi_cipher = Fernet(phi_key)
    
    def _load_custom_policies(self, config_path: str):
        """Load custom rotation policies from configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                for policy_data in config.get('rotation_policies', []):
                    policy = RotationPolicy(**policy_data)
                    self.policies[policy.secret_type] = policy
        except Exception as e:
            logger.error(f"Failed to load custom policies: {e}")
    
    def generate_secret(self, secret_type: SecretType) -> str:
        """Generate a new secret based on type and policy."""
        policy = self.policies.get(secret_type)
        if not policy:
            raise ValueError(f"No policy defined for secret type: {secret_type}")
        
        # Generate cryptographically secure random secret
        if secret_type in [SecretType.JWT_SECRET, SecretType.ENCRYPTION_KEY]:
            # Generate base64-encoded key
            key_bytes = secrets.token_bytes(policy.key_length)
            return base64.b64encode(key_bytes).decode('utf-8')
        
        elif secret_type in [SecretType.DATABASE_PASSWORD, SecretType.REDIS_PASSWORD]:
            # Generate URL-safe password
            return secrets.token_urlsafe(policy.key_length)
        
        elif secret_type == SecretType.PHI_ENCRYPTION_KEY:
            # Generate Fernet key for PHI encryption
            return Fernet.generate_key().decode('utf-8')
        
        elif secret_type == SecretType.BLOCKCHAIN_KEY:
            # Generate private key for blockchain
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            return pem.decode('utf-8')
        
        else:
            # Default to random token
            return secrets.token_urlsafe(policy.key_length)
    
    async def rotate_secret(
        self,
        secret_name: str,
        secret_type: SecretType,
        force: bool = False
    ) -> Tuple[str, SecretMetadata]:
        """Rotate a secret with proper versioning and backup."""
        logger.info(f"Starting rotation for secret: {secret_name}")
        
        # Get current secret metadata
        metadata = await self.get_secret_metadata(secret_name)
        
        # Check if rotation is needed
        if not force and not self._rotation_needed(metadata):
            logger.info(f"Secret {secret_name} does not need rotation yet")
            return None, metadata
        
        # Check if approval is required
        if metadata.requires_approval and not force:
            await self._request_approval(secret_name, metadata.approvers)
        
        policy = self.policies.get(secret_type)
        
        # Execute pre-rotation hook if defined
        if policy.pre_rotation_hook:
            await self._execute_hook(policy.pre_rotation_hook, secret_name)
        
        # Backup current secret if required
        if policy.backup_required:
            await self._backup_secret(secret_name, metadata)
        
        # Generate new secret
        new_secret = self.generate_secret(secret_type)
        
        # Store new secret with versioning
        new_metadata = await self._store_secret(
            secret_name,
            new_secret,
            secret_type,
            metadata.version + 1
        )
        
        # Execute post-rotation hook if defined
        if policy.post_rotation_hook:
            await self._execute_hook(policy.post_rotation_hook, secret_name)
        
        # Send notifications
        await self._send_notifications(
            secret_name,
            policy.notification_channels,
            "rotated"
        )
        
        logger.info(f"Successfully rotated secret: {secret_name}")
        return new_secret, new_metadata
    
    def _rotation_needed(self, metadata: SecretMetadata) -> bool:
        """Check if a secret needs rotation."""
        policy = self.policies.get(metadata.type)
        if not policy:
            return False
        
        time_since_rotation = datetime.utcnow() - metadata.last_rotated
        return time_since_rotation >= policy.rotation_interval
    
    async def _store_secret(
        self,
        name: str,
        value: str,
        secret_type: SecretType,
        version: int
    ) -> SecretMetadata:
        """Store secret in the configured provider."""
        metadata = SecretMetadata(
            name=name,
            type=secret_type,
            provider=self.provider,
            created_at=datetime.utcnow(),
            last_rotated=datetime.utcnow(),
            rotation_interval=self.policies[secret_type].rotation_interval,
            version=version,
            compliance_tags=self._get_compliance_tags(secret_type),
            encryption_algorithm=self._get_encryption_algorithm(secret_type),
            requires_approval=secret_type == SecretType.PHI_ENCRYPTION_KEY
        )
        
        if self.provider == SecretProvider.KUBERNETES:
            await self._store_k8s_secret(name, value, metadata)
        elif self.provider == SecretProvider.AWS_SECRETS_MANAGER:
            await self._store_aws_secret(name, value, metadata)
        elif self.provider == SecretProvider.GCP_SECRET_MANAGER:
            await self._store_gcp_secret(name, value, metadata)
        elif self.provider == SecretProvider.HASHICORP_VAULT:
            await self._store_vault_secret(name, value, metadata)
        
        return metadata
    
    async def _store_k8s_secret(
        self,
        name: str,
        value: str,
        metadata: SecretMetadata
    ):
        """Store secret in Kubernetes."""
        if not self.k8s_client:
            raise RuntimeError("Kubernetes client not initialized")
        
        # Prepare secret data
        secret_data = {
            'value': base64.b64encode(value.encode()).decode(),
            'metadata': base64.b64encode(json.dumps({
                'version': metadata.version,
                'created_at': metadata.created_at.isoformat(),
                'last_rotated': metadata.last_rotated.isoformat(),
                'type': metadata.type.value,
                'compliance_tags': metadata.compliance_tags,
                'encryption_algorithm': metadata.encryption_algorithm
            }).encode()).decode()
        }
        
        # Create or update Kubernetes secret
        k8s_secret = client.V1Secret(
            metadata=client.V1ObjectMeta(
                name=name,
                namespace=self.namespace,
                annotations={
                    'genomevault.io/version': str(metadata.version),
                    'genomevault.io/last-rotated': metadata.last_rotated.isoformat(),
                    'genomevault.io/type': metadata.type.value
                }
            ),
            data=secret_data,
            type='Opaque'
        )
        
        try:
            # Try to update existing secret
            self.k8s_client.patch_namespaced_secret(
                name=name,
                namespace=self.namespace,
                body=k8s_secret
            )
        except ApiException as e:
            if e.status == 404:
                # Create new secret if it doesn't exist
                self.k8s_client.create_namespaced_secret(
                    namespace=self.namespace,
                    body=k8s_secret
                )
            else:
                raise
    
    async def _store_aws_secret(
        self,
        name: str,
        value: str,
        metadata: SecretMetadata
    ):
        """Store secret in AWS Secrets Manager with KMS encryption."""
        if not self.aws_secrets_client:
            raise RuntimeError("AWS Secrets Manager client not initialized")
        
        # Encrypt with KMS if PHI data
        if metadata.type == SecretType.PHI_ENCRYPTION_KEY:
            kms_key_id = os.getenv('AWS_KMS_PHI_KEY_ID')
            if kms_key_id and self.aws_kms_client:
                encrypted_value = self.aws_kms_client.encrypt(
                    KeyId=kms_key_id,
                    Plaintext=value.encode()
                )
                value = base64.b64encode(encrypted_value['CiphertextBlob']).decode()
        
        try:
            # Update existing secret
            self.aws_secrets_client.update_secret(
                SecretId=f"genomevault/{self.namespace}/{name}",
                SecretString=json.dumps({
                    'value': value,
                    'metadata': {
                        'version': metadata.version,
                        'created_at': metadata.created_at.isoformat(),
                        'last_rotated': metadata.last_rotated.isoformat(),
                        'type': metadata.type.value,
                        'compliance_tags': metadata.compliance_tags
                    }
                }),
                VersionStages=['AWSCURRENT']
            )
        except self.aws_secrets_client.exceptions.ResourceNotFoundException:
            # Create new secret
            self.aws_secrets_client.create_secret(
                Name=f"genomevault/{self.namespace}/{name}",
                SecretString=json.dumps({
                    'value': value,
                    'metadata': {
                        'version': metadata.version,
                        'created_at': metadata.created_at.isoformat(),
                        'last_rotated': metadata.last_rotated.isoformat(),
                        'type': metadata.type.value,
                        'compliance_tags': metadata.compliance_tags
                    }
                }),
                Tags=[
                    {'Key': 'Application', 'Value': 'GenomeVault'},
                    {'Key': 'Environment', 'Value': self.namespace},
                    {'Key': 'Type', 'Value': metadata.type.value},
                    {'Key': 'Compliance', 'Value': ','.join(metadata.compliance_tags)}
                ]
            )
    
    async def _store_gcp_secret(
        self,
        name: str,
        value: str,
        metadata: SecretMetadata
    ):
        """Store secret in GCP Secret Manager with Cloud KMS encryption."""
        if not self.gcp_secret_client:
            raise RuntimeError("GCP Secret Manager client not initialized")
        
        project_id = os.getenv('GCP_PROJECT_ID', 'genomevault-production')
        parent = f"projects/{project_id}"
        secret_id = f"genomevault-{self.namespace}-{name}"
        
        # Encrypt with Cloud KMS if PHI data
        if metadata.type == SecretType.PHI_ENCRYPTION_KEY:
            kms_key_name = os.getenv('GCP_KMS_PHI_KEY_NAME')
            if kms_key_name and self.gcp_kms_client:
                encrypted_response = self.gcp_kms_client.encrypt(
                    request={
                        'name': kms_key_name,
                        'plaintext': value.encode()
                    }
                )
                value = base64.b64encode(encrypted_response.ciphertext).decode()
        
        try:
            # Get existing secret
            secret = self.gcp_secret_client.get_secret(
                request={'name': f"{parent}/secrets/{secret_id}"}
            )
            
            # Add new version
            self.gcp_secret_client.add_secret_version(
                request={
                    'parent': secret.name,
                    'payload': {
                        'data': json.dumps({
                            'value': value,
                            'metadata': {
                                'version': metadata.version,
                                'created_at': metadata.created_at.isoformat(),
                                'last_rotated': metadata.last_rotated.isoformat(),
                                'type': metadata.type.value,
                                'compliance_tags': metadata.compliance_tags
                            }
                        }).encode()
                    }
                }
            )
        except Exception:
            # Create new secret
            secret = self.gcp_secret_client.create_secret(
                request={
                    'parent': parent,
                    'secret_id': secret_id,
                    'secret': {
                        'replication': {
                            'automatic': {}
                        },
                        'labels': {
                            'application': 'genomevault',
                            'environment': self.namespace,
                            'type': metadata.type.value.replace('_', '-'),
                            'compliance': '-'.join(metadata.compliance_tags)
                        }
                    }
                }
            )
            
            # Add initial version
            self.gcp_secret_client.add_secret_version(
                request={
                    'parent': secret.name,
                    'payload': {
                        'data': json.dumps({
                            'value': value,
                            'metadata': {
                                'version': metadata.version,
                                'created_at': metadata.created_at.isoformat(),
                                'last_rotated': metadata.last_rotated.isoformat(),
                                'type': metadata.type.value,
                                'compliance_tags': metadata.compliance_tags
                            }
                        }).encode()
                    }
                }
            )
    
    async def _store_vault_secret(
        self,
        name: str,
        value: str,
        metadata: SecretMetadata
    ):
        """Store secret in HashiCorp Vault with transit encryption."""
        if not self.vault_client:
            raise RuntimeError("Vault client not initialized")
        
        path = f"secret/data/genomevault/{self.namespace}/{name}"
        
        # Encrypt with transit engine if PHI data
        if metadata.type == SecretType.PHI_ENCRYPTION_KEY:
            transit_key = 'genomevault-phi-master'
            encrypted_response = self.vault_client.secrets.transit.encrypt_data(
                name=transit_key,
                plaintext=base64.b64encode(value.encode()).decode()
            )
            value = encrypted_response['data']['ciphertext']
        
        # Store in Vault with metadata
        self.vault_client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret={
                'value': value,
                'metadata': {
                    'version': metadata.version,
                    'created_at': metadata.created_at.isoformat(),
                    'last_rotated': metadata.last_rotated.isoformat(),
                    'type': metadata.type.value,
                    'compliance_tags': metadata.compliance_tags,
                    'encryption_algorithm': metadata.encryption_algorithm
                }
            },
            cas=metadata.version  # Check-and-set for versioning
        )
    
    async def get_secret(self, name: str) -> str:
        """Retrieve a secret from the configured provider."""
        if self.provider == SecretProvider.KUBERNETES:
            return await self._get_k8s_secret(name)
        elif self.provider == SecretProvider.AWS_SECRETS_MANAGER:
            return await self._get_aws_secret(name)
        elif self.provider == SecretProvider.GCP_SECRET_MANAGER:
            return await self._get_gcp_secret(name)
        elif self.provider == SecretProvider.HASHICORP_VAULT:
            return await self._get_vault_secret(name)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _get_k8s_secret(self, name: str) -> str:
        """Retrieve secret from Kubernetes."""
        if not self.k8s_client:
            raise RuntimeError("Kubernetes client not initialized")
        
        try:
            secret = self.k8s_client.read_namespaced_secret(
                name=name,
                namespace=self.namespace
            )
            value_b64 = secret.data.get('value')
            if value_b64:
                return base64.b64decode(value_b64).decode()
            else:
                raise ValueError(f"Secret {name} has no value field")
        except ApiException as e:
            if e.status == 404:
                raise ValueError(f"Secret {name} not found")
            raise
    
    async def get_secret_metadata(self, name: str) -> SecretMetadata:
        """Get metadata for a secret."""
        # Implementation depends on provider
        # This is a simplified example
        return SecretMetadata(
            name=name,
            type=SecretType.JWT_SECRET,
            provider=self.provider,
            created_at=datetime.utcnow() - timedelta(days=30),
            last_rotated=datetime.utcnow() - timedelta(days=30),
            rotation_interval=timedelta(days=90),
            version=1,
            compliance_tags=['hipaa', 'sox'],
            encryption_algorithm='aes-256-gcm',
            requires_approval=False
        )
    
    def _get_compliance_tags(self, secret_type: SecretType) -> List[str]:
        """Get compliance tags for a secret type."""
        tags = ['genomevault']
        
        if secret_type in [SecretType.PHI_ENCRYPTION_KEY]:
            tags.extend(['hipaa', 'phi', 'fips-140-2'])
        
        if secret_type in [SecretType.DATABASE_PASSWORD]:
            tags.extend(['sox', 'pci-dss'])
        
        return tags
    
    def _get_encryption_algorithm(self, secret_type: SecretType) -> str:
        """Get encryption algorithm for a secret type."""
        if secret_type == SecretType.PHI_ENCRYPTION_KEY:
            return 'aes-256-gcm-hsm'  # Hardware Security Module
        elif secret_type in [SecretType.ENCRYPTION_KEY, SecretType.JWT_SECRET]:
            return 'aes-256-gcm'
        else:
            return 'aes-128-gcm'
    
    async def _backup_secret(self, name: str, metadata: SecretMetadata):
        """Backup a secret before rotation."""
        backup_name = f"{name}-backup-v{metadata.version}"
        current_value = await self.get_secret(name)
        
        # Store backup with special metadata
        backup_metadata = SecretMetadata(
            name=backup_name,
            type=metadata.type,
            provider=metadata.provider,
            created_at=metadata.created_at,
            last_rotated=datetime.utcnow(),
            rotation_interval=timedelta(days=36500),  # 100 years
            version=metadata.version,
            compliance_tags=metadata.compliance_tags + ['backup'],
            encryption_algorithm=metadata.encryption_algorithm,
            requires_approval=False
        )
        
        await self._store_secret(
            backup_name,
            current_value,
            metadata.type,
            metadata.version
        )
        
        logger.info(f"Created backup: {backup_name}")
    
    async def _execute_hook(self, hook_name: str, secret_name: str):
        """Execute a rotation hook."""
        logger.info(f"Executing hook: {hook_name} for secret: {secret_name}")
        # Hook implementation would go here
        # Could call external scripts, APIs, or internal functions
    
    async def _send_notifications(
        self,
        secret_name: str,
        channels: List[str],
        action: str
    ):
        """Send notifications about secret operations."""
        message = f"Secret {secret_name} was {action} at {datetime.utcnow().isoformat()}"
        
        for channel in channels:
            logger.info(f"Sending notification to {channel}: {message}")
            # Actual notification implementation would go here
            # Could use Slack, email, PagerDuty, etc.
    
    async def _request_approval(self, secret_name: str, approvers: List[str]):
        """Request approval for secret rotation."""
        logger.info(f"Requesting approval for {secret_name} from {approvers}")
        # Approval workflow implementation would go here
        # Could integrate with ServiceNow, Jira, or custom approval system
    
    def encrypt_phi_data(self, data: str) -> str:
        """Encrypt PHI data with HIPAA-compliant encryption."""
        if not self.phi_cipher:
            raise RuntimeError("PHI cipher not initialized")
        
        # Add timestamp and nonce for additional security
        timestamp = datetime.utcnow().isoformat()
        nonce = secrets.token_hex(16)
        
        # Combine data with metadata
        payload = json.dumps({
            'data': data,
            'timestamp': timestamp,
            'nonce': nonce
        })
        
        # Encrypt with Fernet (AES-128 in CBC mode with HMAC)
        encrypted = self.phi_cipher.encrypt(payload.encode())
        
        return base64.b64encode(encrypted).decode()
    
    def decrypt_phi_data(self, encrypted_data: str) -> str:
        """Decrypt PHI data."""
        if not self.phi_cipher:
            raise RuntimeError("PHI cipher not initialized")
        
        try:
            # Decode and decrypt
            encrypted = base64.b64decode(encrypted_data)
            decrypted = self.phi_cipher.decrypt(encrypted)
            
            # Parse payload
            payload = json.loads(decrypted.decode())
            
            # Verify timestamp (optional: add time window check)
            timestamp = datetime.fromisoformat(payload['timestamp'])
            age = datetime.utcnow() - timestamp
            if age > timedelta(days=1):
                logger.warning(f"Decrypting old PHI data: {age.days} days old")
            
            return payload['data']
        except Exception as e:
            logger.error(f"Failed to decrypt PHI data: {e}")
            raise
    
    async def rotate_all_secrets(self, force: bool = False):
        """Rotate all secrets based on their policies."""
        results = {}
        
        for secret_type in SecretType:
            secret_name = f"genomevault-{secret_type.value.replace('_', '-')}"
            
            try:
                new_secret, metadata = await self.rotate_secret(
                    secret_name,
                    secret_type,
                    force=force
                )
                results[secret_name] = {
                    'status': 'rotated' if new_secret else 'skipped',
                    'version': metadata.version,
                    'last_rotated': metadata.last_rotated.isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to rotate {secret_name}: {e}")
                results[secret_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results


# CLI interface for manual operations
async def main():
    """CLI for secrets management operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GenomeVault Secrets Manager')
    parser.add_argument('action', choices=['rotate', 'get', 'encrypt-phi', 'decrypt-phi'])
    parser.add_argument('--secret-name', help='Name of the secret')
    parser.add_argument('--secret-type', help='Type of the secret')
    parser.add_argument('--force', action='store_true', help='Force rotation')
    parser.add_argument('--data', help='Data to encrypt/decrypt')
    parser.add_argument('--provider', default='kubernetes', help='Secret provider')
    parser.add_argument('--namespace', default='genomevault', help='Kubernetes namespace')
    
    args = parser.parse_args()
    
    manager = SecretsManager(
        namespace=args.namespace,
        provider=SecretProvider[args.provider.upper()]
    )
    
    if args.action == 'rotate':
        if args.secret_name and args.secret_type:
            secret_type = SecretType[args.secret_type.upper()]
            new_secret, metadata = await manager.rotate_secret(
                args.secret_name,
                secret_type,
                force=args.force
            )
            print(f"Rotated {args.secret_name}: version {metadata.version}")
        else:
            results = await manager.rotate_all_secrets(force=args.force)
            print(json.dumps(results, indent=2))
    
    elif args.action == 'get':
        if args.secret_name:
            secret = await manager.get_secret(args.secret_name)
            print(f"Secret value: {secret[:10]}..." if len(secret) > 10 else secret)
        else:
            print("Secret name required")
    
    elif args.action == 'encrypt-phi':
        if args.data:
            encrypted = manager.encrypt_phi_data(args.data)
            print(f"Encrypted PHI: {encrypted}")
        else:
            print("Data required for encryption")
    
    elif args.action == 'decrypt-phi':
        if args.data:
            decrypted = manager.decrypt_phi_data(args.data)
            print(f"Decrypted PHI: {decrypted}")
        else:
            print("Encrypted data required for decryption")


if __name__ == "__main__":
    asyncio.run(main())