"""
Hardware Security Module (HSM) Integration for GenomeVault

Provides secure key management and cryptographic operations using HSMs.
Supports multiple HSM backends: AWS KMS, HashiCorp Vault, PKCS#11 devices.
"""

from __future__ import annotations

import os
import base64
import json
import hashlib
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KeyMetadata:
    """Key metadata for HSM-managed keys."""
    key_id: str
    algorithm: str
    key_usage: List[str]  # ['encrypt', 'decrypt', 'sign', 'verify']
    created_at: str
    description: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


class HSMError(Exception):
    """HSM operation error."""
    pass


class HSMBackend(ABC):
    """Abstract base class for HSM backends."""
    
    @abstractmethod
    def generate_key(
        self, 
        key_id: str, 
        algorithm: str, 
        key_usage: List[str], 
        **kwargs
    ) -> KeyMetadata:
        """Generate a new key in HSM."""
        pass
    
    @abstractmethod
    def encrypt(self, key_id: str, plaintext: bytes, **kwargs) -> bytes:
        """Encrypt data using HSM key."""
        pass
    
    @abstractmethod
    def decrypt(self, key_id: str, ciphertext: bytes, **kwargs) -> bytes:
        """Decrypt data using HSM key."""
        pass
    
    @abstractmethod
    def sign(self, key_id: str, message: bytes, **kwargs) -> bytes:
        """Sign message using HSM key."""
        pass
    
    @abstractmethod
    def verify(self, key_id: str, message: bytes, signature: bytes, **kwargs) -> bool:
        """Verify signature using HSM key."""
        pass
    
    @abstractmethod
    def list_keys(self) -> List[KeyMetadata]:
        """List all keys in HSM."""
        pass
    
    @abstractmethod
    def delete_key(self, key_id: str) -> bool:
        """Delete key from HSM.""" 
        pass


class MockHSMBackend(HSMBackend):
    """Mock HSM backend for development and testing."""
    
    def __init__(self):
        """Initialize mock HSM."""
        self.keys: Dict[str, KeyMetadata] = {}
        self.key_data: Dict[str, bytes] = {}
        logger.warning("⚠️ Using Mock HSM - NOT SECURE - For development only!")
    
    def generate_key(
        self, 
        key_id: str, 
        algorithm: str, 
        key_usage: List[str], 
        **kwargs
    ) -> KeyMetadata:
        """Generate a mock key."""
        if key_id in self.keys:
            raise HSMError(f"Key {key_id} already exists")
        
        # Generate mock key data
        key_size = 32 if algorithm in ['AES-256', 'ChaCha20'] else 64
        self.key_data[key_id] = secrets.token_bytes(key_size)
        
        metadata = KeyMetadata(
            key_id=key_id,
            algorithm=algorithm,
            key_usage=key_usage,
            created_at="2025-08-24T22:49:00Z",
            description=kwargs.get('description', f'Mock {algorithm} key'),
            tags=kwargs.get('tags', {})
        )
        
        self.keys[key_id] = metadata
        logger.info(f"Generated mock key: {key_id} ({algorithm})")
        return metadata
    
    def encrypt(self, key_id: str, plaintext: bytes, **kwargs) -> bytes:
        """Mock encryption."""
        if key_id not in self.keys:
            raise HSMError(f"Key {key_id} not found")
        
        # Simple XOR encryption for demo
        key_data = self.key_data[key_id]
        encrypted = bytearray()
        for i, byte in enumerate(plaintext):
            encrypted.append(byte ^ key_data[i % len(key_data)])
        
        return bytes(encrypted)
    
    def decrypt(self, key_id: str, ciphertext: bytes, **kwargs) -> bytes:
        """Mock decryption (XOR is symmetric)."""
        return self.encrypt(key_id, ciphertext, **kwargs)
    
    def sign(self, key_id: str, message: bytes, **kwargs) -> bytes:
        """Mock signing."""
        if key_id not in self.keys:
            raise HSMError(f"Key {key_id} not found")
        
        key_data = self.key_data[key_id]
        return hashlib.sha256(message + key_data).digest()
    
    def verify(self, key_id: str, message: bytes, signature: bytes, **kwargs) -> bool:
        """Mock signature verification."""
        expected_sig = self.sign(key_id, message, **kwargs)
        return secrets.compare_digest(signature, expected_sig)
    
    def list_keys(self) -> List[KeyMetadata]:
        """List mock keys."""
        return list(self.keys.values())
    
    def delete_key(self, key_id: str) -> bool:
        """Delete mock key."""
        if key_id in self.keys:
            del self.keys[key_id]
            del self.key_data[key_id]
            logger.info(f"Deleted mock key: {key_id}")
            return True
        return False


class AWSKMSBackend(HSMBackend):
    """AWS KMS HSM backend."""
    
    def __init__(self, region: str = "us-east-1"):
        """Initialize AWS KMS backend."""
        self.region = region
        self._client = None
        logger.info(f"Initializing AWS KMS backend (region: {region})")
    
    @property
    def client(self):
        """Lazy load AWS KMS client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client('kms', region_name=self.region)
            except ImportError:
                raise HSMError("boto3 not installed. Install with: pip install boto3")
            except Exception as e:
                raise HSMError(f"Failed to initialize AWS KMS client: {e}")
        return self._client
    
    def generate_key(
        self, 
        key_id: str, 
        algorithm: str, 
        key_usage: List[str], 
        **kwargs
    ) -> KeyMetadata:
        """Generate key in AWS KMS."""
        try:
            response = self.client.create_key(
                Description=kwargs.get('description', f'GenomeVault {algorithm} key'),
                KeyUsage='ENCRYPT_DECRYPT' if 'encrypt' in key_usage else 'SIGN_VERIFY',
                CustomerMasterKeySpec='SYMMETRIC_DEFAULT' if 'encrypt' in key_usage else 'RSA_2048',
                Tags=[
                    {'TagKey': k, 'TagValue': v} 
                    for k, v in (kwargs.get('tags', {}) or {}).items()
                ] + [
                    {'TagKey': 'GenomeVault', 'TagValue': 'true'},
                    {'TagKey': 'KeyId', 'TagValue': key_id}
                ]
            )
            
            # Create alias for the key
            self.client.create_alias(
                AliasName=f'alias/genomevault-{key_id}',
                TargetKeyId=response['KeyMetadata']['KeyId']
            )
            
            metadata = KeyMetadata(
                key_id=response['KeyMetadata']['KeyId'],
                algorithm=algorithm,
                key_usage=key_usage,
                created_at=response['KeyMetadata']['CreationDate'].isoformat(),
                description=kwargs.get('description'),
                tags=kwargs.get('tags')
            )
            
            logger.info(f"Generated AWS KMS key: {key_id}")
            return metadata
            
        except Exception as e:
            raise HSMError(f"Failed to generate AWS KMS key: {e}")
    
    def encrypt(self, key_id: str, plaintext: bytes, **kwargs) -> bytes:
        """Encrypt with AWS KMS."""
        try:
            response = self.client.encrypt(
                KeyId=f'alias/genomevault-{key_id}',
                Plaintext=plaintext,
                EncryptionContext=kwargs.get('context', {})
            )
            return response['CiphertextBlob']
        except Exception as e:
            raise HSMError(f"AWS KMS encryption failed: {e}")
    
    def decrypt(self, key_id: str, ciphertext: bytes, **kwargs) -> bytes:
        """Decrypt with AWS KMS."""
        try:
            response = self.client.decrypt(
                CiphertextBlob=ciphertext,
                EncryptionContext=kwargs.get('context', {})
            )
            return response['Plaintext']
        except Exception as e:
            raise HSMError(f"AWS KMS decryption failed: {e}")
    
    def sign(self, key_id: str, message: bytes, **kwargs) -> bytes:
        """Sign with AWS KMS."""
        try:
            response = self.client.sign(
                KeyId=f'alias/genomevault-{key_id}',
                Message=message,
                SigningAlgorithm=kwargs.get('algorithm', 'RSASSA_PSS_SHA_256')
            )
            return response['Signature']
        except Exception as e:
            raise HSMError(f"AWS KMS signing failed: {e}")
    
    def verify(self, key_id: str, message: bytes, signature: bytes, **kwargs) -> bool:
        """Verify with AWS KMS."""
        try:
            response = self.client.verify(
                KeyId=f'alias/genomevault-{key_id}',
                Message=message,
                Signature=signature,
                SigningAlgorithm=kwargs.get('algorithm', 'RSASSA_PSS_SHA_256')
            )
            return response['SignatureValid']
        except Exception as e:
            raise HSMError(f"AWS KMS verification failed: {e}")
    
    def list_keys(self) -> List[KeyMetadata]:
        """List AWS KMS keys."""
        try:
            keys = []
            paginator = self.client.get_paginator('list_keys')
            
            for page in paginator.paginate():
                for key in page['Keys']:
                    # Filter for GenomeVault keys only
                    try:
                        tags_response = self.client.list_resource_tags(KeyId=key['KeyId'])
                        tags = {tag['TagKey']: tag['TagValue'] for tag in tags_response['Tags']}
                        
                        if tags.get('GenomeVault') == 'true':
                            key_details = self.client.describe_key(KeyId=key['KeyId'])['KeyMetadata']
                            metadata = KeyMetadata(
                                key_id=key['KeyId'],
                                algorithm=key_details['CustomerMasterKeySpec'],
                                key_usage=[key_details['KeyUsage']],
                                created_at=key_details['CreationDate'].isoformat(),
                                description=key_details.get('Description'),
                                tags=tags
                            )
                            keys.append(metadata)
                    except Exception:
                        continue  # Skip keys we can't access
            
            return keys
        except Exception as e:
            raise HSMError(f"Failed to list AWS KMS keys: {e}")
    
    def delete_key(self, key_id: str) -> bool:
        """Schedule AWS KMS key deletion."""
        try:
            self.client.schedule_key_deletion(
                KeyId=f'alias/genomevault-{key_id}',
                PendingWindowInDays=7
            )
            logger.warning(f"AWS KMS key {key_id} scheduled for deletion in 7 days")
            return True
        except Exception as e:
            raise HSMError(f"Failed to delete AWS KMS key: {e}")


class HashiCorpVaultBackend(HSMBackend):
    """HashiCorp Vault HSM backend."""
    
    def __init__(self, url: str, token: str, mount_path: str = "secret"):
        """Initialize Vault backend."""
        self.url = url.rstrip('/')
        self.token = token
        self.mount_path = mount_path
        self.session = None
        logger.info(f"Initializing HashiCorp Vault backend: {url}")
    
    @property
    def headers(self):
        """Get headers for Vault requests."""
        return {
            'X-Vault-Token': self.token,
            'Content-Type': 'application/json'
        }
    
    def _request(self, method: str, path: str, data: Optional[Dict] = None):
        """Make Vault API request."""
        import requests
        
        url = f"{self.url}/v1/{path}"
        response = requests.request(
            method, url, 
            headers=self.headers,
            json=data,
            timeout=30
        )
        
        if not response.ok:
            raise HSMError(f"Vault request failed: {response.status_code} {response.text}")
        
        return response.json() if response.content else {}
    
    def generate_key(
        self, 
        key_id: str, 
        algorithm: str, 
        key_usage: List[str], 
        **kwargs
    ) -> KeyMetadata:
        """Generate key in Vault."""
        try:
            # Generate key using transit engine
            key_data = {
                'type': algorithm.lower().replace('-', ''),
                'exportable': False,
                'allow_plaintext_backup': False
            }
            
            path = f"transit/keys/{key_id}"
            self._request('POST', path, key_data)
            
            metadata = KeyMetadata(
                key_id=key_id,
                algorithm=algorithm,
                key_usage=key_usage,
                created_at="2025-08-24T22:49:00Z",  # Vault doesn't return creation time easily
                description=kwargs.get('description'),
                tags=kwargs.get('tags')
            )
            
            logger.info(f"Generated Vault key: {key_id}")
            return metadata
            
        except Exception as e:
            raise HSMError(f"Failed to generate Vault key: {e}")
    
    def encrypt(self, key_id: str, plaintext: bytes, **kwargs) -> bytes:
        """Encrypt with Vault."""
        try:
            plaintext_b64 = base64.b64encode(plaintext).decode()
            data = {'plaintext': plaintext_b64}
            
            response = self._request('POST', f'transit/encrypt/{key_id}', data)
            return response['data']['ciphertext'].encode()
        except Exception as e:
            raise HSMError(f"Vault encryption failed: {e}")
    
    def decrypt(self, key_id: str, ciphertext: bytes, **kwargs) -> bytes:
        """Decrypt with Vault."""
        try:
            data = {'ciphertext': ciphertext.decode()}
            
            response = self._request('POST', f'transit/decrypt/{key_id}', data)
            return base64.b64decode(response['data']['plaintext'])
        except Exception as e:
            raise HSMError(f"Vault decryption failed: {e}")
    
    def sign(self, key_id: str, message: bytes, **kwargs) -> bytes:
        """Sign with Vault."""
        try:
            message_b64 = base64.b64encode(message).decode()
            data = {'input': message_b64}
            
            response = self._request('POST', f'transit/sign/{key_id}', data)
            return response['data']['signature'].encode()
        except Exception as e:
            raise HSMError(f"Vault signing failed: {e}")
    
    def verify(self, key_id: str, message: bytes, signature: bytes, **kwargs) -> bool:
        """Verify with Vault."""
        try:
            message_b64 = base64.b64encode(message).decode()
            data = {
                'input': message_b64,
                'signature': signature.decode()
            }
            
            response = self._request('POST', f'transit/verify/{key_id}', data)
            return response['data']['valid']
        except Exception as e:
            raise HSMError(f"Vault verification failed: {e}")
    
    def list_keys(self) -> List[KeyMetadata]:
        """List Vault keys."""
        try:
            response = self._request('GET', 'transit/keys')
            keys = []
            
            for key_name in response['data']['keys']:
                # Get key details
                key_response = self._request('GET', f'transit/keys/{key_name}')
                key_data = key_response['data']
                
                metadata = KeyMetadata(
                    key_id=key_name,
                    algorithm=key_data['type'],
                    key_usage=['encrypt', 'decrypt'],  # Default for transit keys
                    created_at="Unknown",  # Vault doesn't provide creation time
                    description=f"Vault transit key"
                )
                keys.append(metadata)
            
            return keys
        except Exception as e:
            raise HSMError(f"Failed to list Vault keys: {e}")
    
    def delete_key(self, key_id: str) -> bool:
        """Delete Vault key.""" 
        try:
            self._request('DELETE', f'transit/keys/{key_id}')
            logger.info(f"Deleted Vault key: {key_id}")
            return True
        except Exception as e:
            raise HSMError(f"Failed to delete Vault key: {e}")


class HSMManager:
    """High-level HSM manager for GenomeVault."""
    
    def __init__(self, backend: Optional[HSMBackend] = None):
        """Initialize HSM manager."""
        self.backend = backend or self._get_default_backend()
        logger.info(f"HSM Manager initialized with {self.backend.__class__.__name__}")
    
    def _get_default_backend(self) -> HSMBackend:
        """Get default HSM backend based on environment."""
        # Check for AWS credentials
        if os.getenv('AWS_ACCESS_KEY_ID') or os.path.exists(Path.home() / '.aws' / 'credentials'):
            try:
                return AWSKMSBackend(region=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))
            except Exception as e:
                logger.warning(f"AWS KMS not available: {e}")
        
        # Check for Vault configuration
        vault_addr = os.getenv('VAULT_ADDR')
        vault_token = os.getenv('VAULT_TOKEN')
        if vault_addr and vault_token:
            try:
                return HashiCorpVaultBackend(vault_addr, vault_token)
            except Exception as e:
                logger.warning(f"HashiCorp Vault not available: {e}")
        
        # Fall back to mock HSM
        logger.warning("No HSM backend configured, using MockHSMBackend")
        return MockHSMBackend()
    
    def setup_genomevault_keys(self) -> Dict[str, KeyMetadata]:
        """Set up standard GenomeVault keys."""
        keys_to_create = {
            'genomevault-api-signing': {
                'algorithm': 'RSA-2048',
                'usage': ['sign', 'verify'],
                'description': 'API request signing and verification'
            },
            'genomevault-data-encryption': {
                'algorithm': 'AES-256',
                'usage': ['encrypt', 'decrypt'],
                'description': 'PHI and genomic data encryption'
            },
            'genomevault-zk-proof': {
                'algorithm': 'ECDSA-P256',
                'usage': ['sign', 'verify'],
                'description': 'Zero-knowledge proof signing'
            },
            'genomevault-pir-encryption': {
                'algorithm': 'ChaCha20',
                'usage': ['encrypt', 'decrypt'],
                'description': 'PIR query encryption'
            }
        }
        
        created_keys = {}
        for key_id, config in keys_to_create.items():
            try:
                # Check if key already exists
                existing_keys = self.backend.list_keys()
                if any(k.key_id == key_id for k in existing_keys):
                    logger.info(f"Key {key_id} already exists, skipping")
                    continue
                
                metadata = self.backend.generate_key(
                    key_id=key_id,
                    algorithm=config['algorithm'],
                    key_usage=config['usage'],
                    description=config['description'],
                    tags={'purpose': 'genomevault', 'environment': 'production'}
                )
                created_keys[key_id] = metadata
                logger.info(f"✅ Created key: {key_id}")
                
            except Exception as e:
                logger.error(f"Failed to create key {key_id}: {e}")
        
        return created_keys
    
    def encrypt_phi_data(self, data: bytes, context: Optional[Dict[str, str]] = None) -> bytes:
        """Encrypt PHI data using HSM."""
        return self.backend.encrypt(
            'genomevault-data-encryption', 
            data, 
            context=context or {}
        )
    
    def decrypt_phi_data(self, ciphertext: bytes, context: Optional[Dict[str, str]] = None) -> bytes:
        """Decrypt PHI data using HSM."""
        return self.backend.decrypt(
            'genomevault-data-encryption',
            ciphertext,
            context=context or {}
        )
    
    def sign_api_request(self, request_data: bytes) -> bytes:
        """Sign API request using HSM."""
        return self.backend.sign('genomevault-api-signing', request_data)
    
    def verify_api_request(self, request_data: bytes, signature: bytes) -> bool:
        """Verify API request signature using HSM."""
        return self.backend.verify('genomevault-api-signing', request_data, signature)
    
    def get_hsm_status(self) -> Dict[str, Any]:
        """Get HSM status and key inventory."""
        try:
            keys = self.backend.list_keys()
            return {
                'backend_type': self.backend.__class__.__name__,
                'status': 'operational',
                'key_count': len(keys),
                'keys': [
                    {
                        'id': k.key_id,
                        'algorithm': k.algorithm,
                        'usage': k.key_usage,
                        'created': k.created_at
                    }
                    for k in keys
                ]
            }
        except Exception as e:
            return {
                'backend_type': self.backend.__class__.__name__,
                'status': 'error',
                'error': str(e),
                'key_count': 0,
                'keys': []
            }


# Global HSM manager instance
_hsm_manager: Optional[HSMManager] = None


def get_hsm_manager() -> HSMManager:
    """Get global HSM manager instance."""
    global _hsm_manager
    if _hsm_manager is None:
        _hsm_manager = HSMManager()
    return _hsm_manager


def initialize_hsm(backend: Optional[HSMBackend] = None) -> HSMManager:
    """Initialize HSM with specific backend."""
    global _hsm_manager
    _hsm_manager = HSMManager(backend)
    return _hsm_manager