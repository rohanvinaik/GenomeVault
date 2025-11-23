"""
HSM CLI commands for GenomeVault.

Provides command-line interface for HSM key management and operations.
"""

import click
import json
import sys
from typing import Optional

from genomevault.security.hsm_integration import (
    get_hsm_manager, 
    initialize_hsm,
    MockHSMBackend,
    AWSKMSBackend, 
    HashiCorpVaultBackend,
    HSMError
)
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


@click.group()
def hsm():
    """HSM (Hardware Security Module) management commands."""
    pass


@hsm.command()
@click.option('--backend', type=click.Choice(['auto', 'mock', 'aws-kms', 'vault']), 
              default='auto', help='HSM backend to use')
@click.option('--region', default='us-east-1', help='AWS region (for AWS KMS)')
@click.option('--vault-url', help='HashiCorp Vault URL')
@click.option('--vault-token', help='HashiCorp Vault token')
def status(backend: str, region: str, vault_url: Optional[str], vault_token: Optional[str]):
    """Show HSM status and key inventory."""
    try:
        # Initialize HSM with specified backend
        if backend == 'mock':
            hsm_backend = MockHSMBackend()
        elif backend == 'aws-kms':
            hsm_backend = AWSKMSBackend(region=region)
        elif backend == 'vault':
            if not vault_url or not vault_token:
                click.echo("❌ Vault URL and token required for Vault backend", err=True)
                sys.exit(1)
            hsm_backend = HashiCorpVaultBackend(vault_url, vault_token)
        else:
            hsm_backend = None  # Use auto-detection
        
        if hsm_backend:
            manager = initialize_hsm(hsm_backend)
        else:
            manager = get_hsm_manager()
        
        status_info = manager.get_hsm_status()
        
        click.echo("🔐 GenomeVault HSM Status")
        click.echo("=" * 40)
        click.echo(f"Backend: {status_info['backend_type']}")
        click.echo(f"Status: {'✅' if status_info['status'] == 'operational' else '❌'} {status_info['status']}")
        click.echo(f"Keys: {status_info['key_count']}")
        
        if status_info.get('error'):
            click.echo(f"Error: {status_info['error']}", err=True)
        
        if status_info['keys']:
            click.echo("\nKey Inventory:")
            click.echo("-" * 40)
            for key in status_info['keys']:
                usage = ', '.join(key['usage']) if isinstance(key['usage'], list) else key['usage']
                click.echo(f"  {key['id']}")
                click.echo(f"    Algorithm: {key['algorithm']}")
                click.echo(f"    Usage: {usage}")
                click.echo(f"    Created: {key['created']}")
                click.echo()
        
    except Exception as e:
        click.echo(f"❌ HSM status check failed: {e}", err=True)
        sys.exit(1)


@hsm.command()
@click.option('--backend', type=click.Choice(['mock', 'aws-kms', 'vault']),
              help='HSM backend to use')
@click.option('--region', default='us-east-1', help='AWS region (for AWS KMS)')
@click.option('--vault-url', help='HashiCorp Vault URL')
@click.option('--vault-token', help='HashiCorp Vault token')
@click.option('--force', is_flag=True, help='Force key generation even if keys exist')
def setup(backend: Optional[str], region: str, vault_url: Optional[str], 
          vault_token: Optional[str], force: bool):
    """Set up standard GenomeVault HSM keys."""
    try:
        # Initialize HSM backend
        if backend == 'mock':
            hsm_backend = MockHSMBackend()
        elif backend == 'aws-kms':
            hsm_backend = AWSKMSBackend(region=region)
        elif backend == 'vault':
            if not vault_url or not vault_token:
                click.echo("❌ Vault URL and token required for Vault backend", err=True)
                sys.exit(1)
            hsm_backend = HashiCorpVaultBackend(vault_url, vault_token)
        else:
            hsm_backend = None  # Use auto-detection
        
        if hsm_backend:
            manager = initialize_hsm(hsm_backend)
        else:
            manager = get_hsm_manager()
        
        click.echo("🔐 Setting up GenomeVault HSM keys...")
        click.echo("=" * 50)
        
        # Check existing keys if not forcing
        if not force:
            existing_keys = manager.backend.list_keys()
            genomevault_keys = [k for k in existing_keys if 'genomevault' in k.key_id.lower()]
            if genomevault_keys:
                click.echo(f"Found {len(genomevault_keys)} existing GenomeVault keys:")
                for key in genomevault_keys:
                    click.echo(f"  - {key.key_id}")
                click.echo()
                if not click.confirm("Continue with key setup?"):
                    click.echo("Setup cancelled.")
                    return
        
        created_keys = manager.setup_genomevault_keys()
        
        click.echo(f"\n✅ Key setup complete!")
        if created_keys:
            click.echo(f"Created {len(created_keys)} new keys:")
            for key_id, metadata in created_keys.items():
                click.echo(f"  ✅ {key_id} ({metadata.algorithm})")
        else:
            click.echo("No new keys created (all keys already exist)")
        
    except Exception as e:
        click.echo(f"❌ HSM setup failed: {e}", err=True)
        sys.exit(1)


@hsm.command()
@click.argument('key_id')
@click.argument('algorithm', type=click.Choice(['RSA-2048', 'AES-256', 'ECDSA-P256', 'ChaCha20']))
@click.argument('usage', type=click.Choice(['encrypt', 'decrypt', 'sign', 'verify']))
@click.option('--description', help='Key description')
@click.option('--backend', type=click.Choice(['mock', 'aws-kms', 'vault']),
              help='HSM backend to use')
def create_key(key_id: str, algorithm: str, usage: str, description: Optional[str], 
               backend: Optional[str]):
    """Create a new HSM key."""
    try:
        manager = get_hsm_manager()
        
        key_usage = usage.split(',') if ',' in usage else [usage]
        
        metadata = manager.backend.generate_key(
            key_id=key_id,
            algorithm=algorithm,
            key_usage=key_usage,
            description=description or f'Custom {algorithm} key for {usage}',
            tags={'created_by': 'cli', 'purpose': 'custom'}
        )
        
        click.echo(f"✅ Created key: {key_id}")
        click.echo(f"  Algorithm: {metadata.algorithm}")
        click.echo(f"  Usage: {', '.join(metadata.key_usage)}")
        click.echo(f"  Created: {metadata.created_at}")
        
    except HSMError as e:
        click.echo(f"❌ Key creation failed: {e}", err=True)
        sys.exit(1)


@hsm.command()
@click.argument('key_id')
@click.argument('data')
@click.option('--input-file', type=click.File('rb'), help='Read data from file')
@click.option('--output-file', type=click.File('wb'), help='Write encrypted data to file')
@click.option('--context', help='Encryption context (JSON)')
def encrypt(key_id: str, data: str, input_file, output_file, context: Optional[str]):
    """Encrypt data using HSM key."""
    try:
        manager = get_hsm_manager()
        
        # Get input data
        if input_file:
            plaintext = input_file.read()
        else:
            plaintext = data.encode()
        
        # Parse context
        encryption_context = {}
        if context:
            encryption_context = json.loads(context)
        
        # Encrypt
        ciphertext = manager.backend.encrypt(
            key_id=key_id,
            plaintext=plaintext,
            context=encryption_context
        )
        
        # Output result
        if output_file:
            output_file.write(ciphertext)
            click.echo(f"✅ Encrypted {len(plaintext)} bytes to {output_file.name}")
        else:
            import base64
            encoded = base64.b64encode(ciphertext).decode()
            click.echo(f"✅ Encrypted data: {encoded}")
        
    except Exception as e:
        click.echo(f"❌ Encryption failed: {e}", err=True)
        sys.exit(1)


@hsm.command()
@click.argument('key_id')  
@click.argument('ciphertext')
@click.option('--input-file', type=click.File('rb'), help='Read encrypted data from file')
@click.option('--output-file', type=click.File('wb'), help='Write decrypted data to file')
@click.option('--context', help='Encryption context (JSON)')
def decrypt(key_id: str, ciphertext: str, input_file, output_file, context: Optional[str]):
    """Decrypt data using HSM key."""
    try:
        manager = get_hsm_manager()
        
        # Get input data
        if input_file:
            encrypted_data = input_file.read()
        else:
            import base64
            encrypted_data = base64.b64decode(ciphertext)
        
        # Parse context
        encryption_context = {}
        if context:
            encryption_context = json.loads(context)
        
        # Decrypt
        plaintext = manager.backend.decrypt(
            key_id=key_id,
            ciphertext=encrypted_data,
            context=encryption_context
        )
        
        # Output result
        if output_file:
            output_file.write(plaintext)
            click.echo(f"✅ Decrypted {len(encrypted_data)} bytes to {output_file.name}")
        else:
            click.echo(f"✅ Decrypted data: {plaintext.decode()}")
        
    except Exception as e:
        click.echo(f"❌ Decryption failed: {e}", err=True)
        sys.exit(1)


@hsm.command()
@click.argument('key_id')
@click.option('--confirm', is_flag=True, help='Confirm deletion without prompt')
def delete_key(key_id: str, confirm: bool):
    """Delete HSM key."""
    if not confirm:
        click.confirm(f"Are you sure you want to delete key '{key_id}'?", abort=True)
    
    try:
        manager = get_hsm_manager()
        
        if manager.backend.delete_key(key_id):
            click.echo(f"✅ Deleted key: {key_id}")
        else:
            click.echo(f"❌ Key not found: {key_id}", err=True)
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"❌ Key deletion failed: {e}", err=True)
        sys.exit(1)


@hsm.command()
@click.option('--format', type=click.Choice(['json', 'table']), default='table',
              help='Output format')
def list_keys(format: str):
    """List all HSM keys."""
    try:
        manager = get_hsm_manager()
        keys = manager.backend.list_keys()
        
        if format == 'json':
            key_data = [
                {
                    'key_id': k.key_id,
                    'algorithm': k.algorithm,
                    'key_usage': k.key_usage,
                    'created_at': k.created_at,
                    'description': k.description,
                    'tags': k.tags
                }
                for k in keys
            ]
            click.echo(json.dumps(key_data, indent=2))
        else:
            if not keys:
                click.echo("No keys found.")
                return
            
            click.echo("🔐 HSM Keys")
            click.echo("=" * 80)
            for key in keys:
                usage = ', '.join(key.key_usage) if isinstance(key.key_usage, list) else key.key_usage
                click.echo(f"Key ID: {key.key_id}")
                click.echo(f"  Algorithm: {key.algorithm}")
                click.echo(f"  Usage: {usage}")
                click.echo(f"  Created: {key.created_at}")
                if key.description:
                    click.echo(f"  Description: {key.description}")
                click.echo()
        
    except Exception as e:
        click.echo(f"❌ Failed to list keys: {e}", err=True)
        sys.exit(1)


@hsm.command()
@click.option('--data-size', default=1024, help='Size of test data in bytes')
@click.option('--iterations', default=10, help='Number of test iterations')
def test_performance(data_size: int, iterations: int):
    """Test HSM performance."""
    import time
    import secrets
    
    try:
        manager = get_hsm_manager()
        
        # Ensure we have a test key
        try:
            test_data = {
                'key_id': 'genomevault-performance-test',
                'algorithm': 'AES-256',
                'usage': ['encrypt', 'decrypt'],
                'description': 'Performance test key'
            }
            
            existing_keys = manager.backend.list_keys()
            if not any(k.key_id == test_data['key_id'] for k in existing_keys):
                manager.backend.generate_key(**test_data)
        except Exception:
            pass  # Key might already exist
        
        click.echo(f"🚀 HSM Performance Test")
        click.echo("=" * 40)
        click.echo(f"Data size: {data_size} bytes")
        click.echo(f"Iterations: {iterations}")
        click.echo(f"Backend: {manager.backend.__class__.__name__}")
        click.echo()
        
        # Generate test data
        test_plaintext = secrets.token_bytes(data_size)
        
        # Test encryption performance
        encrypt_times = []
        for i in range(iterations):
            start = time.perf_counter()
            ciphertext = manager.backend.encrypt(test_data['key_id'], test_plaintext)
            encrypt_times.append(time.perf_counter() - start)
        
        # Test decryption performance
        decrypt_times = []
        for i in range(iterations):
            start = time.perf_counter()
            decrypted = manager.backend.decrypt(test_data['key_id'], ciphertext)
            decrypt_times.append(time.perf_counter() - start)
        
        # Calculate statistics
        avg_encrypt = sum(encrypt_times) / len(encrypt_times)
        avg_decrypt = sum(decrypt_times) / len(decrypt_times)
        total_time = sum(encrypt_times) + sum(decrypt_times)
        
        click.echo("Results:")
        click.echo(f"  Encryption: {avg_encrypt*1000:.2f}ms average")
        click.echo(f"  Decryption: {avg_decrypt*1000:.2f}ms average")
        click.echo(f"  Total time: {total_time:.2f}s")
        click.echo(f"  Throughput: {(data_size * iterations * 2) / total_time / 1024:.2f} KB/s")
        
    except Exception as e:
        click.echo(f"❌ Performance test failed: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    hsm()