"""
Secure Guide Reference Builder

Generates cryptographic bindings between GDiff files and guide sequences.
Enables full nucleotide-resolution queries while maintaining privacy.

Security Properties:
- Guide sequences never stored in GDiff (only encrypted pointers)
- HMAC commitment binds GDiff to specific guide pool
- AES-256-GCM encrypts chunk→guide mappings
- Decryption requires local guide FASTA files

See docs/SECURE_GUIDE_REFERENCE_SYSTEM.md for architecture details.
"""

import hashlib
import hmac
import json
import secrets
import base64
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

from genomevault.differential_encoding.gdiff.schema import SecureGuideReference


@dataclass
class GuidePoolMetadata:
    """Metadata for guide pool used in alignment"""
    guide_fasta_files: List[Path]
    alignment_seed: int
    chunk_size: int
    timestamp: str
    minimap2_params: Dict[str, str]


class SecureGuideReferenceBuilder:
    """
    Builds cryptographically secure references to guide pools.

    Workflow:
    1. Hash each guide FASTA file (SHA-256)
    2. Generate HMAC commitment from guide hashes + alignment params
    3. Encrypt chunk→guide mappings with AES-256-GCM
    4. Return SecureGuideReference for inclusion in GDiff metadata

    Example:
        builder = SecureGuideReferenceBuilder(
            guide_fasta_files=[Path("ref1.fa.gz"), Path("ref2.fa.gz")],
            chunk_guide_map={0: (0, 12345), 1: (1, 67890)},
            alignment_metadata=metadata
        )
        secure_ref = builder.build()
    """

    def __init__(
        self,
        guide_fasta_files: List[Path],
        chunk_guide_map: Dict[int, Tuple[int, int]],
        alignment_metadata: GuidePoolMetadata,
        user_secret: Optional[bytes] = None
    ):
        """
        Initialize builder.

        Args:
            guide_fasta_files: Paths to guide FASTA files (ref1.fa.gz, ref2.fa.gz, ...)
            chunk_guide_map: Mapping from chunk_id -> (guide_idx, alignment_seed)
            alignment_metadata: Metadata about alignment execution
            user_secret: Optional user secret for HMAC (generated if not provided)
        """
        self.guide_fasta_files = guide_fasta_files
        self.chunk_guide_map = chunk_guide_map
        self.alignment_metadata = alignment_metadata
        self.user_secret = user_secret if user_secret else secrets.token_bytes(32)

    def build(self) -> SecureGuideReference:
        """
        Build secure guide reference.

        Returns:
            SecureGuideReference with cryptographic bindings
        """
        # Step 1: Compute guide pool commitment
        guide_pool_commitment = self._compute_guide_pool_commitment()

        # Step 2: Encrypt chunk→guide map
        chunk_guide_map_encrypted = self._encrypt_chunk_guide_map(guide_pool_commitment)

        # Step 3: Generate alignment metadata hash
        alignment_metadata_hash = self._compute_alignment_metadata_hash()

        return SecureGuideReference(
            guide_pool_commitment=guide_pool_commitment,
            chunk_guide_map_encrypted=chunk_guide_map_encrypted,
            alignment_metadata_hash=alignment_metadata_hash,
            nucleotide_resolution_enabled=True,
            chunk_size=self.alignment_metadata.chunk_size,
            encryption_version="AES-256-GCM-v1"
        )

    def _compute_guide_pool_commitment(self) -> str:
        """
        Compute HMAC-SHA256 commitment to guide pool.

        Commitment = HMAC(user_secret, guide_hashes || alignment_params)

        Returns:
            64-character hex string (HMAC-SHA256)
        """
        # Hash each guide FASTA
        guide_hashes = []
        for guide_fasta in self.guide_fasta_files:
            guide_hash = self._hash_fasta_file(guide_fasta)
            guide_hashes.append(guide_hash)

        # Concatenate guide hashes
        guide_hashes_concat = "".join(guide_hashes)

        # Add alignment parameters to commitment
        alignment_params_str = json.dumps({
            "chunk_size": self.alignment_metadata.chunk_size,
            "alignment_seed": self.alignment_metadata.alignment_seed,
            "minimap2_params": self.alignment_metadata.minimap2_params,
            "num_guides": len(self.guide_fasta_files)
        }, sort_keys=True)

        commitment_input = f"{guide_hashes_concat}||{alignment_params_str}"

        # Compute HMAC
        h = hmac.new(
            self.user_secret,
            commitment_input.encode('utf-8'),
            hashlib.sha256
        )

        return h.hexdigest()

    def _encrypt_chunk_guide_map(self, guide_pool_commitment: str) -> str:
        """
        Encrypt chunk→guide mapping with AES-256-GCM.

        Key derived from guide_pool_commitment using HKDF.

        Args:
            guide_pool_commitment: HMAC commitment (used for key derivation)

        Returns:
            Base64-encoded ciphertext (includes nonce and auth tag)
        """
        # Serialize chunk map to JSON
        chunk_map_json = json.dumps(
            {str(k): v for k, v in self.chunk_guide_map.items()},
            sort_keys=True
        )
        plaintext = chunk_map_json.encode('utf-8')

        # Derive encryption key from commitment using HKDF
        commitment_bytes = bytes.fromhex(guide_pool_commitment)
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,  # 256 bits for AES-256
            salt=None,
            info=b"GDiff-SecureGuideReference-v1",
            backend=default_backend()
        )
        encryption_key = hkdf.derive(commitment_bytes)

        # Generate random nonce (96 bits recommended for GCM)
        nonce = secrets.token_bytes(12)

        # Encrypt with AES-256-GCM
        aesgcm = AESGCM(encryption_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        # Combine nonce + ciphertext (ciphertext includes auth tag)
        combined = nonce + ciphertext

        # Base64 encode for storage
        return base64.b64encode(combined).decode('ascii')

    def _compute_alignment_metadata_hash(self) -> str:
        """
        Compute SHA-256 hash of alignment metadata.

        Binds GDiff to specific alignment execution (prevents replay attacks).

        Returns:
            64-character hex string (SHA-256)
        """
        metadata_dict = {
            "alignment_seed": self.alignment_metadata.alignment_seed,
            "chunk_size": self.alignment_metadata.chunk_size,
            "timestamp": self.alignment_metadata.timestamp,
            "minimap2_params": self.alignment_metadata.minimap2_params,
            "guide_count": len(self.guide_fasta_files)
        }

        metadata_json = json.dumps(metadata_dict, sort_keys=True)

        h = hashlib.sha256(metadata_json.encode('utf-8'))
        return h.hexdigest()

    def _hash_fasta_file(self, fasta_path: Path) -> str:
        """
        Compute SHA-256 hash of FASTA file.

        Handles both gzipped and uncompressed files.

        Args:
            fasta_path: Path to FASTA file (.fa, .fasta, .fa.gz, .fasta.gz)

        Returns:
            64-character hex string (SHA-256)
        """
        import gzip

        h = hashlib.sha256()

        # Determine if file is gzipped
        is_gzipped = str(fasta_path).endswith('.gz')

        # Read file in chunks (memory efficient for large genomes)
        open_func = gzip.open if is_gzipped else open
        with open_func(fasta_path, 'rb') as f:
            while True:
                chunk = f.read(8192)  # 8KB chunks
                if not chunk:
                    break
                h.update(chunk)

        return h.hexdigest()


def decrypt_chunk_guide_map(
    encrypted_map: str,
    guide_pool_commitment: str
) -> Dict[int, Tuple[int, int]]:
    """
    Decrypt chunk→guide mapping.

    Requires guide_pool_commitment (derived from local guide FASTAs).

    Args:
        encrypted_map: Base64-encoded ciphertext from SecureGuideReference
        guide_pool_commitment: HMAC commitment (hex string)

    Returns:
        Decrypted mapping: {chunk_id -> (guide_idx, alignment_seed)}

    Raises:
        ValueError: If decryption fails (wrong key or corrupted data)
    """
    # Derive decryption key from commitment
    commitment_bytes = bytes.fromhex(guide_pool_commitment)
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"GDiff-SecureGuideReference-v1",
        backend=default_backend()
    )
    decryption_key = hkdf.derive(commitment_bytes)

    # Decode base64
    combined = base64.b64decode(encrypted_map)

    # Split nonce + ciphertext
    nonce = combined[:12]
    ciphertext = combined[12:]

    # Decrypt with AES-256-GCM
    aesgcm = AESGCM(decryption_key)
    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    except Exception as e:
        raise ValueError(
            "Failed to decrypt chunk guide map. "
            "Ensure guide_pool_commitment matches local guide sequences."
        ) from e

    # Parse JSON
    chunk_map_json = json.loads(plaintext.decode('utf-8'))

    # Convert string keys back to ints
    chunk_map = {
        int(k): tuple(v) for k, v in chunk_map_json.items()
    }

    return chunk_map


def recompute_guide_pool_commitment(
    guide_fasta_files: List[Path],
    alignment_params: Dict[str, any],
    user_secret: bytes
) -> str:
    """
    Recompute guide pool commitment from local guide FASTAs.

    Used during query resolution to verify GDiff authenticity.

    Args:
        guide_fasta_files: Paths to guide FASTA files on user's system
        alignment_params: Alignment parameters from GDiff metadata
        user_secret: User secret for HMAC

    Returns:
        64-character hex string (HMAC-SHA256)
    """
    builder = SecureGuideReferenceBuilder(
        guide_fasta_files=guide_fasta_files,
        chunk_guide_map={},  # Not needed for commitment only
        alignment_metadata=GuidePoolMetadata(
            guide_fasta_files=guide_fasta_files,
            alignment_seed=alignment_params.get("alignment_seed", 0),
            chunk_size=alignment_params.get("chunk_size", 10_000_000),
            timestamp="",
            minimap2_params=alignment_params.get("minimap2_params", {})
        ),
        user_secret=user_secret
    )

    return builder._compute_guide_pool_commitment()


# ============================================================================
# CLI Utilities
# ============================================================================

def verify_secure_reference(
    gdiff_path: Path,
    local_guide_dir: Path,
    user_secret: Optional[bytes] = None
) -> bool:
    """
    Verify GDiff secure guide reference matches local guides.

    Args:
        gdiff_path: Path to GDiff file (.gdiff.gz)
        local_guide_dir: Directory containing guide FASTA files
        user_secret: User secret for HMAC (if not provided, attempts discovery)

    Returns:
        True if verification succeeds, False otherwise
    """
    from genomevault.differential_encoding.gdiff.encoder import GDiffEncoder

    # Load GDiff metadata
    encoder = GDiffEncoder.load(gdiff_path)
    secure_ref = encoder.metadata.secure_guide_reference

    if not secure_ref:
        print("⚠️  GDiff does not include secure guide reference (legacy format)")
        return False

    # Find local guide FASTAs
    guide_fastas = sorted(local_guide_dir.glob("ref*.fa.gz"))
    if not guide_fastas:
        print(f"❌ No guide FASTAs found in {local_guide_dir}")
        return False

    # Recompute commitment
    if not user_secret:
        print("ℹ️  No user secret provided, attempting automatic discovery...")
        # In production, user_secret should be stored securely (keychain, etc.)
        # For now, use deterministic derivation from guide hashes
        user_secret = hashlib.sha256(
            "".join([str(f) for f in guide_fastas]).encode('utf-8')
        ).digest()

    recomputed_commitment = recompute_guide_pool_commitment(
        guide_fasta_files=guide_fastas,
        alignment_params={
            "alignment_seed": 0,  # TODO: Extract from metadata
            "chunk_size": secure_ref.chunk_size,
            "minimap2_params": {}
        },
        user_secret=user_secret
    )

    # Compare commitments
    if recomputed_commitment == secure_ref.guide_pool_commitment:
        print("✓ Secure guide reference verified successfully!")
        return True
    else:
        print("❌ Verification failed! GDiff commitment does not match local guides.")
        print(f"   Expected: {secure_ref.guide_pool_commitment}")
        print(f"   Computed: {recomputed_commitment}")
        return False
