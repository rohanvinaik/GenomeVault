"""
HDV Cache Manager

Manages caching and retrieval of hyperdimensional vector (HDV) encodings
to avoid redundant computation.

Storage Structure:
    data/hdv_cache/
    ├── {query_id}/
    │   ├── gdiff/
    │   │   └── differential.gdiff.gz     # Comprehensive GDiff (stays local)
    │   │   └── differential.gdiff.gz.enc # Encrypted (if encryption enabled)
    │   ├── k3/
    │   │   ├── simple_snp_lookup.hdc
    │   │   ├── clinical_risk.hdc
    │   │   └── pharmacogenomics.hdc
    │   ├── k7/
    │   │   ├── clinical_risk.hdc
    │   │   └── full_research_profile.hdc
    │   └── metadata.json
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
import time

@dataclass
class HDVCacheMetadata:
    """Metadata for cached HDV encodings"""
    query_id: str
    k_anonymity: int
    schema_name: str
    created_timestamp: float
    gdiff_path: str
    hdv_path: str
    encoding_time_ms: float
    dimension: int
    hdv_size_bytes: int
    num_variants: int


class HDVCacheManager:
    """
    Manages caching and retrieval of HDV encodings.

    This prevents redundant encoding by:
    1. Checking if HDV already exists for (query_id, k, schema)
    2. Storing new HDVs in organized directory structure
    3. Providing fast lookup by query parameters
    4. Optional AES-256-GCM encryption for GDiff files

    Example:
        >>> cache = HDVCacheManager()
        >>>
        >>> # Check if HDV exists
        >>> cached = cache.get_hdv("patient_123", k_anonymity=3, schema="clinical_risk")
        >>>
        >>> if cached is None:
        >>>     # Generate new HDV
        >>>     hdv = encoder.encode(gdiff, CLINICAL_RISK)
        >>>     cache.store_hdv("patient_123", k_anonymity=3, schema="clinical_risk", hdv=hdv)

        >>> # With encryption
        >>> cache = HDVCacheManager(enable_encryption=True, encryption_password="secure_password")
    """

    def __init__(
        self,
        cache_root: Optional[Path] = None,
        enable_encryption: bool = False,
        encryption_key: Optional[bytes] = None,
        encryption_password: Optional[str] = None
    ):
        """
        Initialize HDV cache manager.

        Args:
            cache_root: Root directory for cache (default: data/hdv_cache/)
            enable_encryption: Enable AES-256-GCM encryption for GDiff files
            encryption_key: 32-byte encryption key (or None to auto-generate)
            encryption_password: Password for key derivation (alternative to encryption_key)
        """
        if cache_root is None:
            cache_root = Path("data/hdv_cache")

        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # Encryption setup
        self.enable_encryption = enable_encryption
        self.secure_storage = None

        if self.enable_encryption:
            # Import here to avoid circular dependency
            from genomevault.storage.gdiff_storage import SecureGDiffStorage

            # Initialize encryption key
            if encryption_password:
                # Derive key from password
                derived_key, salt = SecureGDiffStorage.derive_key_from_password(encryption_password)

                # Save salt for future use (in metadata)
                salt_file = self.cache_root / ".encryption_salt"
                with open(salt_file, 'wb') as f:
                    f.write(salt)
                os.chmod(salt_file, 0o600)

                encryption_key = derived_key

            # Create secure storage instance
            storage_dir = str(self.cache_root / "secure_gdiff")
            self.secure_storage = SecureGDiffStorage(
                storage_dir=storage_dir,
                user_key=encryption_key,
                audit_log_path=str(self.cache_root / "audit.log")
            )

    def get_query_dir(self, query_id: str) -> Path:
        """Get directory for a specific query ID"""
        query_dir = self.cache_root / query_id
        query_dir.mkdir(parents=True, exist_ok=True)
        return query_dir

    def get_k_dir(self, query_id: str, k_anonymity: int) -> Path:
        """Get directory for specific k-anonymity level"""
        k_dir = self.get_query_dir(query_id) / f"k{k_anonymity}"
        k_dir.mkdir(parents=True, exist_ok=True)
        return k_dir

    def get_gdiff_dir(self, query_id: str) -> Path:
        """Get directory for storing GDiff files"""
        gdiff_dir = self.get_query_dir(query_id) / "gdiff"
        gdiff_dir.mkdir(parents=True, exist_ok=True)
        return gdiff_dir

    def get_hdv_path(self, query_id: str, k_anonymity: int, schema_name: str) -> Path:
        """Get path for specific HDV encoding"""
        k_dir = self.get_k_dir(query_id, k_anonymity)
        return k_dir / f"{schema_name}.hdc"

    def get_gdiff_path(self, query_id: str) -> Path:
        """Get path for GDiff file"""
        gdiff_dir = self.get_gdiff_dir(query_id)
        return gdiff_dir / "differential.gdiff.gz"

    def get_metadata_path(self, query_id: str) -> Path:
        """Get path for metadata JSON"""
        return self.get_query_dir(query_id) / "metadata.json"

    def hdv_exists(self, query_id: str, k_anonymity: int, schema_name: str) -> bool:
        """Check if HDV encoding already exists"""
        hdv_path = self.get_hdv_path(query_id, k_anonymity, schema_name)
        return hdv_path.exists()

    def gdiff_exists(self, query_id: str) -> bool:
        """Check if GDiff file already exists"""
        gdiff_path = self.get_gdiff_path(query_id)
        return gdiff_path.exists()

    def get_hdv(self, query_id: str, k_anonymity: int, schema_name: str) -> Optional[Path]:
        """
        Get cached HDV if it exists.

        Returns:
            Path to HDV file if cached, None otherwise
        """
        hdv_path = self.get_hdv_path(query_id, k_anonymity, schema_name)

        if hdv_path.exists():
            return hdv_path

        return None

    def store_hdv(
        self,
        query_id: str,
        k_anonymity: int,
        schema_name: str,
        hdv_encoding,  # HDVEncoding from selective_hdv_encoder
        gdiff_path: Optional[Path] = None
    ) -> Path:
        """
        Store HDV encoding in cache.

        Args:
            query_id: Unique identifier for query
            k_anonymity: k-anonymity level used
            schema_name: Analysis schema used
            hdv_encoding: HDVEncoding object from encoder
            gdiff_path: Optional path to GDiff file

        Returns:
            Path to stored HDV file
        """
        # Save HDV to cache
        hdv_path = self.get_hdv_path(query_id, k_anonymity, schema_name)
        hdv_encoding.save(hdv_path)

        # Update metadata
        metadata = self._load_metadata(query_id)

        cache_entry = {
            "k_anonymity": k_anonymity,
            "schema_name": schema_name,
            "created_timestamp": time.time(),
            "gdiff_path": str(gdiff_path) if gdiff_path else str(self.get_gdiff_path(query_id)),
            "hdv_path": str(hdv_path),
            "encoding_time_ms": hdv_encoding.encoding_time_ms,
            "dimension": hdv_encoding.dimension,
            "hdv_size_bytes": hdv_encoding.hdv_size_bytes,
            "num_variants": hdv_encoding.num_variants_encoded,
        }

        # Add to metadata
        if "hdv_encodings" not in metadata:
            metadata["hdv_encodings"] = []

        metadata["hdv_encodings"].append(cache_entry)
        metadata["query_id"] = query_id
        metadata["last_updated"] = time.time()

        # Save metadata
        self._save_metadata(query_id, metadata)

        return hdv_path

    def store_gdiff(self, query_id: str, gdiff_document) -> Path:
        """
        Store GDiff document in cache (with optional encryption).

        Args:
            query_id: Unique identifier for query
            gdiff_document: GDiffDocument to store

        Returns:
            Path to stored GDiff file (encrypted if encryption enabled)
        """
        if self.enable_encryption and self.secure_storage:
            # Store encrypted GDiff using SecureGDiffStorage
            from genomevault.storage.gdiff_storage import GDiffStorageMetadata

            # Convert GDiffDocument to dict
            gdiff_dict = gdiff_document.to_dict() if hasattr(gdiff_document, 'to_dict') else gdiff_document

            # Create metadata for secure storage
            storage_metadata = GDiffStorageMetadata(
                gdiff_id=query_id,
                query_vcf_path="unknown",  # Set by caller if available
                reference_pool_paths=[],   # Set by caller if available
                k_anonymity=gdiff_document.metadata.k_anonymity,
                created_timestamp=time.time(),
                num_variants=len(gdiff_document.differential_variants),
                compressed_size_bytes=0,  # Will be calculated by secure_storage
                encrypted_size_bytes=0,   # Will be calculated by secure_storage
                encryption_algorithm="AES-256-GCM",
                kdf_algorithm="PBKDF2-HMAC-SHA256",
                kdf_iterations=480000,
                file_permissions="0600",
                checksum_sha256="",  # Will be calculated by secure_storage
            )

            # Save encrypted
            filename = f"{query_id}.gdiff"
            gdiff_path = self.secure_storage.save_gdiff(gdiff_dict, filename, storage_metadata)
        else:
            # Store unencrypted GDiff
            gdiff_path = self.get_gdiff_path(query_id)
            gdiff_document.save(gdiff_path)

        # Update metadata
        metadata = self._load_metadata(query_id)
        metadata["query_id"] = query_id
        metadata["gdiff_path"] = str(gdiff_path)
        metadata["num_variants"] = len(gdiff_document.differential_variants)
        metadata["k_anonymity"] = gdiff_document.metadata.k_anonymity
        metadata["created_timestamp"] = time.time()
        metadata["encrypted"] = self.enable_encryption

        self._save_metadata(query_id, metadata)

        return gdiff_path

    def load_gdiff(self, query_id: str):
        """
        Load GDiff document from cache (with automatic decryption if needed).

        Args:
            query_id: Unique identifier for query

        Returns:
            GDiffDocument or dict with GDiff data

        Raises:
            FileNotFoundError: If GDiff file doesn't exist
            ValueError: If decryption fails
        """
        from genomevault.differential_encoding.gdiff.schema import GDiffDocument

        metadata = self._load_metadata(query_id)
        is_encrypted = metadata.get("encrypted", False)

        if is_encrypted and self.secure_storage:
            # Load encrypted GDiff
            try:
                filename = f"{query_id}.gdiff"
                gdiff_dict = self.secure_storage.load_gdiff(filename)

                # Convert dict back to GDiffDocument
                gdiff_doc = GDiffDocument.from_dict(gdiff_dict)
                return gdiff_doc
            except Exception as e:
                raise ValueError(f"Failed to decrypt GDiff: {e}")
        else:
            # Load unencrypted GDiff
            gdiff_path = self.get_gdiff_path(query_id)

            if not gdiff_path.exists():
                raise FileNotFoundError(f"GDiff file not found: {gdiff_path}")

            return GDiffDocument.load(gdiff_path)

    def list_available_hdvs(self, query_id: str) -> List[Dict]:
        """
        List all available HDVs for a query.

        Returns:
            List of dicts with k_anonymity, schema_name, hdv_path
        """
        metadata = self._load_metadata(query_id)
        return metadata.get("hdv_encodings", [])

    def list_available_schemas(
        self,
        query_id: str,
        k_anonymity: int
    ) -> List[str]:
        """
        List available schemas for specific k-anonymity level.

        Returns:
            List of schema names
        """
        encodings = self.list_available_hdvs(query_id)
        return [
            enc["schema_name"]
            for enc in encodings
            if enc["k_anonymity"] == k_anonymity
        ]

    def get_cache_stats(self, query_id: str) -> Dict:
        """Get cache statistics for a query"""
        metadata = self._load_metadata(query_id)

        encodings = metadata.get("hdv_encodings", [])

        stats = {
            "query_id": query_id,
            "num_encodings": len(encodings),
            "total_hdv_size_bytes": sum(enc["hdv_size_bytes"] for enc in encodings),
            "k_levels_available": sorted(set(enc["k_anonymity"] for enc in encodings)),
            "schemas_available": sorted(set(enc["schema_name"] for enc in encodings)),
            "gdiff_exists": self.gdiff_exists(query_id),
        }

        return stats

    def _load_metadata(self, query_id: str) -> Dict:
        """Load metadata for query"""
        metadata_path = self.get_metadata_path(query_id)

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)

        return {}

    def _save_metadata(self, query_id: str, metadata: Dict):
        """Save metadata for query"""
        metadata_path = self.get_metadata_path(query_id)

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def generate_query_id(self, vcf_path: str, reference_pool: List[str]) -> str:
        """
        Generate unique query ID based on VCF and reference pool.

        Args:
            vcf_path: Path to VCF file
            reference_pool: List of reference genome IDs

        Returns:
            Unique query ID (SHA-256 hash)
        """
        # Create deterministic ID from VCF path + reference pool
        content = f"{vcf_path}:{':'.join(sorted(reference_pool))}"
        query_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        return query_id

    def clear_query_cache(self, query_id: str):
        """Remove all cached data for a query"""
        query_dir = self.get_query_dir(query_id)

        if query_dir.exists():
            import shutil
            shutil.rmtree(query_dir)
