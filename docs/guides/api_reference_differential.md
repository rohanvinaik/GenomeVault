# Differential Encoding API Reference

**Version**: 1.0.0
**Last Updated**: 2025-01-19
**Module**: `genomevault.differential_encoding`

## Table of Contents

1. [Overview](#overview)
2. [Core Classes](#core-classes)
3. [Data Models](#data-models)
4. [Enums and Constants](#enums-and-constants)
5. [Functions](#functions)
6. [Usage Examples](#usage-examples)

---

## Overview

The differential encoding API provides cryptographically secure genomic data compression through hyperdimensional computing. This reference documents all public classes, methods, and functions.

### Quick Import

```python
from genomevault.differential_encoding import (
    # Main classes
    DifferentialGenomicEncoder,
    DifferentialGenomeQuery,
    SecureReferenceGenomeManager,

    # Data models
    Genome,
    Variant,
    ReferenceGenome,
    EncodedGenome,
    DifferentialEncodingMetadata,

    # Enums
    AnalysisType,
    DifferenceType,
    FunctionalImpact,

    # Functions
    compute_variant_differences,
    compute_reference_hash,
    setup_default_references,
)
```

---

## Core Classes

### DifferentialGenomicEncoder

Main encoder class for differential genomic encoding.

```python
class DifferentialGenomicEncoder:
    """
    Encode experimental genomes as cryptographically verified differences
    from reference genomes using hyperdimensional computing.
    """
```

#### Constructor

```python
def __init__(
    self,
    reference_manager: SecureReferenceGenomeManager,
    hypervector_encoder: DifferentialHypervectorEncoder,
    master_seed: Optional[bytes] = None,
)
```

**Parameters**:
- `reference_manager` (SecureReferenceGenomeManager): Manager for reference genome pool
- `hypervector_encoder` (DifferentialHypervectorEncoder): Encoder for feature→hypervector conversion
- `master_seed` (Optional[bytes]): Master seed for deterministic encoding (generates if None)

**Example**:
```python
from genomevault.differential_encoding import (
    SecureReferenceGenomeManager,
    DifferentialHypervectorEncoder,
    DifferentialGenomicEncoder,
)
from pathlib import Path

# Create components
manager = SecureReferenceGenomeManager(Path("references/"))
hv_encoder = DifferentialHypervectorEncoder(dimension=10000, seed=42)

# Create encoder
encoder = DifferentialGenomicEncoder(
    reference_manager=manager,
    hypervector_encoder=hv_encoder,
    master_seed=b"my_secure_seed_32_bytes_long!!!",
)
```

#### Methods

##### encode_genome()

Encode a complete genome using differential encoding.

```python
def encode_genome(
    self,
    genome: Genome,
    analysis_type: AnalysisType,
    bundle_chunks: bool = True,
) -> EncodedGenome
```

**Parameters**:
- `genome` (Genome): Experimental genome to encode
- `analysis_type` (AnalysisType): Chunking strategy (SLIDING_WINDOW, GENE_REGION, etc.)
- `bundle_chunks` (bool): Create bundled genome-level hypervector (default: True)

**Returns**:
- `EncodedGenome`: Encoded genome with hypervectors and metadata

**Raises**:
- `ValueError`: If no references available for genome assembly
- `RuntimeError`: If encoding fails

**Example**:
```python
from genomevault.differential_encoding import Genome, Variant, AnalysisType

genome = Genome(
    genome_id="patient_001",
    assembly="GRCh38",
    chromosomes={
        "chr1": [
            Variant(chromosome="chr1", position=100000, ref="A", alt="G"),
        ]
    }
)

encoded = encoder.encode_genome(
    genome=genome,
    analysis_type=AnalysisType.SLIDING_WINDOW,
    bundle_chunks=True,
)
```

##### encode_chunk()

Encode a single genomic chunk.

```python
def encode_chunk(
    self,
    chunk: GenomeChunk,
    reference_genome: ReferenceGenome,
) -> Tuple[bytes, bytes, DifferentialEncodingMetadata]
```

**Parameters**:
- `chunk` (GenomeChunk): Chunk to encode
- `reference_genome` (ReferenceGenome): Selected reference for this chunk

**Returns**:
- Tuple of (chunk_hypervector, chunk_id, metadata)

**Example**:
```python
chunk = GenomeChunk(
    chunk_id=b"chunk_001",
    chromosome="chr1",
    start_position=100000,
    end_position=200000,
    variants=[...],
)

reference = encoder.reference_manager.get_random_reference(seed)
hv, chunk_id, metadata = encoder.encode_chunk(chunk, reference)
```

---

### DifferentialGenomeQuery

Query interface for encoded genomes.

```python
class DifferentialGenomeQuery:
    """
    Query encoded genomes for variants in specific genomic regions.
    """
```

#### Constructor

```python
def __init__(
    self,
    reference_manager: SecureReferenceGenomeManager,
    hv_encoder: DifferentialHypervectorEncoder,
)
```

**Parameters**:
- `reference_manager` (SecureReferenceGenomeManager): Reference genome manager
- `hv_encoder` (DifferentialHypervectorEncoder): Hypervector encoder

**Example**:
```python
from genomevault.differential_encoding import DifferentialGenomeQuery

query = DifferentialGenomeQuery(
    reference_manager=encoder.reference_manager,
    hv_encoder=encoder.hypervector_encoder,
)
```

#### Methods

##### query_region()

Query a specific genomic region.

```python
def query_region(
    self,
    encoded_genome: EncodedGenome,
    chromosome: str,
    start: int,
    end: int,
) -> QueryResult
```

**Parameters**:
- `encoded_genome` (EncodedGenome): Genome to query
- `chromosome` (str): Chromosome name (e.g., "chr1")
- `start` (int): Start position (inclusive, 0-based)
- `end` (int): End position (exclusive, 0-based)

**Returns**:
- `QueryResult`: Query results with variants, chunks used, and timing

**Example**:
```python
result = query.query_region(
    encoded_genome=encoded,
    chromosome="chr1",
    start_position=100000,
    end_position=200000,
)

print(f"Found {result.variant_count} variants")
print(f"Used {result.chunks_used} chunks")
print(f"Query time: {result.query_time_ms:.2f} ms")
```

##### find_similar_genomes()

Find genomes similar to a query genome.

```python
def find_similar_genomes(
    self,
    query_genome: EncodedGenome,
    database: List[EncodedGenome],
    top_k: int = 10,
) -> List[SimilarityMatch]
```

**Parameters**:
- `query_genome` (EncodedGenome): Query genome
- `database` (List[EncodedGenome]): Database of encoded genomes
- `top_k` (int): Number of top matches to return (default: 10)

**Returns**:
- `List[SimilarityMatch]`: Top k similar genomes with similarity scores

**Example**:
```python
matches = query.find_similar_genomes(
    query_genome=patient_genome,
    database=all_genomes,
    top_k=5,
)

for match in matches:
    print(f"{match.genome_id}: similarity={match.similarity:.4f}")
```

---

### SecureReferenceGenomeManager

Manager for reference genome pools.

```python
class SecureReferenceGenomeManager:
    """
    Manage a pool of reference genomes with cryptographic verification.
    """
```

#### Constructor

```python
def __init__(
    self,
    reference_dir: Optional[Path] = None,
)
```

**Parameters**:
- `reference_dir` (Optional[Path]): Directory containing reference genomes (uses default if None)

**Example**:
```python
from pathlib import Path
from genomevault.differential_encoding import SecureReferenceGenomeManager

manager = SecureReferenceGenomeManager(
    reference_dir=Path("/data/references")
)

print(f"Loaded {manager.reference_count} references")
```

#### Properties

##### reference_count

```python
@property
def reference_count(self) -> int
```

Number of references in the pool.

**Example**:
```python
count = manager.reference_count
```

#### Methods

##### get_random_reference()

Get a random reference genome using cryptographic seed.

```python
def get_random_reference(
    self,
    seed: bytes,
    assembly: Optional[str] = None,
) -> ReferenceGenome
```

**Parameters**:
- `seed` (bytes): Cryptographic seed for deterministic selection
- `assembly` (Optional[str]): Filter by assembly (e.g., "GRCh38")

**Returns**:
- `ReferenceGenome`: Randomly selected reference

**Raises**:
- `ValueError`: If no references available for assembly

**Example**:
```python
import secrets

seed = secrets.token_bytes(32)
reference = manager.get_random_reference(seed, assembly="GRCh38")
```

---

### DifferentialHypervectorEncoder

Hypervector encoder for differential encoding.

```python
class DifferentialHypervectorEncoder:
    """
    Encode variant differences into hyperdimensional space.
    """
```

#### Constructor

```python
def __init__(
    self,
    dimension: int = 10000,
    seed: int = 42,
)
```

**Parameters**:
- `dimension` (int): Hypervector dimension (default: 10000)
- `seed` (int): Random seed for reproducibility (default: 42)

**Example**:
```python
from genomevault.differential_encoding import DifferentialHypervectorEncoder

encoder = DifferentialHypervectorEncoder(
    dimension=10000,
    seed=42,
)
```

#### Methods

##### encode_difference_vector()

Encode variant differences to hypervector.

```python
def encode_difference_vector(
    self,
    differences: List[VariantDifference],
    metadata: Optional[DifferentialEncodingMetadata] = None,
) -> np.ndarray
```

**Parameters**:
- `differences` (List[VariantDifference]): List of variant differences
- `metadata` (Optional[DifferentialEncodingMetadata]): Metadata for context

**Returns**:
- `np.ndarray`: Unit-normalized hypervector (shape: [dimension])

**Example**:
```python
from genomevault.differential_encoding import compute_variant_differences

differences = compute_variant_differences(exp_section, ref_section)
hypervector = encoder.encode_difference_vector(differences)
```

##### similarity()

Compute cosine similarity between hypervectors.

```python
def similarity(
    self,
    hv1: np.ndarray,
    hv2: np.ndarray,
) -> float
```

**Parameters**:
- `hv1` (np.ndarray): First hypervector
- `hv2` (np.ndarray): Second hypervector

**Returns**:
- `float`: Cosine similarity in range [-1, 1]

**Example**:
```python
similarity = encoder.similarity(hv1, hv2)
print(f"Similarity: {similarity:.4f}")
```

---

## Data Models

### Genome

Represents a complete genome with variants.

```python
@dataclass
class Genome:
    genome_id: str
    assembly: str
    chromosomes: Dict[str, List[Variant]]
```

**Fields**:
- `genome_id` (str): Unique genome identifier
- `assembly` (str): Reference assembly (e.g., "GRCh38")
- `chromosomes` (Dict[str, List[Variant]]): Variants by chromosome

**Example**:
```python
from genomevault.differential_encoding import Genome, Variant

genome = Genome(
    genome_id="patient_001",
    assembly="GRCh38",
    chromosomes={
        "chr1": [
            Variant(chromosome="chr1", position=100000, ref="A", alt="G"),
            Variant(chromosome="chr1", position=200000, ref="C", alt="T"),
        ],
        "chr2": [
            Variant(chromosome="chr2", position=150000, ref="G", alt="A"),
        ],
    }
)
```

#### Methods

##### get_chromosome_section()

Get variants in a specific region.

```python
def get_chromosome_section(
    self,
    chromosome: str,
    start_position: Optional[int] = None,
    end_position: Optional[int] = None,
) -> GenomeSection
```

**Parameters**:
- `chromosome` (str): Chromosome name
- `start_position` (Optional[int]): Start position (0-based, inclusive)
- `end_position` (Optional[int]): End position (0-based, exclusive)

**Returns**:
- `GenomeSection`: Section with filtered variants

**Example**:
```python
section = genome.get_chromosome_section("chr1", 100000, 200000)
print(f"Variants in region: {len(section.variants)}")
```

---

### Variant

Represents a single genomic variant.

```python
@dataclass
class Variant:
    chromosome: str
    position: int
    ref: str
    alt: str
    genotype: Optional[str] = None
    quality: Optional[float] = None
    info: Optional[Dict[str, Any]] = None
```

**Fields**:
- `chromosome` (str): Chromosome name (e.g., "chr1")
- `position` (int): Position (0-based)
- `ref` (str): Reference allele
- `alt` (str): Alternate allele
- `genotype` (Optional[str]): Genotype (e.g., "0/1", "1/1")
- `quality` (Optional[float]): Variant quality score
- `info` (Optional[Dict]): Additional annotations

**Example**:
```python
from genomevault.differential_encoding import Variant

variant = Variant(
    chromosome="chr1",
    position=100000,
    ref="A",
    alt="G",
    genotype="0/1",
    quality=99.0,
    info={"IMPACT": "HIGH", "Consequence": "missense_variant"},
)
```

---

### EncodedGenome

Represents an encoded genome with hypervectors and metadata.

```python
@dataclass
class EncodedGenome:
    genome_id: str
    assembly: str
    chunk_hypervectors: List[np.ndarray]
    metadata: List[DifferentialEncodingMetadata]
    bundled_hypervector: np.ndarray
    statistics: Dict[str, Any]
    master_seed: bytes
    encoding_hash: str
    created_at: datetime
    version: str
```

**Fields**:
- `genome_id` (str): Genome identifier
- `assembly` (str): Reference assembly
- `chunk_hypervectors` (List[np.ndarray]): List of chunk hypervectors
- `metadata` (List[DifferentialEncodingMetadata]): List of metadata for each chunk
- `bundled_hypervector` (np.ndarray): Genome-level bundled hypervector
- `statistics` (Dict[str, Any]): Encoding statistics
- `master_seed` (bytes): Master seed for deterministic encoding
- `encoding_hash` (str): Hash of encoding for verification
- `created_at` (datetime): Timestamp when encoding was created
- `version` (str): Encoding version

#### Methods

##### save()

Save encoded genome to disk.

```python
def save(
    self,
    path: Path,
    compress: bool = True,
) -> int
```

**Parameters**:
- `path` (Path): Save path
- `compress` (bool): Use gzip compression (default: True)

**Returns**:
- `int`: Bytes written

**Example**:
```python
from pathlib import Path

path = Path("encoded_genomes/patient_001.enc.gz")
bytes_written = encoded.save(path, compress=True)
print(f"Saved {bytes_written / 1024:.2f} KB")
```

##### load()

Load encoded genome from disk (class method).

```python
@classmethod
def load(cls, path: Path) -> 'EncodedGenome'
```

**Parameters**:
- `path` (Path): Load path

**Returns**:
- `EncodedGenome`: Loaded genome

**Example**:
```python
from genomevault.differential_encoding import EncodedGenome

loaded = EncodedGenome.load(Path("patient_001.enc.gz"))
```

##### verify()

Verify cryptographic integrity.

```python
def verify(self) -> bool
```

**Returns**:
- `bool`: True if all chunks pass verification

**Example**:
```python
if encoded.verify():
    print("✅ Verification passed")
else:
    print("❌ Verification failed")
```

##### storage_size_kb()

Calculate storage size in KB.

```python
def storage_size_kb(self) -> float
```

**Returns**:
- `float`: Storage size in kilobytes

**Example**:
```python
size_kb = encoded.storage_size_kb()
print(f"Storage: {size_kb:.2f} KB")
```

---

### DifferentialEncodingMetadata

Metadata for a single encoded chunk.

```python
@dataclass
class DifferentialEncodingMetadata:
    chunk_id: bytes
    chromosome: str
    start_position: int
    end_position: int
    reference_genome_id: str
    reference_seed: bytes
    reference_hash: bytes
    chunking_strategy: str
    chunking_seed: bytes
    analysis_type: AnalysisType
    difference_counts: Dict[str, int]
    binding_hmac: bytes
    timestamp: str
```

**Key Fields**:
- `chunk_id` (bytes): Unique chunk identifier
- `chromosome`, `start_position`, `end_position`: Genomic region
- `reference_genome_id`, `reference_hash`: Reference genome info
- `difference_counts`: Counts of new mutations, missing variants, genotype differences
- `binding_hmac`: Cryptographic binding

#### Methods

##### get_region_string()

Get human-readable region string.

```python
def get_region_string(self) -> str
```

**Returns**:
- `str`: Region string (e.g., "chr1:100000-200000")

**Example**:
```python
region = metadata.get_region_string()  # "chr1:100000-200000"
```

##### to_json()

Serialize to JSON string.

```python
def to_json(self) -> str
```

**Returns**:
- `str`: JSON-serialized metadata

**Example**:
```python
json_str = metadata.to_json()
```

##### from_json()

Deserialize from JSON (class method).

```python
@classmethod
def from_json(cls, json_str: str) -> 'DifferentialEncodingMetadata'
```

**Parameters**:
- `json_str` (str): JSON string

**Returns**:
- `DifferentialEncodingMetadata`: Deserialized metadata

**Example**:
```python
metadata = DifferentialEncodingMetadata.from_json(json_str)
```

---

## Enums and Constants

### AnalysisType

Chunking strategies for different use cases.

```python
class AnalysisType(Enum):
    SLIDING_WINDOW = "sliding_window"
    GENE_REGION = "gene_region"
    VARIANT_DENSITY = "variant_density"
    FUNCTIONAL_REGIONS = "functional_regions"
    CHROMOSOMAL = "chromosomal"
    CUSTOM_INTERVALS = "custom_intervals"
    POPULATION_STRATIFIED = "population_stratified"
```

**Example**:
```python
from genomevault.differential_encoding import AnalysisType

# Use for encoding
encoded = encoder.encode_genome(
    genome=genome,
    analysis_type=AnalysisType.GENE_REGION,
)
```

---

### DifferenceType

Types of variant differences.

```python
class DifferenceType(Enum):
    NEW_MUTATION = "new_mutation"
    MISSING_VARIANT = "missing_variant"
    GENOTYPE_DIFFERENCE = "genotype_difference"
```

**Example**:
```python
from genomevault.differential_encoding import DifferenceType

if diff.difference_type == DifferenceType.NEW_MUTATION:
    print("New mutation detected")
```

---

### FunctionalImpact

Variant functional impact levels.

```python
class FunctionalImpact(Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    MODIFIER = "MODIFIER"
```

**Example**:
```python
from genomevault.differential_encoding import FunctionalImpact, get_functional_impact

impact = get_functional_impact(variant)
if impact == FunctionalImpact.HIGH:
    print("High impact variant")
```

---

## Functions

### compute_variant_differences()

Compute differences between experimental and reference sections.

```python
def compute_variant_differences(
    experimental: GenomeSection,
    reference: GenomeSection,
) -> List[VariantDifference]
```

**Parameters**:
- `experimental` (GenomeSection): Experimental genome section
- `reference` (GenomeSection): Reference genome section

**Returns**:
- `List[VariantDifference]`: List of differences

**Example**:
```python
from genomevault.differential_encoding import compute_variant_differences

differences = compute_variant_differences(exp_section, ref_section)

new_mutations = sum(1 for d in differences if d.is_new_mutation)
missing_variants = sum(1 for d in differences if d.is_missing)
genotype_diffs = sum(1 for d in differences if d.is_genotype_diff)
```

---

### compute_reference_hash()

Compute cryptographic hash of reference genome.

```python
def compute_reference_hash(reference: ReferenceGenome) -> str
```

**Parameters**:
- `reference` (ReferenceGenome): Reference genome

**Returns**:
- `str`: SHA-256 hash (hex string)

**Example**:
```python
from genomevault.differential_encoding import compute_reference_hash

hash_value = compute_reference_hash(reference)
print(f"Reference hash: {hash_value[:16]}...")
```

---

### setup_default_references()

Setup recommended reference pool for a use case.

```python
def setup_default_references(
    reference_dir: Path,
    use_case: str = "development",
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> SecureReferenceGenomeManager
```

**Parameters**:
- `reference_dir` (Path): Reference directory
- `use_case` (str): Use case ("development", "research", "clinical", "production")
- `progress_callback` (Optional[Callable]): Progress callback (name, current, total)

**Returns**:
- `SecureReferenceGenomeManager`: Manager with loaded references

**Example**:
```python
from pathlib import Path
from genomevault.differential_encoding import setup_default_references

def progress(name, current, total):
    print(f"{name}: {current}/{total}")

manager = setup_default_references(
    reference_dir=Path("references/"),
    use_case="production",
    progress_callback=progress,
)
```

---

### download_reference_genomes()

Download reference genomes from standard sources.

```python
def download_reference_genomes(
    sources: List[str],
    output_dir: Path,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    force: bool = False,
) -> Dict[str, ReferenceGenome]
```

**Parameters**:
- `sources` (List[str]): Reference source names
- `output_dir` (Path): Output directory
- `progress_callback` (Optional[Callable]): Progress callback
- `force` (bool): Force re-download if exists (default: False)

**Returns**:
- `Dict[str, ReferenceGenome]`: Downloaded references

**Example**:
```python
from pathlib import Path
from genomevault.differential_encoding import download_reference_genomes

references = download_reference_genomes(
    sources=["synthetic_test", "1000g_eur_chr22"],
    output_dir=Path("references/"),
    force=False,
)
```

---

### validate_reference_pool()

Validate integrity of reference pool.

```python
def validate_reference_pool(
    reference_manager: SecureReferenceGenomeManager,
) -> ValidationResult
```

**Parameters**:
- `reference_manager` (SecureReferenceGenomeManager): Manager to validate

**Returns**:
- `ValidationResult`: Validation results

**Example**:
```python
from genomevault.differential_encoding import validate_reference_pool

result = validate_reference_pool(manager)

if result.is_valid:
    print(f"✅ Validation passed ({result.reference_count} references)")
else:
    print(f"❌ Validation failed: {result.errors}")
```

---

## Usage Examples

### Complete Workflow

```python
from pathlib import Path
from genomevault.hypervector_transform import UnifiedGenomicEncoder, EncodingMode
from genomevault.differential_encoding import (
    AnalysisType,
    Genome,
    Variant,
    EncodedGenome,
    DifferentialGenomeQuery,
    setup_default_references,
)

# 1. Setup references
reference_dir = Path("references/")
manager = setup_default_references(reference_dir, use_case="development")

# 2. Create encoder
encoder = UnifiedGenomicEncoder(
    mode=EncodingMode.DIFFERENTIAL,
    reference_dir=reference_dir,
    dimension=10000,
    seed=42,
)

# 3. Create genome
genome = Genome(
    genome_id="patient_001",
    assembly="GRCh38",
    chromosomes={
        "chr1": [
            Variant(chromosome="chr1", position=100000, ref="A", alt="G", genotype="0/1"),
        ]
    }
)

# 4. Encode
encoded = encoder.encode_genome(
    genome=genome,
    analysis_type=AnalysisType.GENE_REGION,
    bundle_chunks=True,
)

# 5. Save
encoded.save(Path("patient_001.enc.gz"), compress=True)

# 6. Load and query
loaded = EncodedGenome.load(Path("patient_001.enc.gz"))

query = DifferentialGenomeQuery(
    reference_manager=encoder.reference_manager,
    hv_encoder=encoder.differential_encoder.hypervector_encoder,
)

result = query.query_region(loaded, "chr1", 50000, 150000)
print(f"Found {result.variant_count} variants")
```

---

## See Also

- [User Guide](differential_encoding_guide.md) - Complete usage guide
- [Basic Example](../examples/differential_encoding_basic.py) - Simple walkthrough
- [Advanced Example](../examples/differential_encoding_advanced.py) - Advanced features
- [Reference Setup Guide](reference_genome_setup.md) - Reference genome setup

---

**For support and updates, see**: [GenomeVault Documentation](../README.md)
