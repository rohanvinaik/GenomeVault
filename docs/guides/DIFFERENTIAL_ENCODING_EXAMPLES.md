# Differential Encoding Examples

This directory contains demonstration scripts for the differential encoding pipeline (Sections 2-7.2 of the specification).

## Working Examples

### ✅ storage_demo.py - Storage and Serialization

**Status**: Fully functional
**Tests**: All 20 storage tests passing

Demonstrates:
- Creating mock encoding results
- Creating EncodedGenome from results
- Saving to compressed storage (gzip)
- Loading and integrity verification
- Compression analysis vs. VCF
- Similarity search with bundled hypervectors

**Run**:
```bash
python examples/storage_demo.py
```

**Output**:
- Creates encoded genome file (~2MB compressed)
- Demonstrates 1.88x gzip compression
- Shows full integrity verification
- Displays hypervector properties
- Simulates similarity comparisons

### ✅ Component-Level Examples

All individual component examples are working:

```bash
# Cryptographic primitives (Section 2)
python examples/differential_encoding_demo.py

# Reference management (Section 3)
python examples/reference_management_demo.py

# Chunking strategies (Section 4)
python examples/chunking_demo.py

# Variant differences (Section 5.1)
python examples/variant_differences_demo.py

# Metadata (Section 5.2)
python examples/metadata_demo.py

# Feature vectors (Section 6.1)
python examples/feature_vectors_demo.py

# Hypervector encoding (Section 6.2)
python examples/hypervector_encoding_demo.py
```

## Pending Examples

### ⏳ complete_pipeline_demo.py - Full End-to-End Pipeline

**Status**: Implemented but requires API compatibility fixes
**Blockers**: 4 API compatibility issues

This comprehensive demo showcases the complete workflow:
1. Creating experimental and reference genomes
2. Setting up the encoding pipeline
3. Encoding an experimental genome
4. Bundling chunk hypervectors
5. Saving to compressed storage
6. Loading and verifying integrity
7. Analyzing compression efficiency
8. Demonstrating similarity search

**Known Issues** (tracked in project todo):

1. **API Parameter Naming** (`pipeline.py`):
   - `Genome.get_chromosome_section()`: Uses `start_position`/`end_position` but should use `start`/`end`
   - `ReferenceGenome.get_section()`: Uses `start_position`/`end_position` but should use `start`/`end`

2. **Reference Selection API** (`pipeline.py`):
   - `SecureReferenceGenomeManager.get_random_reference()`: Missing `seed=` parameter
   - Uses `exclude_ids=` but should use `exclude=`

3. **Chunk Size Optimization** (`chunking.py`):
   - Generating 100k+ chunks for simple genomes (should be ~10-100)
   - Sliding window strategy needs parameter tuning

4. **Test Failures** (`test_pipeline.py`):
   - 7 out of 18 pipeline tests failing due to above API issues
   - Tests are correctly written, implementation needs adjustment

**To Run After Fixes**:
```bash
python examples/complete_pipeline_demo.py
```

## Implementation Status

### Completed (Section 7.2 - Storage)
- ✅ `EncodedGenome` dataclass with all required fields
- ✅ `save()` method with gzip compression
- ✅ `load()` classmethod with integrity verification
- ✅ SHA256 hash validation
- ✅ Hypervector normalization validation
- ✅ JSON serialization with hex-encoded numpy arrays
- ✅ Compression ratio calculation
- ✅ All 20 storage tests passing

### Completed (Section 7.1 - Pipeline)
- ✅ `DifferentialGenomicEncoder` class
- ✅ `encode_experimental_genome()` method
- ✅ `bundle_hypervectors()` superposition
- ✅ `EncodingResult` dataclass
- ✅ Cryptographic binding verification
- ✅ Progress callbacks for UI integration
- ✅ Comprehensive error handling and logging
- ⏳ 11 out of 18 pipeline tests passing (requires API fixes)

### Requires API Fixes
- ⏳ Parameter name consistency across modules
- ⏳ Chunk size optimization for practical use
- ⏳ Full integration tests with real genomic data

## Testing

### Run Storage Tests
```bash
pytest tests/differential_encoding/test_storage.py -v
# Expected: 20/20 passing ✅
```

### Run Pipeline Tests
```bash
pytest tests/differential_encoding/test_pipeline.py -v
# Current: 11/18 passing (requires API fixes)
```

### Run All Differential Encoding Tests
```bash
pytest tests/differential_encoding/ -v
```

## Example Output

### Storage Demo Output

```
================================================================================
🧬 DIFFERENTIAL ENCODING STORAGE DEMO
================================================================================

📊 Creating mock encoding result...
  ✅ Created mock result:
     - 50 chunk hypervectors
     - 50 metadata entries
     - Bundled hypervector: (10000,)
     - Total differences: 972

💾 Creating EncodedGenome for patient_demo_001...
  ✅ EncodedGenome created:
     EncodedGenome(id=patient_demo_001, assembly=GRCh38, chunks=50, dimension=10000, size=4028.0KB)

💾 Saving to compressed storage...
  ✅ Saved to: /tmp/patient_demo_001.enc.gz
     - Compressed file size: 2,203,135 bytes (2151.50 KB)
     - Uncompressed JSON size: 4027.97 KB
     - Compression ratio: 1.87x

📂 Loading encoded genome from storage...
  ✅ Loaded: EncodedGenome(...)
     - Chunks: 50
     - Created: 2025-10-19T15:30:59

🔐 Verifying integrity...
  ✅ Integrity verification PASSED
     - Encoding hash matches
     - All hypervectors normalized
     - Dimensions consistent

📊 Hypervector Properties:
  Bundled hypervector:
    - Dimension: 10000
    - Norm: 1.000000 (should be ~1.0)
    - Range: [-0.0363, 0.0381]

🔍 Similarity Comparison Demo:
  Bundled hypervector shape: (10000,)
  Cosine similarity with similar genome: 0.2234
  Cosine similarity with different genome: -0.0033

================================================================================
✅ STORAGE DEMO COMPLETE
================================================================================
```

## Next Steps

To complete the full pipeline demo:

1. **Fix API Compatibility** (tracked in todo):
   - Update `pipeline.py` parameter names to match module interfaces
   - Standardize on `start`/`end` vs `start_position`/`end_position`
   - Add `seed=` parameter to reference selection
   - Fix `exclude_ids` → `exclude`

2. **Optimize Chunking**:
   - Adjust sliding window parameters for reasonable chunk counts
   - Add validation to prevent excessive chunking
   - Update strategy configs with practical defaults

3. **Verify Integration**:
   - Run complete_pipeline_demo.py
   - Verify 18/18 pipeline tests passing
   - Add performance benchmarks

## Architecture

The differential encoding pipeline follows this data flow:

```
Experimental Genome (VCF)
    ↓
[Chunking] → GenomeChunk[]
    ↓
For each chunk:
    [Reference Selection] → ReferenceGenome
    [Section Extraction] → GenomeSection (experimental + reference)
    [Difference Computation] → VariantDifference[]
    [Feature Vector Generation] → features (95D)
    [Hypervector Encoding] → hypervector (10,000D)
    [Metadata Creation] → DifferentialEncodingMetadata
    ↓
[Bundling] → bundled_hypervector (10,000D)
    ↓
[EncodedGenome Creation] → EncodedGenome
    ↓
[Storage] → .enc.gz file (compressed JSON)
```

## Resources

- **Specification**: See `docs/` for full differential encoding specification
- **Tests**: See `tests/differential_encoding/` for comprehensive test suite
- **Implementation**: See `genomevault/differential_encoding/` for module code
- **Examples**: See `examples/` for component-level demonstrations

## Contact

For questions or issues with the differential encoding implementation, please check the project todo list and issue tracker.
