# Getting Started with GenomeVault

Welcome to GenomeVault, the privacy-preserving genomic computing platform. This guide will help you get started with using our API and SDKs for secure genomic analysis.

## What is GenomeVault?

GenomeVault enables privacy-preserving genomic computing through advanced cryptographic and mathematical techniques:

- **Privacy by Design**: Mathematical privacy guarantees, not just encryption
- **No Data Storage**: Your genomic data is processed in-memory and never stored
- **Clinical-Grade**: HIPAA-compliant with comprehensive audit trails
- **Scalable**: From single queries to population-scale analysis

## Core Technologies

### Hyperdimensional Computing (HDC)
Transform genomic variants into high-dimensional vectors (8,000+ dimensions) that:
- Preserve genetic information for analysis
- Provide k-anonymity privacy guarantees
- Enable secure similarity search and clustering
- Achieve 50-100x compression ratios

### Private Information Retrieval (PIR)
Query genomic databases without revealing what you're looking for:
- Server cannot determine which records you accessed
- Cryptographically guaranteed privacy
- Suitable for clinical decision support systems

### Zero-Knowledge Proofs
Verify genomic computations without revealing the underlying data:
- Prove variant analysis results without sharing variants
- Enable third-party validation with privacy
- Support regulatory compliance requirements

## Account Setup

### 1. Create an Account
Visit [console.genomevault.io](https://console.genomevault.io) to create your account.

### 2. Generate API Key
In the console, navigate to "API Keys" and create a new key:
```
API Key: gv_1234567890abcdef1234567890abcdef12345678
```

### 3. Choose Your Environment
- **Sandbox**: Free tier for development and testing
- **Clinical**: HIPAA-compliant environment for healthcare applications
- **Research**: High-throughput environment for large-scale studies

## Installation

### Python SDK
```bash
# Install the Python SDK
pip install genomevault-sdk

# Optional dependencies for specific features
pip install genomevault-sdk[vcf]      # VCF file support
pip install genomevault-sdk[clinical] # Clinical analysis features
pip install genomevault-sdk[all]      # All optional dependencies
```

### JavaScript/TypeScript SDK
```bash
# Install the JavaScript SDK
npm install @genomevault/sdk

# Or with Yarn
yarn add @genomevault/sdk
```

### CLI Tool
```bash
# Install the CLI tool
pip install genomevault-cli

# Verify installation
gv --version
```

## First Steps

### 1. Configure Authentication

#### Python
```python
from genomevault_sdk import GenomeVaultClient

# Initialize client with API key
client = GenomeVaultClient(
    api_key="gv_1234567890abcdef1234567890abcdef12345678"
)

# Or use environment variable
import os
os.environ['GENOMEVAULT_API_KEY'] = 'your-api-key'
client = GenomeVaultClient()  # Automatically uses env variable
```

#### JavaScript
```javascript
import { GenomeVaultClient } from '@genomevault/sdk';

const client = new GenomeVaultClient({
  apiKey: 'gv_1234567890abcdef1234567890abcdef12345678'
});
```

#### CLI
```bash
# Configure your API key
gv config set-api-key gv_1234567890abcdef1234567890abcdef12345678

# Verify configuration
gv config show
```

### 2. Test Your Connection

#### Python
```python
import asyncio

async def test_connection():
    try:
        health = await client.health_check()
        print(f"✓ Connected! API version: {health.version}")
        print(f"  Status: {health.status}")
        print(f"  Services: {health.services}")
    except Exception as e:
        print(f"✗ Connection failed: {e}")

# Run the test
asyncio.run(test_connection())
```

#### JavaScript
```javascript
async function testConnection() {
  try {
    const health = await client.healthCheck();
    console.log(`✓ Connected! API version: ${health.version}`);
    console.log(`  Status: ${health.status}`);
    console.log(`  Services:`, health.services);
  } catch (error) {
    console.error(`✗ Connection failed:`, error.message);
  }
}

testConnection();
```

#### CLI
```bash
gv health
```

## Basic Examples

### Example 1: Encode Genomic Variants

Transform genomic variants into privacy-preserving hypervectors:

#### Python
```python
from genomevault_sdk import GenomeVaultClient, GenomicVariant

async def encode_variants_example():
    client = GenomeVaultClient(api_key="your-api-key")

    # Define genomic variants
    variants = [
        GenomicVariant(
            chrom="1",
            pos=1234567,
            ref="A",
            alt="T",
            impact="missense",
            quality=99.5
        ),
        GenomicVariant(
            chrom="2",
            pos=9876543,
            ref="G",
            alt="C",
            impact="synonymous",
            quality=95.2
        )
    ]

    # Encode to hypervector
    result = await client.encode_variants(
        variants=variants,
        dim=8192,        # 8192-dimensional vector
        binary=False     # Continuous values
    )

    print(f"Encoded {len(variants)} variants")
    print(f"Vector dimension: {result.dim}")
    print(f"Privacy level: {result.privacy_level}")
    print(f"Compression ratio: {result.compression_ratio}%")

    return result

# Run the example
result = asyncio.run(encode_variants_example())
```

#### JavaScript
```javascript
async function encodeVariantsExample() {
  const client = new GenomeVaultClient({
    apiKey: 'your-api-key'
  });

  // Define genomic variants
  const variants = [
    {
      chrom: '1',
      pos: 1234567,
      ref: 'A',
      alt: 'T',
      impact: 'missense',
      quality: 99.5
    },
    {
      chrom: '2',
      pos: 9876543,
      ref: 'G',
      alt: 'C',
      impact: 'synonymous',
      quality: 95.2
    }
  ];

  // Encode to hypervector
  const result = await client.encodeVariants(variants, {
    dim: 8192,     // 8192-dimensional vector
    binary: false  // Continuous values
  });

  console.log(`Encoded ${variants.length} variants`);
  console.log(`Vector dimension: ${result.dim}`);
  console.log(`Privacy level: ${result.privacyLevel}`);
  console.log(`Compression ratio: ${result.compressionRatio}%`);

  return result;
}

encodeVariantsExample();
```

#### CLI
```bash
# Create variants file (variants.json)
cat > variants.json << EOF
[
  {
    "chrom": "1",
    "pos": 1234567,
    "ref": "A",
    "alt": "T",
    "impact": "missense",
    "quality": 99.5
  },
  {
    "chrom": "2",
    "pos": 9876543,
    "ref": "G",
    "alt": "C",
    "impact": "synonymous",
    "quality": 95.2
  }
]
EOF

# Encode variants
gv encode variants variants.json --dim 8192 --output encoded.json

# View results
cat encoded.json
```

### Example 2: Private Information Retrieval

Query a genomic database without revealing which record you're accessing:

#### Python
```python
async def pir_query_example():
    client = GenomeVaultClient(api_key="your-api-key")

    # Execute PIR query for index 42
    result = await client.pir_query(
        index=42,
        timeout_seconds=30
    )

    print(f"Query completed in {result.query_time_ms}ms")
    print(f"Retrieved {len(result.item_base64)} bytes (base64)")

    # Decode the retrieved data
    import base64
    decoded_data = base64.b64decode(result.item_base64)
    print(f"Decoded data: {decoded_data}")

    return result

result = asyncio.run(pir_query_example())
```

#### JavaScript
```javascript
async function pirQueryExample() {
  const client = new GenomeVaultClient({
    apiKey: 'your-api-key'
  });

  // Execute PIR query for index 42
  const result = await client.pirQuery(42, {
    timeoutSeconds: 30
  });

  console.log(`Query completed in ${result.queryTimeMs}ms`);
  console.log(`Retrieved ${result.itemBase64.length} bytes (base64)`);

  // Decode the retrieved data
  const decoded = GenomeVaultClient.decodePIRResponse(result.itemBase64);
  console.log('Decoded data:', decoded);

  return result;
}

pirQueryExample();
```

#### CLI
```bash
# Execute PIR query
gv pir query 42 --timeout 30 --output retrieved.data

# View retrieved data
cat retrieved.data
```

### Example 3: Working with VCF Files

Process genomic variants from standard VCF format:

#### Python
```python
async def process_vcf_example():
    client = GenomeVaultClient(api_key="your-api-key")

    # Encode variants directly from VCF file
    result = await client.encode_vcf_variants(
        vcf_path="samples.vcf",
        dim=8192,
        max_variants=1000  # Limit for demo
    )

    print(f"Processed VCF with {result.dim}-dimensional encoding")
    print(f"Privacy level: {result.privacy_level}")

    return result

# Note: Requires pysam: pip install pysam
result = asyncio.run(process_vcf_example())
```

#### CLI
```bash
# Process VCF file directly
gv encode variants sample.vcf --dim 8192 --output encoded_vcf.json

# View compression stats
jq '.compression_ratio' encoded_vcf.json
```

## Working with Different Data Types

### Genomic Coordinates
```python
# Chromosome formats supported
variants = [
    {"chrom": "1", "pos": 1234567, "ref": "A", "alt": "T"},    # Without chr prefix
    {"chrom": "chr1", "pos": 1234567, "ref": "A", "alt": "T"},  # With chr prefix
    {"chrom": "X", "pos": 1234567, "ref": "G", "alt": "A"},     # Sex chromosomes
    {"chrom": "MT", "pos": 1234567, "ref": "C", "alt": "T"},    # Mitochondrial
]
```

### Variant Impacts
```python
# Supported functional impact types
impacts = [
    "missense",      # Amino acid change
    "nonsense",      # Premature stop codon
    "synonymous",    # No amino acid change
    "frameshift",    # Reading frame shift
    "splice_site",   # Affects splicing
    "intron",        # Within intron
    "intergenic"     # Between genes
]
```

### Quality Scores
```python
# Quality score validation
variant = {
    "chrom": "1",
    "pos": 1234567,
    "ref": "A",
    "alt": "T",
    "quality": 99.5  # 0-100 scale
}
```

## Error Handling

### Common Errors and Solutions

#### Authentication Errors
```python
try:
    result = await client.encode_variants(variants)
except AuthenticationError:
    print("Check your API key configuration")
```

#### Validation Errors
```python
try:
    result = await client.encode_variants(variants)
except ValidationError as e:
    print(f"Validation failed: {e.message}")
    for error in e.validation_errors:
        print(f"  {error.field}: {error.message}")
```

#### Rate Limiting
```python
try:
    result = await client.encode_variants(variants)
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
    await asyncio.sleep(e.retry_after)
    # Retry the request
```

### Best Practices

1. **Handle Rate Limits**: Implement exponential backoff for rate limit errors
2. **Validate Input**: Check variant format before sending to API
3. **Use Batch Operations**: Process multiple variants in single requests when possible
4. **Monitor Usage**: Track your API usage in the console
5. **Secure Keys**: Never expose API keys in client-side code

## Next Steps

Now that you're set up, explore advanced features:

- [Privacy Model Deep Dive](./privacy-model.md)
- [Clinical Use Cases](./clinical-examples.md)
- [Research Applications](./research-examples.md)
- [Performance Optimization](./performance.md)
- [Integration Patterns](./integration.md)

## Getting Help

- **Documentation**: [docs.genomevault.io](https://docs.genomevault.io)
- **API Reference**: [api.genomevault.io/docs](https://api.genomevault.io/docs)
- **Examples**: [github.com/genomevault/examples](https://github.com/genomevault/examples)
- **Support**: [support@genomevault.io](mailto:support@genomevault.io)
- **Community**: [discord.gg/genomevault](https://discord.gg/genomevault)
