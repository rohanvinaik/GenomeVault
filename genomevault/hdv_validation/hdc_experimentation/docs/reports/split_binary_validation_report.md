# Split Binary Architecture Validation Report

**6-Bank Within-Lens Splitting Analysis**

**Generated:** 2025-11-21T09:58:43.451004
**Sample Size:** 9,544 positions

## Architecture Overview

**Split Mapping:**
- **Hydrophobic:** A bank + T bank
- **MajorGroove:** G bank + C bank
- **Hinge:** pos bank + neg bank

**Total:** 6 binary banks, 10,240 dimensions each

## Performance Summary

**Overall Accuracy:** 36.47%

### Per-Nucleotide Accuracy

| Nucleotide | Accuracy | Count |
|------------|----------|-------|
| **A** | 35.87% | 2,813 |
| **T** | 36.38% | 2,779 |
| **G** | 36.23% | 1,924 |
| **C** | 37.67% | 2,028 |

### Confusion Matrix

```
                  A        T        G        C
         A        0        0        0        0
         T        0        0        0        0
         G        0        0        0        0
         C        0        0        0        0
```

## Example Predictions

**✓ chrX:67361092**
- Ground truth: T
- Predicted: T
- Confidence: 0.0062
- Similarities: {'A': -0.0005859375, 'T': 0.0056640625, 'G': -0.0072265625, 'C': -0.009375}

**✓ chrX:113044835**
- Ground truth: G
- Predicted: G
- Confidence: 0.0004
- Similarities: {'A': -0.008984375, 'T': -0.00546875, 'G': -0.005078125, 'C': -0.0150390625}

**✗ chrX:36172889**
- Ground truth: A
- Predicted: T
- Confidence: 0.0014
- Similarities: {'A': -0.0064453125, 'T': -0.0044921875, 'G': -0.008203125, 'C': -0.005859375}

**✗ chrX:110252073**
- Ground truth: A
- Predicted: G
- Confidence: 0.0012
- Similarities: {'A': -0.0037109375, 'T': -0.003125, 'G': -0.001953125, 'C': -0.0064453125}

**✓ chrX:53610860**
- Ground truth: T
- Predicted: T
- Confidence: 0.0072
- Similarities: {'A': 0.0048828125, 'T': 0.0123046875, 'G': 0.005078125, 'C': 0.004296875}

**✗ chrX:106216094**
- Ground truth: G
- Predicted: T
- Confidence: 0.0016
- Similarities: {'A': -0.005859375, 'T': -0.0005859375, 'G': -0.0021484375, 'C': -0.0146484375}

**✗ chrX:15557833**
- Ground truth: G
- Predicted: A
- Confidence: 0.0008
- Similarities: {'A': -0.00078125, 'T': -0.0015625, 'G': -0.0068359375, 'C': -0.0099609375}

**✓ chrX:116417901**
- Ground truth: G
- Predicted: G
- Confidence: 0.0021
- Similarities: {'A': 0.005078125, 'T': -0.0037109375, 'G': 0.0072265625, 'C': 0.0009765625}

**✗ chrX:9429785**
- Ground truth: T
- Predicted: G
- Confidence: 0.0027
- Similarities: {'A': -0.0123046875, 'T': -0.012109375, 'G': -0.009375, 'C': -0.0177734375}

**✓ chrX:4148181**
- Ground truth: A
- Predicted: A
- Confidence: 0.0053
- Similarities: {'A': 0.0078125, 'T': 0.0025390625, 'G': 0.0021484375, 'C': -0.0029296875}

## File Information

**H5 File:** `genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_6banks_split_binary.h5`
**Random Seed:** 42
