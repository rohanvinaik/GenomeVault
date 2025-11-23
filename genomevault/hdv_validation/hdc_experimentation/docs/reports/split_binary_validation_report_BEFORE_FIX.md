# Split Binary Architecture Validation Report

**6-Bank Within-Lens Splitting Analysis**

**Generated:** 2025-11-21T01:45:04.507945
**Sample Size:** 94 positions

## Architecture Overview

**Split Mapping:**
- **Hydrophobic:** A bank + T bank
- **MajorGroove:** G bank + C bank
- **Hinge:** pos bank + neg bank

**Total:** 6 binary banks, 10,240 dimensions each

## Performance Summary

**Overall Accuracy:** 26.60%

### Per-Nucleotide Accuracy

| Nucleotide | Accuracy | Count |
|------------|----------|-------|
| **A** | 9.09% | 22 |
| **T** | 14.29% | 28 |
| **G** | 37.04% | 27 |
| **C** | 52.94% | 17 |

### Confusion Matrix

```
                  A        T        G        C
         A        0        0        0        0
         T        0        0        0        0
         G        0        0        0        0
         C        0        0        0        0
```

## Example Predictions

**✗ chrX:67361092**
- Ground truth: T
- Predicted: A
- Confidence: 0.0062
- Similarities: {'A': 0.0056640625, 'T': -0.0005859375, 'G': -0.0072265625, 'C': -0.009375}

**✓ chrX:113044835**
- Ground truth: G
- Predicted: G
- Confidence: 0.0004
- Similarities: {'A': -0.00546875, 'T': -0.008984375, 'G': -0.005078125, 'C': -0.0150390625}

**✓ chrX:36172889**
- Ground truth: A
- Predicted: A
- Confidence: 0.0014
- Similarities: {'A': -0.0044921875, 'T': -0.0064453125, 'G': -0.008203125, 'C': -0.005859375}

**✗ chrX:110252073**
- Ground truth: A
- Predicted: G
- Confidence: 0.0012
- Similarities: {'A': -0.003125, 'T': -0.0037109375, 'G': -0.001953125, 'C': -0.0064453125}

**✗ chrX:53610860**
- Ground truth: T
- Predicted: A
- Confidence: 0.0072
- Similarities: {'A': 0.0123046875, 'T': 0.0048828125, 'G': 0.005078125, 'C': 0.004296875}

**✗ chrX:106216094**
- Ground truth: G
- Predicted: A
- Confidence: 0.0016
- Similarities: {'A': -0.0005859375, 'T': -0.005859375, 'G': -0.0021484375, 'C': -0.0146484375}

**✓ chr1:14811133**
- Ground truth: T
- Predicted: T
- Confidence: 0.0018
- Similarities: {'A': -0.0091796875, 'T': -0.007421875, 'G': -0.011328125, 'C': -0.0189453125}

**✗ chr1:118767834**
- Ground truth: G
- Predicted: A
- Confidence: 0.0008
- Similarities: {'A': -0.0140625, 'T': -0.0189453125, 'G': -0.01484375, 'C': -0.0216796875}

**✗ chr1:150849099**
- Ground truth: T
- Predicted: A
- Confidence: 0.0035
- Similarities: {'A': -0.0056640625, 'T': -0.0091796875, 'G': -0.0142578125, 'C': -0.010546875}

**✗ chr1:111929115**
- Ground truth: C
- Predicted: A
- Confidence: 0.0104
- Similarities: {'A': 0.00234375, 'T': -0.008984375, 'G': -0.0080078125, 'C': -0.0083984375}

## File Information

**H5 File:** `genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_6banks_split_binary.h5`
**Random Seed:** 42
