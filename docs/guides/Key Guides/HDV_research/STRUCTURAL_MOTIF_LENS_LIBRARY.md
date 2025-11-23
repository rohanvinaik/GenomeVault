# Structural Motif Lens Library & Magnitude-Based Compositional Weighting

**Version**: 3.0 (Consolidated Edition - Clarity & Theory Integration)
**Date**: November 21, 2025
**Status**: Production Implementation

---

## Abstract: Lenses as Structured Priors

The **Structural Motif Lens Library** is a decoding-time enhancement that exploits **genomic structure as Bayesian priors**. By treating known motifs as hypervector templates and using compositional bias as automatic gain control, we implement a **multi-hypothesis disambiguation** system that dramatically improves accuracy in low-confidence regions.

**Core Innovation**: Biology has spent billions of years refining highly stereotyped patterns (ALUs, CpGs, TATA boxes). These aren't noise—they're **incredibly high-quality training data** with crisp, sharp features. Our lens library encodes these patterns as decompressor-side information.

**Three-Layer Architecture:**
1. **Physical lenses** (Bank 1-3): Capture universal biophysical properties
2. **Motif lenses** (Library): Encode known genomic patterns
3. **Compositional weighting**: Bayesian priors based on local AT/GC content

Together, these create a **genomic Monty Hall framework** where each lens reveals information that collapses the probability space.

---

## Part I: The Decoding Challenge

### Why Baseline HDC Struggles

**The fundamental problem:**

```
High-dimensional encoding creates distributed representations:
  - Signal: Coherent across D dimensions
  - Noise: Random, √D scaling

Expected accuracy in "easy" regions: 95-99%
Expected accuracy in "difficult" regions: 40-60%

Difficult regions include:
  - Low BAM coverage (<10×)
  - Homopolymer runs (A₇, T₁₀)
  - Repetitive elements (Alu, LINE)
  - GC-extreme regions (>75% or <25%)
```

**The baseline decoder:**
```python
# Naive similarity-only decoding
def baseline_decode(position, chunk_vectors):
    similarities = {
        'A': cosine_sim(query_A_vector, chunk_vectors),
        'T': cosine_sim(query_T_vector, chunk_vectors),
        'G': cosine_sim(query_G_vector, chunk_vectors),
        'C': cosine_sim(query_C_vector, chunk_vectors),
    }
    return max(similarities, key=similarities.get)
```

**What's missing:**
- No use of **genomic structure** (we know CpG islands are GC-rich!)
- No use of **motif knowledge** (we know what Alu looks like!)
- No use of **compositional bias** (AT is rare in CpG islands!)

---

### The Multi-Lens Solution: Monty Hall for Genomics

**The genomic Monty Hall framework:**

```
Initial state: Position i could be {A, T, G, C}
  P(A) = P(T) = P(G) = P(C) = 0.25

Lens 1 (Hydrophobic Bank): "Strong AT signal"
  → Reveals: NOT G, NOT C
  → P(A) = P(T) = 0.50, P(G) = P(C) = 0.0

Lens 2 (Major Groove Bank): "Weak signal"
  → Confirms: NOT GC pair
  → P(A) = P(T) = 0.50 (unchanged)

Lens 3 (Hinge Bank): "Purine step detected"
  → Reveals: Must be purine (A or G)
  → But G already eliminated → Must be A!
  → P(A) = 1.0

Compositional prior: "80% GC region"
  → Bayesian update: P(A | 80% GC) = 0.10 normally
  → But other lenses strongly indicate A
  → Final: P(A) = 0.90 (high confidence despite rarity)
```

**Each lens opens a "door" and reveals information.** The final probability is the product of all constraints.

---

## Part II: Architectural Foundations

### Design Choice: 3 Ternary vs 6 Binary Banks

This system stores **3 ternary banks** {-1, 0, +1} directly, not 6 binary banks {0, 1}. This section documents the architectural analysis that led to this choice.

#### Comparison Table

| Dimension | 3 Ternary Banks | 6 Binary Banks | Winner |
|-----------|-----------------|----------------|--------|
| **Storage** | 0.75D bytes (with 2-bit packing) | 0.75D bytes (with 1-bit packing) | **TIE** |
| **Query Speed** | Direct access, no reconstruction | Requires 3 subtractions (6D ops) | **3 Ternary** |
| **Encoding Compute** | 3 sparsification ops | 6 sparsification ops (50% more) | **3 Ternary** |
| **Accuracy (Monty Hall)** | Natural signed similarity | Requires reconstruction step | **3 Ternary** |
| **Information Theory** | Sub-Shannon via D >> N orthogonality | Sub-Shannon via D >> N orthogonality | **TIE** |

#### Detailed Analysis

**1. Storage Efficiency: TIE**

**3 Ternary Banks:**
- Ternary {-1, 0, +1} requires 2 bits per element (4 states, 3 used)
- 3 banks × D dimensions × 2 bits = 6D bits = **0.75D bytes**
- HDF5 storage: `dtype=np.int8` (1 byte), 25% overhead

**6 Binary Banks:**
- Binary {0, 1} requires 1 bit per element
- 6 banks × D dimensions × 1 bit = 6D bits = **0.75D bytes**
- Optimal packing achieves same storage

**Verdict:** With optimal packing, both formats achieve 0.75D bytes. In practice, both use int8 storage for simplicity.

**2. Query Speed: 3 Ternary WINS**

**3 Ternary Banks:**
```python
# Direct access - O(1) per bank
bank1 = chunk_vectors['bank1']  # Hydrophobic
bank2 = chunk_vectors['bank2']  # Major Groove
bank3 = chunk_vectors['bank3']  # Hinge
# Total: 3 memory reads
```

**6 Binary Banks:**
```python
# Reconstruction required - O(D) per bank
bank1 = bank0_pos - bank0_neg  # Hydrophobic
bank2 = bank1_pos - bank1_neg  # Major Groove
bank3 = bank2_pos - bank2_neg  # Hinge
# Total: 6 memory reads + 3D subtraction ops
```

**Cost:** 6 binary banks require **6D additional operations per query** (3 subtractions × D dimensions).

**Verdict:** 3 ternary banks eliminate reconstruction overhead. For D=5,120, this saves 15,360 operations per query.

**3. Encoding Compute: 3 Ternary WINS**

**3 Ternary Banks:**
- Apply threshold to 3 accumulators
- Sparsify: `{-1, 0, +1}` directly
- **Cost:** 3 sparsification operations

**6 Binary Banks:**
- Split each accumulator into positive/negative components
- Sparsify: `{0, 1}` for each of 6 banks
- **Cost:** 6 sparsification operations

**Verdict:** 3 ternary banks require **50% less compute during encoding**.

**4. Accuracy & Genomic Monty Hall: 3 Ternary WINS**

**Genomic Monty Hall Framework:**
- Cross-validate 3 orthogonal chemical lenses (Hydrophobic, Major Groove, Hinge)
- Each lens provides signed similarity: positive, negative, or neutral
- Natural fit for ternary representation

**3 Ternary Banks:**
```python
# Direct signed similarity computation
sim_A = np.dot(query_pos_vector, bank1)  # Hydrophobic lens
# bank1[i] = -1 → decreases similarity (anti-A)
# bank1[i] = +1 → increases similarity (pro-A)
# bank1[i] = 0 → neutral (no information)
```

**6 Binary Banks:**
```python
# Requires reconstruction before similarity
bank1_ternary = bank0_pos - bank0_neg  # Extra step
sim_A = np.dot(query_pos_vector, bank1_ternary)
```

**Verdict:** Ternary values directly encode "pro", "anti", "neutral" states, aligning perfectly with Monty Hall's signed constraints.

**5. Information Theory (Sub-Shannon Encoding): TIE**

**Key Insight:** The "Shannon violation" comes from **high-dimensional orthogonal projection** (D >> N), NOT from storage format.

**How Sub-Shannon Encoding Works:**

Classical Shannon Limit:
- 4 nucleotides → log₂(4) = 2 bits/nucleotide minimum

GenomeVault's Apparent "Violation":
- Storage: 0.75D bytes for N nucleotides
- For D=5,120 and N=1,024: (0.75 × 5,120 × 8 bits) / 1,024 = **30 bits/nucleotide**

**Wait, that's 15× WORSE than Shannon!**

**The Resolution:** This is **not** direct nucleotide storage. It's a **distributed encoding** where:
1. Each nucleotide influences D dimensions (positional binding)
2. Orthogonal random projections create D-dimensional "smear"
3. SNR amplification: D/N = 5.0 provides 5× redundancy
4. Recovery requires **decoding** (similarity search + Monty Hall)

**Local Information Density:**
- Within each chunk, compositional constraints reduce effective alphabet size
- Magnitude weighting applies Bayesian priors (e.g., AT rare in GC-rich regions)
- Lens library provides structural templates (e.g., "this is Alu")
- Effective information per position: **< 2 bits/nucleotide** after constraints

**Both Formats Achieve This:**
- 3 ternary banks: High-D orthogonal projection with ternary quantization
- 6 binary banks: High-D orthogonal projection with binary quantization
- **Both rely on D >> N for sub-Shannon gains**

**Verdict:** TIE. Information-theoretic advantage comes from dimensionality, not storage format.

#### Final Recommendation: 3 Ternary Banks

**Chosen Architecture:**
- **Storage:** 3 ternary banks {-1, 0, +1}
- **Format:** `dtype=np.int8` (HDF5)
- **Shape:** `(num_chunks, 3, D)`

**Justification:**
1. **Speed:** No reconstruction overhead (6D ops saved per query)
2. **Simplicity:** Natural alignment with Genomic Monty Hall framework
3. **Encoding:** 50% less compute during sparsification
4. **Accuracy:** Direct signed similarity computation
5. **Storage:** Same efficiency as 6 binary with optimal packing

**Trade-off:** Slightly less intuitive than binary, but the performance and conceptual advantages outweigh this.

**Implementation References:**
- Encoder: `encoders/encode_3bank_split_architecture.py` (lines 337-344)
- Decoder: `decoders/lens_aware_decoder_CORRECTED.py`
- Validation: `validate_split_binary.py`

---

### Split Binary and Ternary Storage

The encoder uses **split binary quantization** (creating ternary from two accumulators) but stores **3 ternary banks** {-1, 0, +1}:

```
Given N=1,024 bp chunk with 80% GC content:

Bank 1 (Hydrophobic): {-1=A, 0=GC, +1=T}
  - Effective N ≈ 0.20 × 1,024 = ~205 positions (A+T only)
  - SNR = D / N_eff = 5,120 / 205 = 25.0

Bank 2 (Major Groove): {-1=C, 0=AT, +1=G}
  - Effective N ≈ 0.80 × 1,024 = ~819 positions (G+C in GC-rich)
  - SNR = D / N_eff = 5,120 / 819 = 6.3

Bank 3 (Hinge): {-1=RY, 0=neutral, +1=YR}
  - Effective N ≈ 0.40 × 1,024 = ~410 positions (dinucleotide steps)
  - SNR = D / N_eff = 5,120 / 410 = 12.5
```

**Storage**: Each bank is stored as int8 ternary {-1, 0, +1}

**Key Insight**: Bank 1 (Hydrophobic) has HIGHER SNR in GC-rich regions (fewer active positions), but magnitude-based weighting applies **compositional priors** (Bayesian: AT calls should be rare in GC-rich chunks).

---

## Part III: The Lens Library

### Design Philosophy: Biology as Big Data

**Key insight**: The motifs we care about are:
1. **Highly stereotyped**: ALUs are 99.5% conserved
2. **Evolutionarily refined**: Billions of years of selection
3. **Abundant**: 11% of genome is Alu alone
4. **Crisp**: Strong signal-to-noise (not gradual trends)

**This gives us:**
- Natural justification for aggressive thresholding (tight filters)
- Low risk of deleting real structure (patterns are so sharp)
- Massive training set (the genome itself)
- Cross-validation opportunity (same motif, different technologies)

**Overfitting risk is asymmetric:**

```
In "easy" regions (95% accuracy):
  - BAM is already very accurate
  - Little to gain, little to lose
  - Lenses provide marginal benefit

In "difficult" regions (40-60% accuracy):
  - BAM errors are systematic
  - Huge upside from contextual priors
  - Even slightly risky priors may be net beneficial
```

**The 10-12% that matter:** Short-read pipelines struggle predictably in specific regions. **That's where lenses shine.**

---

### Lens Categories: Three Types of Side Information

#### Category 1: Physical Lenses (Built into Banks)

These are **already encoded** in the split-bank architecture:

| Lens | Bank | Information |
|------|------|-------------|
| Hydrophobic skeleton | Bank 1 | AT-richness, structural rigidity |
| Hydrogen bond topology | Bank 2 | GC-richness, TF binding potential |
| Mechanical flexibility | Bank 3 | YR/RY steps, chromatin accessibility |

**These are NOT in the library—they're the foundation.**

#### Category 2: Genomic Format Priors (Lens Library)

These capture **known genomic motifs**:

| Lens | Prevalence | Size | Bank Signatures |
|------|------------|------|-----------------|
| **ALU_YI** | 11% | ~300 bp | Bank 1: LOW (AT-poor), Bank 2: HIGH (GC-rich + A-tail) |
| **CPG_ISLAND** | 1% | 200-2000 bp | Bank 1: SILENT (no AT), Bank 2: SATURATED (GC), Bank 3: HIGH VARIANCE (CG steps) |
| **TATA_BOX** | 0.1% | ~30 bp | Bank 1: SATURATED (TATA = AT-rich), Bank 3: PERFECT ALTERNATION (Pyr-Pur) |
| **POLY_A** | ~2% | 20-100 bp | Bank 1: CONSTANT (A), Bank 3: FLATLINE (no transitions) |
| **L1_LINE** | 17% | ~6 kb | Bank 1/2: BIMODAL (AT-rich 5' + GC 3') |
| **TELOMERIC** | <0.01% | 10-15 kb | Bank 3: 6bp PERIODICITY (TTAGGG repeat) |
| **CAG_REPEAT** | <0.01% | 30-600 bp | Bank 3: 3bp PERIODICITY (clinical marker) |

**Storage**: ~300 KB (20 lenses × 3 banks × D dimensions × int8)
- Each lens: 3 × 5,120 × 1 byte = ~15 KB
- 20 lenses: ~300 KB (uncompressed)

**Construction:**
```python
# Build consensus hypervector from known instances
alu_instances = find_all_alus(reference_genome)
alu_encoded = [encode_chunk(instance) for instance in alu_instances]

# Average to get consensus (central limit theorem)
alu_lens = {
    'bank1': np.mean([x['bank1'] for x in alu_encoded], axis=0),
    'bank2': np.mean([x['bank2'] for x in alu_encoded], axis=0),
    'bank3': np.mean([x['bank3'] for x in alu_encoded], axis=0),
}
```

#### Category 3: Technology Priors (Optional, Separate)

These capture **platform-specific artifacts**:

| Lens | Platform | Correction |
|------|----------|------------|
| Homopolymer collapse | Nanopore | Correct systematic A₇→A₅ errors |
| GC bias | Illumina | Account for PCR amplification artifacts |
| 5-mer context | Nanopore | Model pore current dependencies |

**Critical distinction:**
- **Genomic lenses** should work identically on Illumina, Nanopore, PacBio, T2T references
- **Technology lenses** are platform-specific corrections
- **Never conflate these** (risk of building a BAM compressor instead of DNA compressor)

---

### Bank 3 as Texture Detector

Bank 3 (Hinge) encodes dinucleotide flexibility patterns. The **rhythm** of Y-R vs R-Y steps creates distinct textures:

**Texture Types**:
- **HOMOPOLYMER**: High magnitude, low variance (Poly-A/T runs)
- **ALTERNATING**: High variance, periodic (TATA boxes - Pyr-Pur-Pyr-Pur)
- **CPG_LIKE**: High magnitude, high variance (CG dinucleotide steps)
- **ALU_LIKE**: Moderate magnitude/variance, GC-rich with A-tail
- **COMPLEX_CODING**: High variance, no pattern (random coding sequences)

**Implementation**:
```python
def classify_texture(hinge_vector):
    """
    Classify genomic region based on Bank 3 pattern

    Uses three features:
      - Magnitude: Overall flexibility signal strength
      - Variance: Flexibility pattern variability
      - Zero-crossing rate (ZCR): Alternation frequency
    """
    magnitude = np.linalg.norm(hinge_vector)
    variance = np.var(hinge_vector)

    # Zero-Crossing Rate (ZCR) - O(N) rhythm detector
    # Optimized vs FFT: O(N) instead of O(N log N)
    # Perfect for binary Purine/Pyrimidine signals
    sign_changes = np.diff(np.sign(hinge_vector)) != 0
    zcr = np.sum(sign_changes) / len(hinge_vector)

    if magnitude > HIGH and zcr < 0.05:
        return 'HOMOPOLYMER'  # Low ZCR = steady state
    elif zcr > 0.8:
        return 'ALTERNATING'  # High ZCR = TATA-like rapid oscillation
    elif magnitude > HIGH and variance > MODERATE:
        return 'CPG_LIKE'
    elif variance > HIGH and magnitude < MODERATE:
        return 'COMPLEX_CODING'
    else:
        return 'ALU_LIKE'
```

**ZCR Interpretation**:

| Pattern | ZCR | Biology |
|---------|-----|---------|
| TATA box | 0.8-1.0 | Perfect Pyr-Pur-Pyr-Pur alternation |
| Homopolymer | 0.0-0.05 | All Pur or all Pyr (no transitions) |
| Random coding | ~0.5 | Average transition rate |
| CpG island | 0.4-0.6 | CG steps (moderate transitions) |

**Computational Cost**: ~10 operations (magnitude + variance + ZCR), O(N)
**Compare to FFT**: ~500 operations for frequency domain

**This is the "texture detector" that routes to appropriate lenses.**

---

## Part IV: Compositional Weighting

### Magnitude-Based Bayesian Priors

**Purpose**: Automatic gain control (AGC) based on local AT/GC composition.

**The insight:**

```
In 80% GC region:
  Bank 1 (Hydrophobic, AT): Sparse, ~205 active positions
  Bank 2 (Major Groove, GC): Dense, ~819 active positions

Naive decoding treats both equally → overpredicts AT

Smart decoding applies Bayesian prior:
  P(A | GC-rich) = 0.10 (rare)
  P(G | GC-rich) = 0.40 (common)
```

**Math** (LINEAR weighting, NOT squared):
```python
def magnitude_aware_decoding(chunk_vectors, query_position):
    # Step 1: Compute magnitudes (cached per chunk - only 7 ops overhead)
    mag_bank1 = np.linalg.norm(chunk_vectors['bank1'])  # Hydrophobic (AT) magnitude
    mag_bank2 = np.linalg.norm(chunk_vectors['bank2'])  # Major Groove (GC) magnitude
    total_mag = mag_bank1 + mag_bank2

    # Step 2: LINEAR compositional ratios (not squared - biologically safer)
    AT_ratio = mag_bank1 / total_mag  # Proportion of AT in chunk
    GC_ratio = mag_bank2 / total_mag

    # Step 3: Apply as compositional priors
    # High AT_ratio → AT calls more likely (Bayesian prior)
    # High GC_ratio → GC calls more likely

    weighted_score = (
        AT_ratio * sim_bank1 +   # Weight Hydrophobic bank by AT composition
        GC_ratio * sim_bank2 +   # Weight Major Groove bank by GC composition
        1.0 * sim_bank3          # Bank 3 (Hinge) is composition-independent
    )
```

**Why LINEAR (not squared)**:
- Squared weighting (e.g., 0.2² = 0.04) is TOO aggressive
- Over-suppresses rare but true nucleotides (e.g., A in GC-rich region)
- Linear weighting (0.2) applies Bayesian prior without excessive suppression
- Lens library provides the main constraint, magnitude is gentle bias

**Computational Cost**:
- Texture classification: ~5-10 ops (ZCR + magnitude + variance, O(N))
- Magnitude weighting: 7 ops (with caching)
- **Total overhead**: ~0.1% per query (negligible)

---

### Cross-Coupling as Information (β and γ Terms)

**Recall from split-bank architecture:**

```python
Bank1[i] = α·f(AT_signal)          # Primary (α=1.0)
         + β·context(i-1:i+1)       # Local context (β=0.1-0.3)
         + γ·groove_coupling        # Cross-channel (γ=0.05-0.1)
```

**These aren't bugs—they're features:**

**β (local context):**
- Encodes dinucleotide dependencies
- Provides error correction (if neighbors disagree, suspect error)
- Captured explicitly in Bank 3 (Hinge)

**γ (cross-channel coupling):**
- Sparse clean signals "ground" dense noisy ones
- In GC-rich region: AT bank (sparse, SNR=25) constrains GC bank (dense, SNR=6)
- Acts like GPS satellites (few strong signals anchor many weak ones)

**Using cross-coupling for confidence:**

```python
def compute_confidence(position, lens_weight_λ):
    """
    How does confidence change as we trust a lens more?

    λ=0: Ignore lens (baseline)
    λ=1: Trust lens fully
    """
    lens_adjusted = {
        'bank1': chunk['bank1'] + λ * lens['bank1'],
        'bank2': chunk['bank2'] + λ * lens['bank2'],
        'bank3': chunk['bank3'] + λ * lens['bank3'],
    }

    sims = compute_similarities(lens_adjusted)
    confidence = max(sims.values()) - median(sims.values())
    return confidence

# Analyze confidence trajectory
for λ in np.linspace(0, 1, 20):
    conf[λ] = compute_confidence(position, λ)

# Classify behavior
if d/dλ conf > 0 everywhere:
    status = "Lens consistent with data"
elif conf peaks then drops:
    status = "Overfitting - possible mismatch"
    # These positions are INTERESTING
```

**This creates an "error field"** that identifies positions where priors and data disagree—exactly where you'd want to investigate.

---

## Part V: The Full Decoding Pipeline

### Order Matters: Lens → Magnitude → Similarity

**Pipeline (correct order):**

```python
def context_aware_decode(position, chunk_vectors, lens_library):
    # Step 1: Classify texture using Bank 3 (Hinge)
    texture = classify_texture(chunk_vectors['bank3'])

    # Step 2: Select lens based on texture
    candidate_lenses = get_lenses_for_texture(texture)
    best_lens = match_best_lens(chunk_vectors, candidate_lenses)

    # Step 3: Apply lens overlay (biological context)
    if best_lens is not None:
        lens_adjusted = {
            'bank1': chunk_vectors['bank1'] + 0.3 * best_lens['bank1'],  # Hydrophobic
            'bank2': chunk_vectors['bank2'] + 0.3 * best_lens['bank2'],  # Major Groove
            'bank3': chunk_vectors['bank3'] + 0.3 * best_lens['bank3'],  # Hinge
        }
    else:
        lens_adjusted = chunk_vectors

    # Step 4: Compute similarities
    sims = compute_nucleotide_similarities(position, lens_adjusted)

    # Step 5: Apply magnitude-based compositional weighting (LINEAR)
    mag_AT = np.linalg.norm(lens_adjusted['bank1'])  # Hydrophobic magnitude
    mag_GC = np.linalg.norm(lens_adjusted['bank2'])  # Major Groove magnitude
    AT_weight = mag_AT / (mag_AT + mag_GC)
    GC_weight = mag_GC / (mag_AT + mag_GC)

    final_scores = {
        'A': AT_weight * sims['A'] + sims['A_hinge'],
        'T': AT_weight * sims['T'] + sims['T_hinge'],
        'G': GC_weight * sims['G'] + sims['G_hinge'],
        'C': GC_weight * sims['C'] + sims['C_hinge'],
    }

    return max(final_scores, key=final_scores.get)
```

**Why this order**:
1. Lens provides biological truth (e.g., "this is Alu, expect GC")
2. Magnitude applies compositional prior within that context
3. Together: strong constraint without over-suppressing rare signals

---

### Performance Expectations

**Baseline (no lenses, no magnitude):**
```
Easy regions (95%+ coverage): 95-99% accuracy
Difficult regions (low coverage): 40-50% accuracy
Overall: ~85% accuracy
```

**With lens library only:**
```
Easy regions: 95-99% (unchanged)
Difficult regions: 50-60% (+10 pts)
Overall: ~88% accuracy (+3%)
```

**With lens + magnitude weighting:**
```
Easy regions: 95-99% (unchanged)
Difficult regions: 55-65% (+15 pts)
Overall: ~90% accuracy (+5%)
```

**Key insight:** Most of the gain comes from **difficult regions** (10-12% of genome).
The lens/magnitude system specifically targets where BAM pipelines struggle.

---

## Part VI: Confidence Trajectory Analysis - The Critical Second Pass

### Rescuing Real Biological Variation

The **confidence trajectory analysis** is the key innovation that prevents lenses from forcing all data to consensus patterns. By sweeping lens weight λ from 0→1 and analyzing how confidence changes, we can distinguish between:
- **Consensus matches**: Genome agrees with canonical motif
- **Real biological variation**: SNPs, rare alleles that differ from consensus
- **Lens mismatches**: Wrong motif template applied

**The critical pattern: Peaks then drops** ⭐
When confidence peaks at intermediate λ (~0.3-0.4) then drops as you approach λ=1, you're seeing **real individual variation** that differs from the consensus motif. This isn't overfitting noise—it's signal about how this specific genome differs from the canonical pattern.

---

### Three Trajectory Patterns

**Pattern 1: Monotonic Increase** → Trust Consensus (λ=1)
```
Confidence: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            λ=0 ----------------------> λ=1

This genome matches the canonical motif.
Action: Use full lens weight.
```

**Pattern 2: Peaks Then Drops** → Real Variation (λ_optimal) ⭐
```
Confidence: [0.3, 0.5, 0.7, 0.85, 0.7, 0.5, 0.3]
            λ=0 ---------↑---------- λ=1
                        peak at λ≈0.3

This genome has a SNP/rare allele vs consensus.
Action: Use λ at peak (optimal balance).
```

**Pattern 3: Monotonic Decrease** → Wrong Lens (λ=0)
```
Confidence: [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
            λ=0 ----------------------> λ=1

Wrong lens selected (e.g., applied Alu to LINE).
Action: Ignore lens, use raw bank signals.
```

---

### Example: SNP in Alu Element

```
Position 1000 in Alu:
  Canonical Alu consensus: A
  This genome: G (rs12345678, 15% MAF)

Naive decoder (λ=1): Forces to consensus A (WRONG)

Trajectory-aware decoder:
  Confidence at λ=0.0: 0.55 (G moderate signal)
  Confidence at λ=0.3: 0.78 (PEAK - context helps, G wins)
  Confidence at λ=1.0: 0.45 (consensus forces A, conflict)

  Pattern: PEAK_THEN_DROP
  Action: Use λ=0.3 → Decodes as G (CORRECT)
```

**Why it works**: At λ=0.3, the lens provides helpful Alu context ("this is a repetitive element") without over-constraining to the consensus base. The data's strong G signal appropriately wins.

---

### The 0.1% That Matters

In a 3 Gbp genome, 0.1% = **3 million variant positions**. These are:
- Common SNPs (population variation)
- Rare alleles (individual-specific)
- De novo mutations
- Population-specific variants

Preserving these is **essential for population genomics and precision medicine**.

---

### Implementation: Two-Pass Decoder

```python
def two_pass_decode_with_trajectory_analysis(chunk_vectors, lens_library, positions):
    """
    First pass: Standard lens-aware decoding (λ=1)
    Second pass: Confidence trajectory refinement for uncertain positions
    """

    # ===== FIRST PASS: Standard Decoding =====
    first_pass_results = {}

    for pos in positions:
        # Select lens based on texture
        texture = classify_texture(chunk_vectors['bank3'])
        lens = select_best_lens(texture, lens_library)

        # Decode with full lens weight
        call, confidence = decode_with_lens(
            pos, chunk_vectors, lens, λ=1
        )

        first_pass_results[pos] = {
            'call': call,
            'confidence': confidence,
            'lens_used': lens
        }

    # ===== SECOND PASS: Trajectory Analysis =====
    final_results = {}

    for pos, result in first_pass_results.items():
        # Only analyze low-confidence positions (< 0.6)
        if result['confidence'] >= CONFIDENCE_THRESHOLD:
            final_results[pos] = result  # High confidence → keep as-is
            continue

        # Sweep lens weight λ from 0 to 1
        λ_range = np.linspace(0, 1, 20)
        confidence_trajectory = []

        for λ in λ_range:
            call_λ, conf_λ = decode_with_lens(
                pos, chunk_vectors, result['lens_used'], λ=λ
            )
            confidence_trajectory.append(conf_λ)

        # Classify trajectory pattern
        pattern = classify_trajectory_pattern(confidence_trajectory)

        if pattern == 'MONOTONIC_INCREASE':
            # Consensus match → use λ=1
            final_results[pos] = result

        elif pattern == 'PEAK_THEN_DROP':
            # REAL VARIATION → use λ at peak
            peak_idx = np.argmax(confidence_trajectory)
            λ_optimal = λ_range[peak_idx]

            call_optimal, conf_optimal = decode_with_lens(
                pos, chunk_vectors, result['lens_used'], λ=λ_optimal
            )

            final_results[pos] = {
                'call': call_optimal,
                'confidence': conf_optimal,
                'pattern': 'real_variation',
                'lambda_optimal': λ_optimal,
                'annotation': f'SNP/variant vs consensus at λ={λ_optimal:.2f}'
            }

        elif pattern == 'MONOTONIC_DECREASE':
            # Wrong lens → ignore it
            call_raw, conf_raw = decode_with_lens(
                pos, chunk_vectors, lens=None, λ=0
            )

            final_results[pos] = {
                'call': call_raw,
                'confidence': conf_raw,
                'pattern': 'lens_mismatch',
                'annotation': 'Lens rejected, using raw signals'
            }

    return final_results


def classify_trajectory_pattern(confidences):
    """
    Classify confidence trajectory into one of three patterns
    """
    gradient = np.gradient(confidences)

    # Pattern 1: Monotonic increase (allow tiny fluctuations)
    if all(gradient > -0.01):
        return 'MONOTONIC_INCREASE'

    # Pattern 2: Peak then drop
    peak_idx = np.argmax(confidences)
    if peak_idx < len(confidences) - 5:  # Peak not at end
        if confidences[-1] < 0.8 * confidences[peak_idx]:
            return 'PEAK_THEN_DROP'  # ⭐ The critical case

    # Pattern 3: Monotonic decrease
    if all(gradient < 0.01):
        return 'MONOTONIC_DECREASE'

    return 'UNCERTAIN'
```

**Computational cost**: ~4× first pass (only analyzes low-confidence positions)

---

### The Safety Valve: Approximate Orthogonality

The β and γ cross-coupling terms between lenses create **controlled interference** that makes the peak-then-drop pattern observable:

**Why "peaks then drops" patterns exist at all**:

```
Perfect orthogonality (no cross-coupling):
  Lens 1 says "A" with 100% confidence
  No other lens can contradict
  Result: Overfit to consensus, lose all variants

Approximate orthogonality (β, γ ≠ 0):
  Lens 1 says "A" (from Alu consensus)
  Lens 2 says "Moderate purine signal" (consistent with A or G)
  Lens 3 says "Some flexibility" (doesn't strongly favor either)

  At λ=0.3:
    Lens 1 provides Alu context (helpful)
    Lens 2/3 create slight "drag" toward G (via β, γ coupling)
    Data's strong G signal wins → confidence peak

  At λ=1.0:
    Lens 1 fully dominates → forces toward A
    Lens 2/3 drag is overwhelmed
    Data-lens conflict → confidence crashes
```

**The redundancy isn't waste—it's the mechanism** that prevents overfitting and creates the signal we use to detect real variation.

**Information-theoretic view**:
```
Mutual information between lenses: I(Lens1, Lens2) ≠ 0

This creates:
  - Error correction when lenses agree
  - Mismatch detection when lenses disagree

Both are essential for preserving individual genomic variation.
```

---

### Summary: The Critical Innovation

**What it does**: Identifies the 0.1% with real variation, automatically tunes λ per position, preserves variants while leveraging priors

**How**: Sweep λ from 0→1, classify trajectory shape, re-decode with optimal λ

**Why it matters**: Enables population genomics, precision medicine; turns approximate orthogonality into variation detection

**Biological parallel**:
> Lenses encode what's conserved (99.9%). Trajectory analysis preserves what's evolving (0.1%). Together: pattern + exceptions = biology.

---

## Part VII: Validation & Avoiding Pitfalls

### Cross-Technology Validation

**To ensure we're capturing biology, not BAM artifacts:**

```python
# Test 1: Core lens invariance
lens_correlations_illumina = compute_lens_usage(illumina_BAM)
lens_correlations_nanopore = compute_lens_usage(nanopore_BAM)
lens_correlations_reference = compute_lens_usage(T2T_CHM13)

# Core lenses (Alu, CpG, TATA) should be consistent
assert correlation(illumina, reference) > 0.90
assert correlation(nanopore, reference) > 0.90

# If strong correlation only in ONE technology → red flag

# Test 2: Synthetic validation
synthetic_reads = generate_reads_with_known_errors(reference)
decode_synthetic = context_aware_decode(synthetic_reads, lens_library)

# Compare known truth vs decoded
accuracy_synthetic = compare(synthetic_reads, decode_synthetic)
# This isolates encoder performance from BAM quality
```

**Technology-specific lenses (if needed) go in a separate layer:**

```python
core_lenses = {
    'alu': alu_consensus,
    'cpg': cpg_consensus,
    'tata': tata_consensus,
    # ... (should work on ALL technologies)
}

tech_lenses = {
    'illumina': {
        'gc_bias_correction': ...,
        'quality_score_model': ...,
    },
    'nanopore': {
        'homopolymer_correction': ...,
        'kmer_context_model': ...,
    },
}

# Always apply core first, tech second
decoded = apply_core_lenses(chunk, core_lenses)
decoded = apply_tech_correction(decoded, tech_lenses[platform])
```

---

### Novel Lens Discovery via Clustering

**Future direction:** Let the system discover new lenses automatically.

**Algorithm:**

```python
# Step 1: Cluster encoded chunks by similarity
from sklearn.cluster import MiniBatchKMeans

all_chunks = load_all_encoded_chunks()  # Shape: (3.37M, 3, 5120)
flattened = all_chunks.reshape(3_370_089, -1)  # Flatten banks

kmeans = MiniBatchKMeans(n_clusters=500, batch_size=10000)
cluster_labels = kmeans.fit_predict(flattened)

# Step 2: Identify large, tight clusters
for cluster_id in range(500):
    cluster_chunks = all_chunks[cluster_labels == cluster_id]

    # Compute cluster statistics
    cluster_size = len(cluster_chunks)
    intra_cluster_variance = np.var(cluster_chunks, axis=0)

    # Filter for large, tight clusters
    if cluster_size > 1000 and intra_cluster_variance < THRESHOLD:
        # This is a candidate novel lens!
        novel_lens = np.mean(cluster_chunks, axis=0)

        # Validate by checking if it matches known motif
        # or represents genuinely new pattern
        validate_novel_lens(novel_lens)
```

**Expected discoveries:**
- Tissue-specific regulatory motifs
- Rare but conserved structural elements
- Technology-artifact patterns (to be filtered)

**Cross-validation:** Test on multiple genomes/technologies to ensure biological relevance.

---

## Part VIII: Summary

### The Three-Layer Framework

```
Layer 1 (Physical): Split-bank architecture
  - Bank 1: Hydrophobic skeleton (AT-exclusive)
  - Bank 2: Hydrogen bond surface (GC-exclusive)
  - Bank 3: Mechanical flexibility (dinucleotide context)
  → Provides orthogonal biophysical signals

Layer 2 (Genomic): Lens library
  - Alu, CpG, TATA, LINE, etc.
  - Precomputed consensus hypervectors
  - Selected via texture classification (Bank 3)
  → Provides structural motif priors

Layer 3 (Compositional): Magnitude weighting
  - Linear AT/GC ratio from bank magnitudes
  - Applies Bayesian compositional priors
  - Gentle nudge (lens does heavy lifting)
  → Provides automatic gain control
```

### The Monty Hall Cascade

```
Query: "Which base at position i?"

Physical lenses reveal:
  Bank 1: "Strong AT signal" → NOT {G,C}
  Bank 2: "Weak GC signal" → Confirms NOT {G,C}
  Bank 3: "Purine step" → Must be purine → Must be A

Genomic lens reveals:
  "This matches Alu consensus" → Expect GC-rich with A-tail
  → Position i in A-tail region → Strongly supports A

Compositional prior:
  Magnitude shows 20% AT in chunk → P(A|local) = 0.10 normally
  → But other constraints override → P(A|all_evidence) = 0.90

Final: A with 90% confidence
```

**Each layer reveals information that collapses the probability space.**

### Key Principles

1. **Order matters**: Lens → Magnitude → Similarity
2. **Linear weighting**: Avoid over-suppression of rare signals
3. **Cross-validation**: Test on multiple platforms
4. **Core vs Tech**: Separate biological from artifactual
5. **Approximate orthogonality**: Cross-coupling is information (β, γ)
6. **Trajectory analysis**: Preserve the 0.1% of real variation

### Expected Impact

**Without lens/magnitude** (baseline): 20% uncertain positions

**With lens library only**: 14% uncertain (6% reduction, 30% improvement)

**With lens + magnitude**: 10% uncertain (10% reduction, 50% improvement)

**On uncertain positions**:
- Expected accuracy improvement: 10-15 percentage points
- From baseline ~50% → enhanced ~60-65%

**Overall accuracy improvement**: +5-10% absolute

---

## Appendix: Implementation References

### Production Files

```
Encoder (3-bank ternary):
  genomevault/hdv_validation/hdc_experimentation/encoders/
    encode_3bank_split_architecture.py

Decoder (lens-aware):
  genomevault/hdv_validation/hdc_experimentation/decoders/
    lens_aware_decoder_CORRECTED.py

Lens Builder:
  genomevault/hdv_validation/hdc_experimentation/encoders/
    build_lens_library.py

Texture Classifier:
  Implemented in lens_aware_decoder_CORRECTED.py
  (classify_texture function, ZCR-based)

Validation:
  genomevault/hdv_validation/hdc_experimentation/
    validate_split_binary.py
```

### Key Parameters

```python
# Lens overlay weight
LENS_ALPHA = 0.3  # How much to trust lens (vs raw data)

# Texture classification thresholds
HIGH_MAGNITUDE = 75th percentile of bank magnitudes
MODERATE_VARIANCE = 50th percentile of variances
ZCR_ALTERNATING = 0.8  # TATA-like rapid transitions
ZCR_HOMOPOLYMER = 0.05  # Steady state

# Magnitude weighting
# Uses LINEAR ratios (not squared)
# AT_weight = mag_AT / (mag_AT + mag_GC)

# Confidence trajectory
CONFIDENCE_THRESHOLD = 0.6  # Triggers second-pass analysis
LAMBDA_SAMPLES = 20  # Sweep resolution for trajectory
```

---

## Implementation Status

**Phase 1** (Completed): Design & documentation
**Phase 2** (Completed): Lens library builder + magnitude-aware decoder
  - `decoders/lens_aware_decoder_CORRECTED.py` (3-ternary architecture)
  - `encoders/build_lens_library.py` (lens library builder)
**Phase 3** (Future): Novel lens discovery via clustering
**Phase 4** (Future): Two-pass trajectory analysis implementation

---

## References

- Split Binary Architecture: `SPLIT_BANK_ARCHITECTURE.md`
- Experimental Data: `EXPERIMENTAL_DATA_COLLECTION.md`
- Implementation: `../decoders/lens_aware_decoder_CORRECTED.py`
- Encoder: `../encoders/encode_3bank_split_architecture.py`
- Lens Builder: `../encoders/build_lens_library.py`

---

**Version**: 3.0 (Consolidated Edition - Clarity & Theory Integration)
**Last Updated**: November 21, 2025
**Next Review**: After cross-platform validation complete

---

## Colophon

The lens library implements **structured Bayesian priors** for genomic decoding. By exploiting billions of years of evolutionary refinement, we turn biology's "big data" into actionable side information.

**The bottom line**: Every lens is a door that Monty Hall opens. Every door reveals information. Enough doors, and you solve the puzzle.
