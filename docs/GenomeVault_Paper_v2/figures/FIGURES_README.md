# Figures Required for GenomeVault Paper

This document lists all figures referenced in the paper and provides specifications for creating them.

## Figure 1: Pipeline Overview (`pipeline_overview.pdf`)

**Description:** GenomeVault end-to-end pipeline from FASTQ to federated learning

**Content:**
- 7-stage flowchart with arrows
- Each stage labeled with:
  - Input/output formats
  - Latency (seconds/milliseconds)
  - Compression ratio
  - Security level
- Color-coded by privacy layer (blue = data, green = crypto, red = blockchain)

**Data Source:**
- `benchmarks/run_alignment_optimized_pipeline.py` results
- `docs/reports/COMPLETE_BENCHMARK_RESULTS.md`

**Suggested Tool:** Python matplotlib/seaborn or TikZ

---

## Figure 2: Dual-Barrier Architecture (`dual_barrier.pdf`)

**Description:** SHA-256² dual-barrier security architecture diagram

**Content:**
- Two parallel protection layers:
  - Layer 1: AES-256-GCM encryption box (256 bits)
  - Layer 2: Alignment randomization (260 bits)
- Mathematical formula showing multiplicative security: 2^256 × 2^260 = 2^516
- Attack arrows showing both must be broken simultaneously

**Data Source:**
- `docs/guides/HYPERVECTOR_SECURITY.md`
- Theoretical analysis section

**Suggested Tool:** TikZ or Inkscape

---

## Figure 3: Multi-Run Consensus Accuracy (`multirun_consensus.pdf`)

**Description:** Plot showing error rate vs. number of consensus runs

**Content:**
- X-axis: Number of runs (1, 3, 5, 7, 9)
- Y-axis: Error rate (log scale, 0.0001% to 10%)
- Exponential decay curve
- Horizontal line at 99.9% (clinical threshold)
- Data points with error bars

**Data Source:**
- `docs/reports/ALIGNMENT_OPTIMIZATION_RESULTS_SUMMARY.md`
- Consensus accuracy table

**Suggested Tool:** Python matplotlib

**Code Skeleton:**
```python
import matplotlib.pyplot as plt
import numpy as np

runs = [1, 3, 5, 7, 9]
error_rates = [5.0, 1.4, 0.1, 0.01, 0.001]  # percent

plt.semilogy(runs, error_rates, 'o-', linewidth=2)
plt.axhline(y=0.1, color='r', linestyle='--', label='Clinical threshold')
plt.xlabel('Number of Consensus Runs')
plt.ylabel('Error Rate (%)')
plt.title('Consensus Accuracy Improvement')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('multirun_consensus.pdf')
```

---

## Figure 4: HDC Collision Probability (`hdc_collision.pdf`)

**Description:** Collision probability vs. hypervector dimension

**Content:**
- X-axis: Dimension (2048, 4096, 8192, 16384, 32768)
- Y-axis: Collision probability (log scale)
- Curve showing exponential decrease
- Horizontal line at 10^-4 (target threshold)
- Annotation: "GenomeVault uses D=8192"

**Data Source:**
- Theoretical formula from paper Section 3 (Mathematical Foundations)
- P_collision ≤ exp(-D·δ²/2) / sqrt(2πD)

**Suggested Tool:** Python matplotlib

---

## Figure 5: Pipeline Stage Breakdown (`pipeline_breakdown.pdf`)

**Description:** Bar chart of latency contributions by pipeline stage

**Content:**
- X-axis: Pipeline stages
- Y-axis: Latency (milliseconds, log scale)
- Stacked or grouped bars showing:
  - Differential Encoding: 1360 ms
  - HDC Encoding: 0.5 ms
  - ZK Proof: 740 ms
  - Blockchain: 80 ms
  - PIR Query: 4.33 ms

**Data Source:**
- `benchmark_results/full_pipeline_results/pipeline_run_*/pipeline_results.json`
- Table in paper Section 5 (Performance Evaluation)

**Suggested Tool:** Python matplotlib/seaborn

---

## Figure 6: Scaling Analysis (`scaling_variants.pdf`)

**Description:** Pipeline latency vs. variant count

**Content:**
- X-axis: Variant count (100k, 500k, 1M, 2M, 4M, 5M)
- Y-axis: Total latency (seconds)
- Linear fit: T(n) = 1.2 + 0.00035n
- Data points from benchmarks
- Shaded region showing confidence interval

**Data Source:**
- Run pipeline with varying variant counts using synthetic data
- `benchmarks/full_pipeline_synthetic_data.sh` with different parameters

**Suggested Tool:** Python matplotlib

---

## Figure 7: Storage Cost Comparison (`storage_comparison.pdf`)

**Description:** Storage cost for 100M genomes across different formats

**Content:**
- Bar chart with logarithmic Y-axis
- Formats: FASTQ, BAM, CRAM, VCF, GenomeVault
- Heights: 150 PB, 60 PB, 20 PB, 5 PB, 0.039 PB
- Cost annotations in USD/year
- Highlight GenomeVault's 38.4× savings

**Data Source:**
- Economic analysis in paper Section 6
- `docs/reports/COMPLETE_BENCHMARK_RESULTS.md`

**Suggested Tool:** Python matplotlib

---

## Figure 8: Economic Scaling (`economic_scaling.pdf`)

**Description:** Total storage cost vs. biobank size

**Content:**
- X-axis: Number of genomes (log scale: 10k, 100k, 1M, 10M, 100M, 1B)
- Y-axis: Annual cost (USD, log scale)
- Two lines: VCF (steep), GenomeVault (flat)
- Crossover point annotation
- Cost savings region shaded

**Data Source:**
- Economic model section in paper
- \$0.02/GB/month cloud storage pricing

**Suggested Tool:** Python matplotlib

---

## Figure 9: ZK Proof Lifecycle (`zk_proof_flow.pdf`)

**Description:** Flow diagram of zero-knowledge proof generation and verification

**Content:**
- Flowchart showing:
  1. Variant set → Circuit input
  2. Proof generation (0.768s)
  3. Blockchain commitment
  4. Verification (<10ms)
  5. Result (valid/invalid)
- Include proof size (743 bytes) and constraint count (117,143)

**Data Source:**
- `genomevault/zk_proofs/` implementation
- `benchmarks/zk_groth16_benchmark.py` results

**Suggested Tool:** TikZ or diagrams.net

---

## Figure 10: PIR Query Flow (`pir_flow.pdf`)

**Description:** Information-theoretic PIR protocol diagram

**Content:**
- 3-server architecture
- Client generates XOR-secret-shared queries
- Each server computes partial response
- Client XORs responses to recover result
- Mathematical formula: I(Query; Server_i) = 0
- Latency annotation: 6.85ms

**Data Source:**
- `genomevault/pir/it_pir_protocol.py`
- IT-PIR implementation details

**Suggested Tool:** TikZ

---

## Figure 11: Blockchain Attestation Architecture (`blockchain_architecture.pdf`)

**Description:** Blockchain integration showing smart contract and off-chain storage

**Content:**
- Smart contract on Polygon PoS (on-chain)
- Merkle tree commitments
- IPFS/Filecoin off-chain storage
- Arrows showing data flow
- Cost: $0.01/attestation, <100ms confirmation

**Data Source:**
- `genomevault/blockchain/attestation_registry.py`
- `docs/reports/BLOCKCHAIN_INTEGRATION_COMPLETE.md`

**Suggested Tool:** TikZ or diagrams.net

---

## Figure 12: KAN-HD Architecture (`kan_hd_pipeline.pdf`)

**Description:** KAN-HD encoding with learnable B-splines

**Content:**
- Input: Variant vector
- B-spline basis functions (visualized)
- Hypervector output
- Selective decode path (reverse arrow)
- Accuracy: 99.7% for target regions
- Latency: 15ms encoding

**Data Source:**
- KAN-HD theory from paper outline
- Future work section (not yet fully implemented)

**Suggested Tool:** Python matplotlib + TikZ

---

## Figure Generation Scripts

All figures can be generated using:

```bash
# Install dependencies
pip install matplotlib seaborn numpy pandas

# Generate all figures
python analysis/generate_paper_figures.py --output docs/GenomeVault_Paper_v2/figures/

# Individual figure generation
python analysis/plot_pipeline_breakdown.py
python analysis/plot_consensus_accuracy.py
python analysis/plot_scaling_analysis.py
```

## Figure Specifications

**Format:** PDF (vector graphics for scalability)
**Resolution:** 300 DPI minimum for raster elements
**Fonts:** Match paper font (default LaTeX Computer Modern)
**Colors:** Use colorblind-friendly palette (e.g., Seaborn "colorblind")
**Size:** Single column (3.5 inches) or double column (7 inches) width

## Data Availability

All benchmark data for figures is available in:
- `benchmark_results/` directory
- `docs/reports/` summaries
- Raw data can be regenerated using scripts in `benchmarks/`

## Notes

- Figures 1, 2, 9, 10, 11 are primarily schematic (TikZ recommended)
- Figures 3, 4, 5, 6, 7, 8 are data-driven (matplotlib recommended)
- Figure 12 combines theory and data (hybrid approach)

For questions about figure specifications, see paper sections or contact authors.
