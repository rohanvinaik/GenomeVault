#!/usr/bin/env python3
"""
Insert figures into LaTeX file at specific locations.
"""

from pathlib import Path

# Paths
INPUT_FILE = Path("docs/paper_submission/GenomeVault_With_Figures.tex")
OUTPUT_FILE = Path("docs/paper_submission/GenomeVault_Final.tex")
FIG_DIR = "/Users/rohanvinaik/genomevault/docs/paper_figures"

# Figure insertions: (after_line_containing, figure_file, caption, label)
FIGURES = [
    (
        "Figure 1 shows ROC curves and score distributions",
        "figure1_roc_distributions.pdf",
        "ROC Curves and Score Distributions. (A) Aggregate ROC curve showing perfect identification (AUC=1.000). (B) Per-fold ROC curves across 5 folds. (C) Score distributions for genuine vs impostor pairs showing complete separation. (D) DET curve on log-log scale.",
        "fig:roc"
    ),
    (
        "\\textbf{Figure 2: Hyperdimensional Encoding Process}",
        "figure2_hdc_encoding.pdf",
        "Hyperdimensional Encoding Process. (A) Variant binding operation combining chromosome, position, allele, and genotype vectors. (B) Position interpolation preserving linkage disequilibrium. (C) Bundling across multiple variants. (D) Sparsity application for storage optimization.",
        "fig:hdc_encoding"
    ),
    (
        "\\textbf{Figure 3: Zero-Knowledge Proof Circuit}",
        "figure3_zk_performance.pdf",
        "Zero-Knowledge Proof Performance Analysis. (A) Circuit diagram showing 15,234 constraints. (B) Proving time vs constraint count. (C) Memory usage scaling. (D) Backend comparison across metrics.",
        "fig:zk_performance"
    ),
    (
        "\\textbf{Figure 4: PIR Performance Scaling}",
        "figure4_pir_scaling.pdf",
        "Private Information Retrieval Performance Scaling. (A) Latency vs database size. (B) CPIR vs IT-PIR comparison. (C) Network impact analysis. (D) Sharding strategy for large databases.",
        "fig:pir_scaling"
    ),
    (
        "\\textbf{Figure 5: Security Analysis}",
        "figure5_security_analysis.pdf",
        "Security Analysis. (A) Attribute inference attack results across privacy configurations. (B) Privacy configuration comparison. (C) Information leakage bounds showing \\$<\\$7 bits per query. (D) Rate limiting analysis demonstrating effective protection.",
        "fig:security_analysis"
    ),
    (
        "GenomeVault achieves 177",  # After the comparison table in Section 4.9
        "figure6_differential_encoding.pdf",
        "Differential Encoding Performance. (A) Pipeline diagram showing 6-step differential encoding process. (B) Encoding time comparison between GenomeVault and traditional systems (log scale). (C) Storage efficiency comparison showing 2,116× compression (log scale). (D) Chunking strategy performance across different use cases.",
        "fig:differential_encoding"
    ),
]


def create_figure_block(figure_file: str, caption: str, label: str) -> str:
    """Create a figure environment block."""
    return f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=\\textwidth]{{{FIG_DIR}/{figure_file}}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}
"""


def main():
    """Main entry point."""
    print("Inserting figures into LaTeX file...")

    # Read input file
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"✓ Read {INPUT_FILE} ({len(lines)} lines)")

    # Process figures in reverse order to maintain line numbers
    for marker, figure_file, caption, label in reversed(FIGURES):
        # Find the line containing the marker
        found = False
        for i, line in enumerate(lines):
            if marker in line:
                # Insert figure block after this line
                figure_block = create_figure_block(figure_file, caption, label)
                lines.insert(i + 1, figure_block)
                print(f"✓ Inserted {figure_file} after line {i+1}")
                found = True
                break

        if not found:
            print(f"⚠ Warning: Could not find marker: {marker}")

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"\n✓ Generated {OUTPUT_FILE}")
    print(f"  Size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

    return 0


if __name__ == '__main__':
    exit(main())
