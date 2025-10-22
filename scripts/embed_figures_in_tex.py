#!/usr/bin/env python3
"""
Embed figures in LaTeX file generated from markdown.

Adds graphicx/float packages and inserts proper figure environments
with embedded PDF figures at appropriate locations.
"""

import re
from pathlib import Path

# Paths
TEX_FILE = Path("docs/paper_submission/GenomeVault_Complete.tex")
OUTPUT_FILE = Path("docs/paper_submission/GenomeVault_With_Figures.tex")
FIGURE_DIR = Path("/Users/rohanvinaik/genomevault/docs/paper_figures")

# Figure definitions: (marker_text, figure_file, caption, label)
FIGURES = [
    (
        "ROC curves and score distributions demonstrating perfect separation",
        "figure1_roc_distributions.pdf",
        "ROC Curves and Score Distributions. (A) Aggregate ROC curve showing perfect identification (AUC=1.000). (B) Per-fold ROC curves across 5 folds. (C) Score distributions for genuine vs impostor pairs showing complete separation. (D) DET curve on log-log scale.",
        "fig:roc"
    ),
    (
        "illustrates the HDC encoding process",
        "figure2_hdc_encoding.pdf",
        "Hyperdimensional Encoding Process. (A) Variant binding operation combining chromosome, position, allele, and genotype vectors. (B) Position interpolation preserving linkage disequilibrium. (C) Bundling across multiple variants. (D) Sparsity application for storage optimization.",
        "fig:hdc_encoding"
    ),
    (
        "shows ZK proof performance scaling",
        "figure3_zk_performance.pdf",
        "Zero-Knowledge Proof Performance Analysis. (A) Circuit diagram showing 15,234 constraints. (B) Proving time vs constraint count. (C) Memory usage scaling. (D) Backend comparison across metrics.",
        "fig:zk_performance"
    ),
    (
        "illustrates PIR performance scaling",
        "figure4_pir_scaling.pdf",
        "Private Information Retrieval Performance Scaling. (A) Latency vs database size. (B) CPIR vs IT-PIR comparison. (C) Network impact analysis. (D) Sharding strategy for large databases.",
        "fig:pir_scaling"
    ),
    (
        "visualizes the security analysis results",
        "figure5_security_analysis.pdf",
        "Security Analysis. (A) Attribute inference attack results across privacy configurations. (B) Privacy configuration comparison. (C) Information leakage bounds showing $<$7 bits per query. (D) Rate limiting analysis demonstrating effective protection.",
        "fig:security_analysis"
    ),
    (
        "NEW differential encoding",
        "figure6_differential_encoding.pdf",
        "Differential Encoding Performance. (A) Pipeline diagram showing 6-step differential encoding process. (B) Encoding time comparison between GenomeVault and traditional systems (log scale). (C) Storage efficiency comparison showing 2,116× compression (log scale). (D) Chunking strategy performance across different use cases.",
        "fig:differential_encoding"
    ),
]


def add_packages(content: str) -> str:
    """Add graphicx and float packages after geometry."""

    # Find where to insert packages (after geometry)
    geometry_pattern = r'\\usepackage\[margin=1in\]\{geometry\}'

    packages_to_add = r"""\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}"""

    content = content.replace(
        r'\usepackage[margin=1in]{geometry}',
        packages_to_add,
        1
    )

    return content


def insert_figure(content: str, marker: str, figure_file: str, caption: str, label: str) -> str:
    """Insert figure environment at appropriate location."""

    # Create figure block
    figure_block = f"""

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=\\textwidth]{{{str(FIGURE_DIR)}/{figure_file}}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}
"""

    # Find marker in content and insert figure after the paragraph
    # Look for the marker text followed by a period and newline
    pattern = re.escape(marker) + r'[^.]*\.'

    def replacer(match):
        return match.group(0) + figure_block

    content = re.sub(pattern, replacer, content, count=1, flags=re.IGNORECASE)

    return content


def main():
    """Main entry point."""
    print("Embedding figures in LaTeX file...")

    # Read original tex file
    if not TEX_FILE.exists():
        print(f"Error: {TEX_FILE} not found")
        return 1

    with open(TEX_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"✓ Read {TEX_FILE}")

    # Add packages
    content = add_packages(content)
    print("✓ Added graphicx and float packages")

    # Insert each figure
    for marker, figure_file, caption, label in FIGURES:
        if not (FIGURE_DIR / figure_file).exists():
            print(f"⚠ Warning: Figure not found: {figure_file}")
            continue

        original_len = len(content)
        content = insert_figure(content, marker, figure_file, caption, label)

        if len(content) > original_len:
            print(f"✓ Inserted {figure_file}")
        else:
            print(f"⚠ Warning: Could not find marker for {figure_file}: '{marker}'")

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n✓ Generated {OUTPUT_FILE}")
    print(f"  Size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

    return 0


if __name__ == '__main__':
    exit(main())
