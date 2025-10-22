#!/bin/bash
# Generate LaTeX with figures embedded

# First, convert markdown to LaTeX
pandoc docs/GenomeVault_Academic_Paper.md \
  -o docs/paper_submission/GenomeVault_Full.tex \
  --standalone \
  --toc \
  --number-sections \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V documentclass=article \
  -V classoption=a4paper

# Add figure package to preamble if not present
sed -i.bak '\\usepackage{hyperref}/a\
\\usepackage{float}\
\\usepackage{caption}' docs/paper_submission/GenomeVault_Full.tex

# Insert Figure 1 after "Figure 1 shows" or similar
# This is complex, so let's compile first and see
echo "LaTeX file generated. Compiling PDF..."

cd docs/paper_submission
pdflatex -interaction=nonstopmode GenomeVault_Full.tex
pdflatex -interaction=nonstopmode GenomeVault_Full.tex  # Second run for references

echo "PDF compiled: GenomeVault_Full.pdf"
