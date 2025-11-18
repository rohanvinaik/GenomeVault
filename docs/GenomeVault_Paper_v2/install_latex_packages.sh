#!/bin/bash
# Install missing LaTeX packages for GenomeVault Paper compilation

echo "Installing LaTeX packages (requires admin password)..."
echo ""

# Install required packages
sudo tlmgr install \
    multirow \
    booktabs \
    algorithms \
    algorithmicx \
    xcolor \
    tikz \
    pgf \
    caption \
    subcaption \
    float

echo ""
echo "✓ LaTeX packages installed successfully!"
echo ""
echo "You can now compile the paper with:"
echo "  pdflatex -output-directory=compiled GenomeVault_Paper.tex"
