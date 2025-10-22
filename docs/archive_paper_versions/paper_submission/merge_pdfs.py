#!/usr/bin/env python3
"""
Merge GenomeVault manuscript and appendices into a single PDF.
"""

import sys
from pathlib import Path

try:
    from PyPDF2 import PdfMerger
except ImportError:
    print("PyPDF2 not installed. Trying pypdf...")
    try:
        from pypdf import PdfMerger
    except ImportError:
        print("ERROR: Neither PyPDF2 nor pypdf is installed.")
        print("Install with: pip install pypdf")
        sys.exit(1)

def merge_pdfs():
    """Merge all PDF files into a single document."""

    # Define files to merge in order
    pdf_files = [
        "GenomeVault_Manuscript.pdf",
        "AppendixA_Hypervector_Security.pdf",
        "AppendixB_ZK_Proofs.pdf",
        "AppendixC_Cost_Analysis.pdf"
    ]

    output_file = "GenomeVault_Complete_Submission.pdf"

    # Check that all files exist
    missing_files = []
    for pdf_file in pdf_files:
        if not Path(pdf_file).exists():
            missing_files.append(pdf_file)

    if missing_files:
        print("ERROR: The following PDF files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)

    # Merge PDFs
    print("Merging PDFs...")
    merger = PdfMerger()

    for pdf_file in pdf_files:
        print(f"  Adding: {pdf_file}")
        merger.append(pdf_file)

    # Write output
    print(f"  Writing: {output_file}")
    merger.write(output_file)
    merger.close()

    print(f"\n✓ Successfully created: {output_file}")

    # Print file size
    size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")

    # Print page count
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(output_file)
        print(f"  Total pages: {len(reader.pages)}")
    except:
        try:
            from pypdf import PdfReader
            reader = PdfReader(output_file)
            print(f"  Total pages: {len(reader.pages)}")
        except:
            pass

if __name__ == "__main__":
    merge_pdfs()
