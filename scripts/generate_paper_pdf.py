"""
Generate PDF from Academic Paper Markdown

Converts the GenomeVault academic paper from Markdown to PDF using pandoc.
Supports various output formats and citation styles.

Requirements:
    - pandoc installed (brew install pandoc or apt-get install pandoc)
    - Optional: LaTeX for high-quality PDF (brew install mactex)

Usage:
    python scripts/generate_paper_pdf.py
    python scripts/generate_paper_pdf.py --format latex --csl nature.csl
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_pandoc_installed() -> bool:
    """Check if pandoc is installed."""
    return shutil.which('pandoc') is not None


def generate_pdf(
    input_file: Path,
    output_file: Path,
    format: str = 'pdf',
    template: Path = None,
    bibliography: Path = None,
    csl: Path = None,
    metadata: dict = None,
) -> bool:
    """
    Generate PDF from markdown using pandoc.

    Args:
        input_file: Input markdown file
        output_file: Output PDF file
        format: Output format ('pdf', 'latex', 'html')
        template: Optional pandoc template
        bibliography: Optional bibliography file
        csl: Optional CSL style file
        metadata: Optional metadata dictionary

    Returns:
        True if successful, False otherwise
    """

    # Build pandoc command
    cmd = ['pandoc', str(input_file)]

    # Output file
    cmd.extend(['-o', str(output_file)])

    # Format-specific options
    if format == 'pdf':
        # PDF options (using xelatex for Unicode support)
        cmd.extend([
            '--pdf-engine=xelatex',
            '--variable', 'geometry:margin=1in',
            '--variable', 'fontsize=11pt',
            '--variable', 'documentclass=article',
            '--number-sections',
            '--toc',
            '--toc-depth=3',
        ])
    elif format == 'latex':
        cmd.extend([
            '--standalone',
            '--number-sections',
        ])
    elif format == 'html':
        cmd.extend([
            '--standalone',
            '--self-contained',
            '--number-sections',
            '--toc',
            '--toc-depth=3',
            '--mathjax',
        ])

    # Template
    if template and template.exists():
        cmd.extend(['--template', str(template)])
        logger.info(f"Using template: {template}")

    # Bibliography
    if bibliography and bibliography.exists():
        cmd.extend(['--bibliography', str(bibliography)])
        logger.info(f"Using bibliography: {bibliography}")

    # CSL style
    if csl and csl.exists():
        cmd.extend(['--csl', str(csl)])
        logger.info(f"Using CSL style: {csl}")
    elif csl:
        logger.warning(f"CSL file not found: {csl}")

    # Metadata
    if metadata:
        for key, value in metadata.items():
            cmd.extend(['--metadata', f'{key}={value}'])

    # Additional options for better quality
    cmd.extend([
        '--highlight-style=tango',
        '--citeproc',  # For citations (replaces deprecated pandoc-citeproc)
    ])

    # Execute pandoc
    logger.info(f"Generating {format.upper()} output...")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        logger.info(f"Successfully generated: {output_file}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Pandoc conversion failed")
        logger.error(f"Error output:\n{e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("Pandoc not found. Please install pandoc:")
        logger.error("  macOS: brew install pandoc")
        logger.error("  Linux: apt-get install pandoc")
        logger.error("  Windows: choco install pandoc")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate PDF from GenomeVault academic paper"
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('docs/GenomeVault_Academic_Paper.md'),
        help='Input markdown file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('docs/GenomeVault_Academic_Paper.pdf'),
        help='Output PDF file'
    )
    parser.add_argument(
        '--format',
        choices=['pdf', 'latex', 'html'],
        default='pdf',
        help='Output format'
    )
    parser.add_argument(
        '--template',
        type=Path,
        help='Pandoc template file'
    )
    parser.add_argument(
        '--bibliography',
        type=Path,
        default=Path('docs/references.bib'),
        help='Bibliography file (BibTeX)'
    )
    parser.add_argument(
        '--csl',
        type=Path,
        default=Path('docs/nature.csl'),
        help='Citation style (CSL file)'
    )
    parser.add_argument(
        '--title',
        default='GenomeVault: Privacy-Preserving Genomic Computing',
        help='Document title'
    )
    parser.add_argument(
        '--author',
        help='Author name(s)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check pandoc is installed
    if not check_pandoc_installed():
        logger.error("Pandoc is not installed")
        logger.error("Install with: brew install pandoc (macOS)")
        return 1

    # Check input exists
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Prepare metadata
    metadata = {'title': args.title}
    if args.author:
        metadata['author'] = args.author

    # Generate PDF
    logger.info(f"Converting {args.input} to {args.format.upper()}")
    success = generate_pdf(
        input_file=args.input,
        output_file=args.output,
        format=args.format,
        template=args.template,
        bibliography=args.bibliography if args.bibliography.exists() else None,
        csl=args.csl if args.csl and args.csl.exists() else None,
        metadata=metadata,
    )

    if success:
        logger.info(f"✓ PDF generated: {args.output}")
        logger.info(f"  Size: {args.output.stat().st_size / 1024:.1f} KB")
        return 0
    else:
        logger.error("✗ PDF generation failed")
        return 1


if __name__ == '__main__':
    exit(main())
