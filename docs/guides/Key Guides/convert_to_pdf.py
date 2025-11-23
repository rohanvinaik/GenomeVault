#!/usr/bin/env python3
"""
Convert markdown files to print-ready PDFs via HTML with beautiful styling.
"""

import markdown
import os
from pathlib import Path

# Beautiful CSS for printing
PRINT_CSS = """
@page {
    size: letter;
    margin: 0.75in 0.75in 1in 0.75in;
}

@media print {
    body {
        font-family: 'Georgia', 'Times New Roman', serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #000;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Helvetica', 'Arial', sans-serif;
        page-break-after: avoid;
        break-after: avoid;
    }
    
    h1 {
        font-size: 24pt;
        border-bottom: 3px solid #333;
        padding-bottom: 0.3em;
        margin-top: 0;
    }
    
    h2 {
        font-size: 20pt;
        border-bottom: 2px solid #666;
        padding-bottom: 0.2em;
        margin-top: 1.5em;
    }
    
    h3 {
        font-size: 16pt;
        margin-top: 1.2em;
        color: #333;
    }
    
    h4 {
        font-size: 14pt;
        margin-top: 1em;
        color: #444;
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
        page-break-inside: avoid;
        font-size: 10pt;
    }
    
    table th {
        background-color: #e0e0e0;
        border: 1px solid #999;
        padding: 8px;
        text-align: left;
        font-weight: bold;
    }
    
    table td {
        border: 1px solid #ccc;
        padding: 6px;
    }
    
    table tr:nth-child(even) {
        background-color: #f5f5f5;
    }
    
    code {
        font-family: 'Courier New', 'Consolas', monospace;
        background-color: #f0f0f0;
        padding: 2px 4px;
        border-radius: 3px;
        font-size: 10pt;
    }
    
    pre {
        background-color: #f5f5f5;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 10px;
        overflow-x: auto;
        page-break-inside: avoid;
        font-size: 9pt;
    }
    
    pre code {
        background-color: transparent;
        padding: 0;
    }
    
    blockquote {
        border-left: 4px solid #ccc;
        margin: 1em 0;
        padding-left: 1em;
        color: #666;
        font-style: italic;
    }
    
    ul, ol {
        margin: 0.5em 0;
        padding-left: 2em;
    }
    
    li {
        margin: 0.3em 0;
    }
    
    strong {
        font-weight: bold;
        color: #000;
    }
    
    em {
        font-style: italic;
    }
    
    hr {
        border: none;
        border-top: 1px solid #ccc;
        margin: 2em 0;
    }
    
    /* Page breaks */
    h1, h2 {
        page-break-before: auto;
    }
    
    /* Avoid breaks after headings */
    h1, h2, h3, h4, h5, h6 {
        page-break-after: avoid;
    }
    
    /* Keep tables together */
    table, figure {
        page-break-inside: avoid;
    }
    
    /* Footer with page numbers */
    @bottom-right {
        content: counter(page);
    }
}

@media screen {
    body {
        font-family: 'Georgia', 'Times New Roman', serif;
        font-size: 16px;
        line-height: 1.6;
        max-width: 900px;
        margin: 40px auto;
        padding: 0 20px;
        background-color: #fff;
        color: #333;
    }
    
    h1 {
        font-size: 2.5em;
        border-bottom: 3px solid #333;
        padding-bottom: 0.3em;
        margin-top: 0;
    }
    
    h2 {
        font-size: 2em;
        border-bottom: 2px solid #666;
        padding-bottom: 0.2em;
        margin-top: 1.5em;
    }
    
    h3 {
        font-size: 1.5em;
        margin-top: 1.2em;
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
    }
    
    table th {
        background-color: #e0e0e0;
        border: 1px solid #999;
        padding: 12px;
        text-align: left;
    }
    
    table td {
        border: 1px solid #ccc;
        padding: 10px;
    }
    
    table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    code {
        font-family: 'Courier New', 'Consolas', monospace;
        background-color: #f4f4f4;
        padding: 2px 6px;
        border-radius: 3px;
    }
    
    pre {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 15px;
        overflow-x: auto;
    }
    
    pre code {
        background-color: transparent;
        padding: 0;
    }
    
    blockquote {
        border-left: 4px solid #ccc;
        margin: 1em 0;
        padding-left: 1em;
        color: #666;
    }
}
"""

def convert_markdown_to_html(md_file_path, output_dir):
    """Convert a markdown file to a styled HTML file ready for PDF printing."""
    
    # Read markdown content
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    md = markdown.Markdown(extensions=[
        'extra',           # Tables, fenced code blocks, etc.
        'codehilite',      # Code syntax highlighting
        'toc',             # Table of contents
        'sane_lists',      # Better list handling
    ])
    html_content = md.convert(md_content)
    
    # Get the filename without extension
    base_name = Path(md_file_path).stem
    
    # Create full HTML document
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{base_name}</title>
    <style>
{PRINT_CSS}
    </style>
</head>
<body>
    <div class="document-title">
        <h1>{base_name.replace('_', ' ')}</h1>
        <p style="color: #666; font-style: italic;">Generated: {Path(md_file_path).stat().st_mtime}</p>
    </div>
    
{html_content}

    <footer style="margin-top: 3em; padding-top: 1em; border-top: 1px solid #ccc; color: #666; font-size: 0.9em;">
        <p>Document: {base_name} | Print this page (Ctrl+P / Cmd+P) and save as PDF</p>
    </footer>
</body>
</html>"""
    
    # Write HTML file
    output_path = Path(output_dir) / f"{base_name}.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    return output_path

def main():
    # Input files
    md_files = [
        "/Users/rohanvinaik/genomevault/docs/guides/Key Guides/HDV_DIMENSIONALITY_AND_BITPACKING_GUIDE.md",
        "/Users/rohanvinaik/genomevault/docs/guides/Key Guides/QUANTIZATION_COMPARISON_MATRIX.md",
        "/Users/rohanvinaik/genomevault/docs/guides/Key Guides/UNIPOLAR_DUAL_VECTOR_ARCHITECTURE.md",
        "/Users/rohanvinaik/genomevault/docs/guides/Key Guides/HDV_ENCODING_ARCHITECTURE_OPTIMIZATION_THEORY.md"
    ]
    
    # Output directory
    output_dir = "/Users/rohanvinaik/genomevault/docs/guides/Key Guides/PDF_Output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Converting markdown files to print-ready HTML...\n")
    
    generated_files = []
    for md_file in md_files:
        if os.path.exists(md_file):
            print(f"Processing: {Path(md_file).name}")
            html_path = convert_markdown_to_html(md_file, output_dir)
            generated_files.append(html_path)
            print(f"  → Generated: {html_path}\n")
        else:
            print(f"WARNING: File not found: {md_file}\n")
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE!")
    print("="*70)
    print(f"\nGenerated {len(generated_files)} HTML files in:")
    print(f"  {output_dir}\n")
    
    print("TO CREATE PDFs:")
    print("  1. Open each HTML file in your browser")
    print("  2. Press Ctrl+P (Windows/Linux) or Cmd+P (Mac)")
    print("  3. Select 'Save as PDF' as the destination")
    print("  4. Adjust margins if needed (recommend: 0.75in)")
    print("  5. Save!")
    
    print("\nGenerated files:")
    for html_file in generated_files:
        print(f"  • {html_file.name}")
    
    # Create a master index file
    index_path = Path(output_dir) / "index.html"
    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>GenomeVault Documentation - Print Index</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #333;
            padding-bottom: 10px;
        }}
        ul {{
            list-style: none;
            padding: 0;
        }}
        li {{
            margin: 15px 0;
        }}
        a {{
            color: #0066cc;
            text-decoration: none;
            font-size: 1.2em;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .description {{
            color: #666;
            margin-top: 5px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>GenomeVault Key Guides - Print-Ready Documents</h1>
    <p>Click on any document below to open it in your browser, then print to PDF.</p>
    
    <ul>
"""
    
    for html_file in generated_files:
        index_html += f'        <li><a href="{html_file.name}">{html_file.stem.replace("_", " ")}</a></li>\n'
    
    index_html += """    </ul>
    
    <hr>
    <p style="color: #666; font-size: 0.9em;">
        <strong>Printing Tips:</strong><br>
        • Use portrait orientation for best results<br>
        • Set margins to 0.75 inches on all sides<br>
        • Enable background graphics for tables<br>
        • Consider duplex (2-sided) printing to save paper
    </p>
</body>
</html>"""
    
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    print(f"\n✓ Created index file: {index_path}")
    print(f"  Open this in your browser to access all documents!")

if __name__ == "__main__":
    main()
