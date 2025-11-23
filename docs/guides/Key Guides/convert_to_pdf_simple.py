#!/usr/bin/env python3
"""
Simple markdown to HTML converter for print-ready PDFs.
No external dependencies required.
"""

import re
import os
from pathlib import Path
from datetime import datetime

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
        page-break-before: auto;
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
        white-space: pre-wrap;
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
        page-break-after: avoid;
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
        overflow-x: auto;
        display: block;
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

def simple_markdown_to_html(md_content):
    """Convert markdown to HTML using simple regex patterns."""
    html = md_content
    
    # Escape HTML
    html = html.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Code blocks (```...```)
    html = re.sub(
        r'```(\w+)?\n(.*?)```',
        lambda m: f'<pre><code class="language-{m.group(1) or ""}">{m.group(2)}</code></pre>',
        html,
        flags=re.DOTALL
    )
    
    # Inline code (`...`)
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    
    # Headers
    html = re.sub(r'^##### (.*?)$', r'<h5>\1</h5>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.*?)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    
    # Bold and italic
    html = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', html)
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
    html = re.sub(r'___(.*?)___', r'<strong><em>\1</em></strong>', html)
    html = re.sub(r'__(.*?)__', r'<strong>\1</strong>', html)
    html = re.sub(r'_(.*?)_', r'<em>\1</em>', html)
    
    # Horizontal rules
    html = re.sub(r'^---+$', '<hr>', html, flags=re.MULTILINE)
    
    # Tables
    lines = html.split('\n')
    in_table = False
    new_lines = []
    
    for i, line in enumerate(lines):
        if '|' in line and i + 1 < len(lines) and re.match(r'^\|?[\s\-:|]+\|?$', lines[i + 1]):
            # Start of table
            in_table = True
            new_lines.append('<table>')
            new_lines.append('<thead>')
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            new_lines.append('<tr>')
            for cell in cells:
                new_lines.append(f'<th>{cell}</th>')
            new_lines.append('</tr>')
            new_lines.append('</thead>')
            new_lines.append('<tbody>')
        elif in_table and '|' in line and not re.match(r'^\|?[\s\-:|]+\|?$', line):
            # Table row
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            new_lines.append('<tr>')
            for cell in cells:
                new_lines.append(f'<td>{cell}</td>')
            new_lines.append('</tr>')
        elif in_table and '|' not in line:
            # End of table
            new_lines.append('</tbody>')
            new_lines.append('</table>')
            in_table = False
            new_lines.append(line)
        elif not re.match(r'^\|?[\s\-:|]+\|?$', line):
            # Regular line
            new_lines.append(line)
    
    if in_table:
        new_lines.append('</tbody>')
        new_lines.append('</table>')
    
    html = '\n'.join(new_lines)
    
    # Lists
    html = re.sub(r'^\* (.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^- (.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^\d+\. (.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    
    # Wrap consecutive <li> in <ul>
    html = re.sub(r'(<li>.*?</li>\n)+', lambda m: f'<ul>\n{m.group(0)}</ul>\n', html, flags=re.DOTALL)
    
    # Paragraphs
    lines = html.split('\n')
    in_para = False
    new_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not any(line.startswith(tag) for tag in ['<h', '<p', '<ul', '<ol', '<li', '<table', '<pre', '<hr', '<blockquote']):
            if not in_para:
                new_lines.append('<p>')
                in_para = True
            new_lines.append(line)
        else:
            if in_para:
                new_lines.append('</p>')
                in_para = False
            if line:
                new_lines.append(line)
    
    if in_para:
        new_lines.append('</p>')
    
    html = '\n'.join(new_lines)
    
    return html

def convert_markdown_to_html(md_file_path, output_dir):
    """Convert a markdown file to a styled HTML file ready for PDF printing."""
    
    # Read markdown content
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = simple_markdown_to_html(md_content)
    
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
{html_content}

    <footer style="margin-top: 3em; padding-top: 1em; border-top: 1px solid #ccc; color: #666; font-size: 0.9em;">
        <p>Document: {base_name} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        <p>Print this page (Ctrl+P / Cmd+P) and save as PDF</p>
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
    
    print("Converting markdown files to print-ready HTML...")
    print("=" * 70)
    
    generated_files = []
    for md_file in md_files:
        if os.path.exists(md_file):
            print(f"\nProcessing: {Path(md_file).name}")
            html_path = convert_markdown_to_html(md_file, output_dir)
            generated_files.append(html_path)
            print(f"  ✓ Generated: {html_path.name}")
        else:
            print(f"\n⚠ WARNING: File not found: {md_file}")
    
    print("\n" + "=" * 70)
    print(f"✓ CONVERSION COMPLETE! Generated {len(generated_files)} files")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}\n")
    
    print("TO CREATE PDFs:")
    print("  1. Open each HTML file in your browser")
    print("  2. Press Ctrl+P (Windows/Linux) or Cmd+P (Mac)")
    print("  3. Select 'Save as PDF' as the destination")
    print("  4. Click 'Save'")
    
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
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #333;
            padding-bottom: 10px;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        ul {{
            list-style: none;
            padding: 0;
        }}
        li {{
            margin: 15px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-left: 4px solid #0066cc;
            border-radius: 4px;
        }}
        li:hover {{
            background-color: #e9e9e9;
        }}
        a {{
            color: #0066cc;
            text-decoration: none;
            font-size: 1.2em;
            font-weight: bold;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .instructions {{
            background-color: #ffffcc;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
            border-left: 4px solid #ffcc00;
        }}
        .instructions h3 {{
            margin-top: 0;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>GenomeVault Key Guides</h1>
        <p style="color: #666;">Click on any document below to open it, then print to PDF.</p>
        
        <div class="instructions">
            <h3>📄 How to Print to PDF:</h3>
            <ol>
                <li>Click on a document link below</li>
                <li>Press <strong>Ctrl+P</strong> (Windows/Linux) or <strong>Cmd+P</strong> (Mac)</li>
                <li>Select <strong>"Save as PDF"</strong> as the destination</li>
                <li>Adjust margins if needed (recommend: 0.75 inches)</li>
                <li>Click Save!</li>
            </ol>
        </div>
        
        <h2>Documents:</h2>
        <ul>
"""
    
    for html_file in generated_files:
        title = html_file.stem.replace('_', ' ')
        index_html += f'            <li><a href="{html_file.name}">{title}</a></li>\n'
    
    index_html += f"""        </ul>
        
        <p style="color: #999; font-size: 0.9em; margin-top: 30px;">
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
        </p>
    </div>
</body>
</html>"""
    
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    print(f"\n✓ Created index file: index.html")
    print(f"  Open this in your browser for easy access to all documents!\n")

if __name__ == "__main__":
    main()
