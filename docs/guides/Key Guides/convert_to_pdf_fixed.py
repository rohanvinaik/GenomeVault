#!/usr/bin/env python3
"""
Improved markdown to HTML converter for print-ready PDFs.
Better handling of code blocks, tables, and formatting.
"""

import re
import os
from pathlib import Path
from datetime import datetime
import html

# Improved CSS with better code block handling
PRINT_CSS = """
@page {
    size: letter;
    margin: 0.75in 0.75in 1in 0.75in;
}

* {
    box-sizing: border-box;
}

@media print {
    body {
        font-family: 'Georgia', 'Times New Roman', serif;
        font-size: 11pt;
        line-height: 1.5;
        color: #000;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Helvetica', 'Arial', sans-serif;
        page-break-after: avoid;
        break-after: avoid;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    
    h1 {
        font-size: 20pt;
        border-bottom: 2px solid #333;
        padding-bottom: 0.3em;
        margin-top: 0;
    }
    
    h2 {
        font-size: 16pt;
        border-bottom: 1px solid #666;
        padding-bottom: 0.2em;
        margin-top: 1.2em;
    }
    
    h3 {
        font-size: 14pt;
        margin-top: 1em;
        color: #333;
    }
    
    h4 {
        font-size: 12pt;
        margin-top: 0.8em;
        color: #444;
    }
    
    h5, h6 {
        font-size: 11pt;
        margin-top: 0.6em;
        color: #555;
    }
    
    p {
        margin: 0.5em 0;
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 0.8em 0;
        page-break-inside: avoid;
        font-size: 9pt;
    }
    
    table th {
        background-color: #e0e0e0;
        border: 1px solid #999;
        padding: 6px;
        text-align: left;
        font-weight: bold;
    }
    
    table td {
        border: 1px solid #ccc;
        padding: 5px;
    }
    
    table tr:nth-child(even) {
        background-color: #f8f8f8;
    }
    
    code {
        font-family: 'Courier New', 'Consolas', 'Monaco', monospace;
        background-color: #f0f0f0;
        padding: 1px 3px;
        border-radius: 2px;
        font-size: 9pt;
        word-wrap: break-word;
    }
    
    pre {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 3px;
        padding: 8px;
        margin: 0.8em 0;
        overflow-x: auto;
        page-break-inside: avoid;
        font-size: 8pt;
        line-height: 1.3;
    }
    
    pre code {
        background-color: transparent;
        padding: 0;
        font-size: 8pt;
        display: block;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    blockquote {
        border-left: 3px solid #ccc;
        margin: 0.8em 0;
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
    }
    
    em {
        font-style: italic;
    }
    
    hr {
        border: none;
        border-top: 1px solid #ccc;
        margin: 1.5em 0;
        page-break-after: avoid;
    }
    
    h1, h2, h3, h4, h5, h6 {
        page-break-after: avoid;
    }
    
    table, figure, pre {
        page-break-inside: avoid;
    }
}

@media screen {
    body {
        font-family: 'Georgia', 'Times New Roman', serif;
        font-size: 15px;
        line-height: 1.6;
        max-width: 900px;
        margin: 40px auto;
        padding: 0 20px;
        background-color: #fff;
        color: #333;
    }
    
    h1 {
        font-size: 2.2em;
        border-bottom: 2px solid #333;
        padding-bottom: 0.3em;
        margin-top: 0;
    }
    
    h2 {
        font-size: 1.8em;
        border-bottom: 1px solid #666;
        padding-bottom: 0.2em;
        margin-top: 1.5em;
    }
    
    h3 {
        font-size: 1.4em;
        margin-top: 1.2em;
    }
    
    h4 {
        font-size: 1.2em;
        margin-top: 1em;
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
        display: block;
        overflow-x: auto;
    }
    
    table th {
        background-color: #e0e0e0;
        border: 1px solid #999;
        padding: 10px;
        text-align: left;
    }
    
    table td {
        border: 1px solid #ccc;
        padding: 8px;
    }
    
    table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    code {
        font-family: 'Courier New', 'Consolas', 'Monaco', monospace;
        background-color: #f4f4f4;
        padding: 2px 5px;
        border-radius: 3px;
        font-size: 0.9em;
    }
    
    pre {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 12px;
        overflow-x: auto;
        margin: 1em 0;
    }
    
    pre code {
        background-color: transparent;
        padding: 0;
        font-size: 0.9em;
        display: block;
        white-space: pre;
        overflow-x: auto;
    }
    
    blockquote {
        border-left: 4px solid #ccc;
        margin: 1em 0;
        padding-left: 1em;
        color: #666;
    }
}
"""

def convert_markdown_to_html(md_content):
    """Convert markdown to HTML with improved handling."""
    
    # First, extract and protect code blocks
    code_blocks = []
    def protect_code_block(match):
        code_blocks.append(match.group(2))
        return f"__CODE_BLOCK_{len(code_blocks)-1}__"
    
    # Extract fenced code blocks
    md_content = re.sub(
        r'```(\w+)?\n(.*?)```',
        protect_code_block,
        md_content,
        flags=re.DOTALL
    )
    
    # Extract inline code
    inline_codes = []
    def protect_inline_code(match):
        inline_codes.append(match.group(1))
        return f"__INLINE_CODE_{len(inline_codes)-1}__"
    
    md_content = re.sub(r'`([^`\n]+)`', protect_inline_code, md_content)
    
    # Now process the rest
    lines = md_content.split('\n')
    html_lines = []
    in_list = False
    in_table = False
    table_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Headers
        if line.startswith('# '):
            html_lines.append(f'<h1>{line[2:]}</h1>')
        elif line.startswith('## '):
            html_lines.append(f'<h2>{line[3:]}</h2>')
        elif line.startswith('### '):
            html_lines.append(f'<h3>{line[4:]}</h3>')
        elif line.startswith('#### '):
            html_lines.append(f'<h4>{line[5:]}</h4>')
        elif line.startswith('##### '):
            html_lines.append(f'<h5>{line[6:]}</h5>')
        elif line.startswith('###### '):
            html_lines.append(f'<h6>{line[7:]}</h6>')
        
        # Horizontal rule
        elif re.match(r'^---+$', line.strip()):
            html_lines.append('<hr>')
        
        # Table detection
        elif '|' in line and i + 1 < len(lines) and re.match(r'^\|?[\s\-:|]+\|?$', lines[i + 1]):
            # Start of table
            in_table = True
            table_lines = [line]
            i += 1  # Skip separator line
            
        elif in_table and '|' in line and line.strip():
            table_lines.append(line)
            
        elif in_table and (not line.strip() or '|' not in line):
            # End of table, process it
            html_lines.append(process_table(table_lines))
            in_table = False
            table_lines = []
            if line.strip():
                html_lines.append(f'<p>{line}</p>')
        
        # Lists
        elif re.match(r'^[\*\-\+] ', line):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            content = re.sub(r'^[\*\-\+] ', '', line)
            html_lines.append(f'<li>{content}</li>')
            
        elif re.match(r'^\d+\. ', line):
            if not in_list:
                html_lines.append('<ol>')
                in_list = 'ordered'
            content = re.sub(r'^\d+\. ', '', line)
            html_lines.append(f'<li>{content}</li>')
        
        elif in_list and not line.strip():
            if in_list == 'ordered':
                html_lines.append('</ol>')
            else:
                html_lines.append('</ul>')
            in_list = False
            html_lines.append('')
        
        # Regular paragraph
        elif line.strip():
            html_lines.append(f'<p>{line}</p>')
        else:
            if in_list:
                if in_list == 'ordered':
                    html_lines.append('</ol>')
                else:
                    html_lines.append('</ul>')
                in_list = False
            html_lines.append('')
        
        i += 1
    
    # Close any open lists
    if in_list:
        if in_list == 'ordered':
            html_lines.append('</ol>')
        else:
            html_lines.append('</ul>')
    
    # Process remaining table
    if in_table and table_lines:
        html_lines.append(process_table(table_lines))
    
    html_content = '\n'.join(html_lines)
    
    # Apply text formatting (bold, italic)
    html_content = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', html_content)
    html_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_content)
    html_content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html_content)
    html_content = re.sub(r'___(.+?)___', r'<strong><em>\1</em></strong>', html_content)
    html_content = re.sub(r'__(.+?)__', r'<strong>\1</strong>', html_content)
    html_content = re.sub(r'_(.+?)_', r'<em>\1</em>', html_content)
    
    # Restore inline code
    for idx, code in enumerate(inline_codes):
        escaped_code = html.escape(code)
        html_content = html_content.replace(
            f'__INLINE_CODE_{idx}__',
            f'<code>{escaped_code}</code>'
        )
    
    # Restore code blocks
    for idx, code in enumerate(code_blocks):
        escaped_code = html.escape(code)
        html_content = html_content.replace(
            f'__CODE_BLOCK_{idx}__',
            f'<pre><code>{escaped_code}</code></pre>'
        )
    
    return html_content

def process_table(table_lines):
    """Process a markdown table into HTML."""
    if not table_lines:
        return ''
    
    html = ['<table>']
    
    # Header row
    header = table_lines[0]
    cells = [cell.strip() for cell in header.split('|') if cell.strip()]
    html.append('<thead><tr>')
    for cell in cells:
        # Remove any markdown formatting from header
        cell = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', cell)
        html.append(f'<th>{cell}</th>')
    html.append('</tr></thead>')
    
    # Body rows
    if len(table_lines) > 1:
        html.append('<tbody>')
        for row in table_lines[1:]:
            if '|' in row:
                cells = [cell.strip() for cell in row.split('|') if cell.strip()]
                html.append('<tr>')
                for cell in cells:
                    # Apply formatting
                    cell = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', cell)
                    cell = re.sub(r'\*(.+?)\*', r'<em>\1</em>', cell)
                    html.append(f'<td>{cell}</td>')
                html.append('</tr>')
        html.append('</tbody>')
    
    html.append('</table>')
    return '\n'.join(html)

def create_html_file(md_file_path, output_dir):
    """Create a print-ready HTML file from markdown."""
    
    # Read markdown
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert to HTML
    html_content = convert_markdown_to_html(md_content)
    
    # Get filename
    base_name = Path(md_file_path).stem
    title = base_name.replace('_', ' ')
    
    # Create full HTML document
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{PRINT_CSS}
    </style>
</head>
<body>
{html_content}

    <footer style="margin-top: 2em; padding-top: 1em; border-top: 1px solid #ccc; font-size: 0.85em; color: #666;">
        <p><strong>Document:</strong> {title} | <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        <p style="font-size: 0.9em;">Press Ctrl+P (Windows/Linux) or Cmd+P (Mac) to print as PDF</p>
    </footer>
</body>
</html>"""
    
    # Write output
    output_path = Path(output_dir) / f"{base_name}.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    return output_path

def main():
    md_files = [
        "/Users/rohanvinaik/genomevault/docs/guides/Key Guides/HDV_DIMENSIONALITY_AND_BITPACKING_GUIDE.md",
        "/Users/rohanvinaik/genomevault/docs/guides/Key Guides/QUANTIZATION_COMPARISON_MATRIX.md",
        "/Users/rohanvinaik/genomevault/docs/guides/Key Guides/UNIPOLAR_DUAL_VECTOR_ARCHITECTURE.md",
        "/Users/rohanvinaik/genomevault/docs/guides/Key Guides/HDV_ENCODING_ARCHITECTURE_OPTIMIZATION_THEORY.md"
    ]
    
    output_dir = "/Users/rohanvinaik/genomevault/docs/guides/Key Guides/PDF_Output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Converting markdown files to print-ready HTML...")
    print("=" * 70)
    
    generated_files = []
    for md_file in md_files:
        if os.path.exists(md_file):
            print(f"\n✓ Processing: {Path(md_file).name}")
            html_path = create_html_file(md_file, output_dir)
            generated_files.append(html_path)
        else:
            print(f"\n⚠ File not found: {md_file}")
    
    print("\n" + "=" * 70)
    print(f"✓ SUCCESS! Generated {len(generated_files)} HTML files")
    print("=" * 70)
    print(f"\nOutput: {output_dir}\n")
    
    # Create index
    index_path = Path(output_dir) / "index.html"
    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>GenomeVault Key Guides - Print Ready</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #333;
            margin-top: 0;
            font-size: 2.5em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
        }}
        .subtitle {{
            color: #666;
            font-size: 1.1em;
            margin-bottom: 30px;
        }}
        .docs-list {{
            list-style: none;
            padding: 0;
            margin: 30px 0;
        }}
        .docs-list li {{
            margin: 15px 0;
            background: #f8f9fa;
            border-radius: 8px;
            transition: all 0.3s ease;
            border-left: 4px solid #667eea;
        }}
        .docs-list li:hover {{
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            background: #e9ecef;
        }}
        .docs-list a {{
            display: block;
            padding: 20px;
            color: #333;
            text-decoration: none;
            font-size: 1.15em;
            font-weight: 500;
        }}
        .docs-list a:hover {{
            color: #667eea;
        }}
        .instructions {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            border-radius: 8px;
            margin: 25px 0;
        }}
        .instructions h3 {{
            margin-top: 0;
            color: #856404;
        }}
        .instructions ol {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .instructions li {{
            margin: 8px 0;
            color: #856404;
        }}
        .instructions strong {{
            color: #533f03;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #e9ecef;
            color: #6c757d;
            font-size: 0.9em;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 GenomeVault Key Guides</h1>
        <p class="subtitle">Print-ready documentation for HDV encoding and optimization</p>
        
        <div class="instructions">
            <h3>📄 How to Create PDFs:</h3>
            <ol>
                <li>Click on a document below to open it</li>
                <li>Press <strong>Cmd+P</strong> (Mac) or <strong>Ctrl+P</strong> (Windows/Linux)</li>
                <li>In the print dialog, select <strong>"Save as PDF"</strong></li>
                <li>Choose your save location and click <strong>Save</strong></li>
            </ol>
            <p style="margin: 10px 0 0 0;"><em>💡 Tip: The documents are already optimized for printing with proper margins and page breaks!</em></p>
        </div>
        
        <h2 style="color: #333; margin-top: 30px;">Available Documents:</h2>
        <ul class="docs-list">
"""
    
    for html_file in generated_files:
        title = html_file.stem.replace('_', ' ')
        index_html += f'            <li><a href="{html_file.name}">📄 {title}</a></li>\n'
    
    index_html += f"""        </ul>
        
        <div class="footer">
            Generated: {datetime.now().strftime("%B %d, %Y at %H:%M")}
        </div>
    </div>
</body>
</html>"""
    
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    print("Generated files:")
    for f in generated_files:
        print(f"  • {f.name}")
    print(f"\n✓ Index: index.html (open this for easy access)")

if __name__ == "__main__":
    main()
