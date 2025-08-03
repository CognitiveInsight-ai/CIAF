#!/usr/bin/env python3
"""
Markdown to PDF Converter for CIAF Patent Document

This script converts the CIAF Patent Markdown file to a professional PDF
with proper formatting, styling, and image handling.
"""

import markdown2
from weasyprint import HTML, CSS
from pathlib import Path
import os

def convert_markdown_to_pdf(markdown_file, output_pdf=None, css_file=None):
    """Convert Markdown file to PDF with professional styling."""
    
    # Set up paths
    markdown_path = Path(markdown_file)
    if output_pdf is None:
        output_pdf = markdown_path.with_suffix('.pdf')
    
    print(f"Converting {markdown_path} to {output_pdf}")
    
    # Read the markdown file
    with open(markdown_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert markdown to HTML
    html = markdown2.markdown(
        markdown_content,
        extras=[
            'fenced-code-blocks',
            'tables',
            'header-ids',
            'footnotes',
            'toc',
            'strike',
            'break-on-newline'
        ]
    )
    
    # Create a complete HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>CIAF Patent Document</title>
        <style>
            body {{
                font-family: 'Times New Roman', serif;
                line-height: 1.6;
                margin: 2cm;
                color: #333;
                font-size: 12pt;
            }}
            
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #2c3e50;
                padding-bottom: 10px;
                font-size: 24pt;
                text-align: center;
                margin-bottom: 30px;
            }}
            
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 5px;
                font-size: 18pt;
                margin-top: 30px;
            }}
            
            h3 {{
                color: #5d6d7e;
                font-size: 14pt;
                margin-top: 20px;
            }}
            
            p {{
                text-align: justify;
                margin-bottom: 12px;
            }}
            
            strong {{
                color: #2c3e50;
            }}
            
            code {{
                background-color: #f8f9fa;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
            }}
            
            pre {{
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 5px;
                padding: 15px;
                overflow-x: auto;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
                line-height: 1.4;
            }}
            
            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 20px auto;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px;
            }}
            
            .header-info {{
                text-align: center;
                margin-bottom: 40px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }}
            
            .claims {{
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 5px;
                padding: 15px;
                margin: 20px 0;
            }}
            
            .abstract {{
                background-color: #e8f4fd;
                border: 1px solid #74b9ff;
                border-radius: 5px;
                padding: 15px;
                margin: 20px 0;
                font-style: italic;
            }}
            
            ul, ol {{
                margin-bottom: 15px;
            }}
            
            li {{
                margin-bottom: 5px;
            }}
            
            hr {{
                border: none;
                border-top: 2px solid #bdc3c7;
                margin: 30px 0;
            }}
            
            @page {{
                margin: 2cm;
                @bottom-center {{
                    content: counter(page) " of " counter(pages);
                    font-size: 10pt;
                    color: #666;
                }}
            }}
            
            @media print {{
                body {{
                    font-size: 11pt;
                }}
                h1 {{
                    page-break-before: avoid;
                }}
                h2, h3 {{
                    page-break-after: avoid;
                }}
                pre, img {{
                    page-break-inside: avoid;
                }}
            }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    try:
        HTML(string=full_html, base_url=str(markdown_path.parent)).write_pdf(output_pdf)
        print(f"✅ Successfully created PDF: {output_pdf}")
        return True
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        return False

def main():
    """Main function to convert the CIAF Patent document."""
    
    # Set up file paths
    current_dir = Path.cwd()
    markdown_file = current_dir / "CIAF_Patent.md"
    pdf_file = current_dir / "CIAF_Patent.pdf"
    
    if not markdown_file.exists():
        print(f"❌ Markdown file not found: {markdown_file}")
        print(f"Current directory: {current_dir}")
        print("Available files:")
        for file in current_dir.glob("*.md"):
            print(f"  - {file.name}")
        return False
    
    # Convert to PDF
    success = convert_markdown_to_pdf(markdown_file, pdf_file)
    
    if success:
        print(f"\n🎉 Patent document successfully converted to PDF!")
        print(f"📄 Input:  {markdown_file}")
        print(f"📋 Output: {pdf_file}")
        print(f"📊 Size:   {pdf_file.stat().st_size / 1024:.1f} KB")
    
    return success

if __name__ == "__main__":
    main()
