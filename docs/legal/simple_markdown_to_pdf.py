#!/usr/bin/env python3
"""
Simple Markdown to PDF Converter for CIAF Patent Document

This script converts the CIAF Patent Markdown file to PDF using reportlab
which has better Windows compatibility than weasyprint.
"""

import markdown2
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.platypus import Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, blue, darkblue
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from pathlib import Path
import re
import html2text

def parse_markdown_to_elements(markdown_content):
    """Parse markdown content and return structured elements for PDF."""
    
    # Convert markdown to HTML first
    html = markdown2.markdown(
        markdown_content,
        extras=[
            'fenced-code-blocks',
            'tables', 
            'header-ids',
            'footnotes',
            'break-on-newline'
        ]
    )
    
    # Create styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        spaceBefore=20,
        alignment=TA_CENTER,
        textColor=darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=darkblue
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=15,
        textColor=blue
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        leftIndent=0,
        rightIndent=0
    )
    
    # Parse the content line by line
    lines = markdown_content.split('\n')
    elements = []
    current_paragraph = []
    in_code_block = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines in code blocks
        if in_code_block and line == '':
            continue
            
        # Handle code blocks
        if line.startswith('```'):
            if in_code_block:
                # End of code block
                if current_paragraph:
                    code_text = '\n'.join(current_paragraph)
                    elements.append(Paragraph(f'<font name="Courier" size="9">{code_text}</font>', body_style))
                    current_paragraph = []
                in_code_block = False
            else:
                # Start of code block
                in_code_block = True
            continue
            
        if in_code_block:
            current_paragraph.append(line)
            continue
        
        # Handle headers
        if line.startswith('# '):
            if current_paragraph:
                elements.append(Paragraph(' '.join(current_paragraph), body_style))
                current_paragraph = []
            title_text = line[2:].strip()
            elements.append(Paragraph(title_text, title_style))
            elements.append(Spacer(1, 12))
            
        elif line.startswith('## '):
            if current_paragraph:
                elements.append(Paragraph(' '.join(current_paragraph), body_style))
                current_paragraph = []
            header_text = line[3:].strip()
            elements.append(Paragraph(header_text, heading_style))
            elements.append(Spacer(1, 8))
            
        elif line.startswith('### '):
            if current_paragraph:
                elements.append(Paragraph(' '.join(current_paragraph), body_style))
                current_paragraph = []
            subheader_text = line[4:].strip()
            elements.append(Paragraph(subheader_text, subheading_style))
            elements.append(Spacer(1, 6))
            
        # Handle images (simplified)
        elif '<img' in line or line.startswith('!['):
            if current_paragraph:
                elements.append(Paragraph(' '.join(current_paragraph), body_style))
                current_paragraph = []
            elements.append(Paragraph('<i>[Image placeholder - Original document contains technical diagrams]</i>', body_style))
            elements.append(Spacer(1, 12))
            
        # Handle horizontal rules
        elif line.startswith('---') or line.startswith('***'):
            if current_paragraph:
                elements.append(Paragraph(' '.join(current_paragraph), body_style))
                current_paragraph = []
            elements.append(Spacer(1, 20))
            
        # Handle empty lines
        elif line == '':
            if current_paragraph:
                elements.append(Paragraph(' '.join(current_paragraph), body_style))
                current_paragraph = []
            elements.append(Spacer(1, 6))
            
        # Regular content
        else:
            # Clean up markdown formatting
            line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)  # Bold
            line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', line)      # Italic
            line = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', line)  # Code
            current_paragraph.append(line)
    
    # Add any remaining paragraph
    if current_paragraph:
        elements.append(Paragraph(' '.join(current_paragraph), body_style))
    
    return elements

def convert_markdown_to_pdf_simple(markdown_file, output_pdf=None):
    """Convert Markdown file to PDF using reportlab."""
    
    # Set up paths
    markdown_path = Path(markdown_file)
    if output_pdf is None:
        output_pdf = markdown_path.with_suffix('.pdf')
    
    print(f"Converting {markdown_path} to {output_pdf}")
    
    # Read the markdown file
    with open(markdown_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Parse markdown to elements
    elements = parse_markdown_to_elements(markdown_content)
    
    try:
        # Build PDF
        doc.build(elements)
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
    success = convert_markdown_to_pdf_simple(markdown_file, pdf_file)
    
    if success:
        print(f"\n🎉 Patent document successfully converted to PDF!")
        print(f"📄 Input:  {markdown_file}")
        print(f"📋 Output: {pdf_file}")
        print(f"📊 Size:   {pdf_file.stat().st_size / 1024:.1f} KB")
    
    return success

if __name__ == "__main__":
    main()
