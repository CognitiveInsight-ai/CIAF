#!/usr/bin/env python3
"""
Enhanced Markdown to PDF Converter for CIAF Patent Document with Image Support

This script converts the CIAF Patent Markdown file to PDF using reportlab
and includes actual images from the Images directory.
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
import os
from PIL import Image

def get_image_path(image_ref, base_dir):
    """Extract and resolve image path from markdown image reference."""
    
    # Handle different image reference formats
    if image_ref.startswith('!['):
        # Standard markdown format: ![alt text](path)
        match = re.search(r'!\[.*?\]\((.*?)\)', image_ref)
        if match:
            image_path = match.group(1)
        else:
            return None
    elif '<img' in image_ref:
        # HTML img tag format
        match = re.search(r'src=["\']([^"\']+)["\']', image_ref)
        if match:
            image_path = match.group(1)
        else:
            return None
    else:
        return None
    
    # Clean up path
    image_path = image_path.strip()
    
    # Handle relative paths
    if image_path.startswith('/PatentDocs/Images/'):
        image_path = image_path.replace('/PatentDocs/Images/', 'Images/')
    elif image_path.startswith('/Images/'):
        image_path = image_path.replace('/Images/', 'Images/')
    
    # Resolve full path
    full_path = Path(base_dir) / image_path
    
    # Check if file exists (handle .png.png issue)
    if not full_path.exists():
        # Try with .png.png extension
        png_png_path = Path(str(full_path) + '.png')
        if png_png_path.exists():
            full_path = png_png_path
        # Try other common extensions
        elif full_path.with_suffix('.png').exists():
            full_path = full_path.with_suffix('.png')
        elif full_path.with_suffix('.jpg').exists():
            full_path = full_path.with_suffix('.jpg')
        elif full_path.with_suffix('.jpeg').exists():
            full_path = full_path.with_suffix('.jpeg')
    
    return full_path if full_path.exists() else None

def create_image_element(image_path, max_width=6*inch, max_height=4*inch):
    """Create a reportlab Image element with proper sizing."""
    
    try:
        # Open image to get dimensions
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        
        # Calculate scaling to fit within max dimensions while preserving aspect ratio
        width_scale = max_width / img_width
        height_scale = max_height / img_height
        scale = min(width_scale, height_scale, 1.0)  # Don't scale up
        
        final_width = img_width * scale
        final_height = img_height * scale
        
        # Create reportlab Image
        return RLImage(str(image_path), width=final_width, height=final_height)
        
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        return None

def parse_markdown_to_elements_with_images(markdown_content, base_dir):
    """Parse markdown content and return structured elements for PDF with image support."""
    
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
    
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=12,
        spaceBefore=6,
        alignment=TA_CENTER,
        textColor=blue,
        fontName='Helvetica-Oblique'
    )
    
    # Parse the content line by line
    lines = markdown_content.split('\n')
    elements = []
    current_paragraph = []
    in_code_block = False
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines in code blocks
        if in_code_block and line == '':
            i += 1
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
            i += 1
            continue
            
        if in_code_block:
            current_paragraph.append(line)
            i += 1
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
            
        # Handle images
        elif line.startswith('![') or '<img' in line:
            if current_paragraph:
                elements.append(Paragraph(' '.join(current_paragraph), body_style))
                current_paragraph = []
            
            # Get image path
            image_path = get_image_path(line, base_dir)
            
            if image_path:
                # Create and add image
                img_element = create_image_element(image_path)
                if img_element:
                    elements.append(Spacer(1, 12))
                    elements.append(img_element)
                    
                    # Look for caption in next line
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line.startswith('*') and next_line.endswith('*'):
                            caption_text = next_line[1:-1]  # Remove asterisks
                            elements.append(Paragraph(caption_text, caption_style))
                            i += 1  # Skip the caption line
                    
                    elements.append(Spacer(1, 12))
                else:
                    elements.append(Paragraph(f'<i>[Image not found: {image_path}]</i>', body_style))
            else:
                elements.append(Paragraph('<i>[Image placeholder - Could not resolve path]</i>', body_style))
            
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
        
        i += 1
    
    # Add any remaining paragraph
    if current_paragraph:
        elements.append(Paragraph(' '.join(current_paragraph), body_style))
    
    return elements

def convert_markdown_to_pdf_with_images(markdown_file, output_pdf=None):
    """Convert Markdown file to PDF using reportlab with image support."""
    
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
    
    # Parse markdown to elements with images
    base_dir = markdown_path.parent
    elements = parse_markdown_to_elements_with_images(markdown_content, base_dir)
    
    try:
        # Build PDF
        doc.build(elements)
        print(f"✅ Successfully created PDF with images: {output_pdf}")
        return True
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to convert the CIAF Patent document with images."""
    
    # Set up file paths
    current_dir = Path.cwd()
    markdown_file = current_dir / "CIAF_Patent.md"
    pdf_file = current_dir / "CIAF_Patent_with_Images.pdf"
    
    if not markdown_file.exists():
        print(f"❌ Markdown file not found: {markdown_file}")
        print(f"Current directory: {current_dir}")
        print("Available files:")
        for file in current_dir.glob("*.md"):
            print(f"  - {file.name}")
        return False
    
    # Check for Images directory
    images_dir = current_dir / "Images"
    if images_dir.exists():
        print(f"📁 Found Images directory with files:")
        for img_file in images_dir.glob("*"):
            print(f"  - {img_file.name}")
    else:
        print(f"⚠️  No Images directory found at {images_dir}")
    
    # Convert to PDF
    success = convert_markdown_to_pdf_with_images(markdown_file, pdf_file)
    
    if success:
        print(f"\n🎉 Patent document successfully converted to PDF with images!")
        print(f"📄 Input:  {markdown_file}")
        print(f"📋 Output: {pdf_file}")
        print(f"📊 Size:   {pdf_file.stat().st_size / 1024:.1f} KB")
    
    return success

if __name__ == "__main__":
    main()
