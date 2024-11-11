from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import Color, HexColor
from io import BytesIO

def create_header_footer():
    """Create color for header and footer"""
    return HexColor('#1e88e5')

def create_flashcard(front, back, width, height):
    """
    Create a flashcard table with front and back content.
    
    Args:
        front: Front side content
        back: Back side content
        width: Card width
        height: Card height
        
    Returns:
        Table object representing the flashcard
    """
    card_style = ParagraphStyle(
        'CardStyle',
        parent=getSampleStyleSheet()['Normal'],
        fontSize=10,
        leading=12,
        alignment=1
    )
    
    data = [
        [Paragraph(front, card_style)],
        [Paragraph(f"Answer: {back}", card_style)]
    ]
    
    card = Table(data, colWidths=[width], rowHeights=[height/2, height/2])
    card.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, 0), HexColor('#e3f2fd')),
        ('BACKGROUND', (0, 1), (0, 1), HexColor('#bbdefb')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return card

def generate_pdf(content):
    """Generate PDF with study materials."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    story = []
    
    # Define styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.black,
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.black,
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        spaceBefore=6,
        spaceAfter=6
    )
    
    sidebar_style = ParagraphStyle(
        'CustomSidebar',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.gray,
        backColor=HexColor('#f5f5f5')
    )
    
    # Create title page - simplified version
    header = Table(
        [[Paragraph(content['title'], title_style)]],  # Removed subtitle/date
        colWidths=[doc.width],
        style=[
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('PADDING', (0, 0), (-1, -1), 20),
        ]
    )
    story.append(header)
    story.append(PageBreak())
    
    # Format main content
    def format_markdown(text):
        """Convert markdown to formatted paragraphs"""
        lines = text.split('\n')
        formatted_lines = []
        current_paragraph = []
        in_list = False
        
        for line in lines:
            line = line.strip()
            
            # Handle headings
            if line.startswith('#'):
                if current_paragraph:
                    formatted_lines.append((' '.join(current_paragraph), 'body'))
                    current_paragraph = []
                heading_level = len(line.split()[0])  # Count the #s
                formatted_lines.append((line.lstrip('#').strip(), 'heading'))
                
            # Handle numbered lists
            elif line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                if current_paragraph:
                    formatted_lines.append((' '.join(current_paragraph), 'body'))
                    current_paragraph = []
                # Clean up the numbered list format
                text = line.split('.', 1)[1].strip()
                # Handle bold text in lists
                text = text.replace('**', '')
                formatted_lines.append((text, 'bullet'))
                in_list = True
                
            # Handle bullet points
            elif line.startswith(('- ', '• ')):
                if current_paragraph:
                    formatted_lines.append((' '.join(current_paragraph), 'body'))
                    current_paragraph = []
                # Clean up bullet points and bold text
                text = line.lstrip('- •').strip()
                text = text.replace('**', '')
                formatted_lines.append((text, 'bullet'))
                in_list = True
                
            # Handle empty lines
            elif not line:
                if current_paragraph:
                    formatted_lines.append((' '.join(current_paragraph), 'body'))
                    current_paragraph = []
                in_list = False
                
            # Regular text
            else:
                # Clean up bold markers
                line = line.replace('**', '')
                if in_list and line.startswith(' '):
                    # Continue list item on new line
                    formatted_lines[-1] = (formatted_lines[-1][0] + ' ' + line.strip(), 'bullet')
                else:
                    current_paragraph.append(line)
                    in_list = False
        
        if current_paragraph:
            formatted_lines.append((' '.join(current_paragraph), 'body'))
        
        return formatted_lines
    
    # Create main content
    main_content = []
    formatted_content = format_markdown(content['main_content'])
    for text, style_type in formatted_content:
        if style_type == 'heading':
            main_content.append(Paragraph(text, heading_style))
        elif style_type == 'bullet':
            main_content.append(Paragraph(text, body_style))
            main_content.append(Spacer(1, 3))
        else:
            main_content.append(Paragraph(text, body_style))
            main_content.append(Spacer(1, 6))
    
    # Create sidebar content
    sidebar_content = [[Paragraph("Key Concepts", heading_style)]]
    for concept in content['sidebar_content'].split('\n'):
        if concept.strip():
            sidebar_content.append([Paragraph("• " + concept.strip(), sidebar_style)])
    
    sidebar = Table(
        sidebar_content,
        colWidths=[doc.width * 0.25],
        style=[
            ('BACKGROUND', (0, 0), (-1, -1), HexColor('#f5f5f5')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 20),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 20),
        ]
    )
    
    # Add content page by page
    for i in range(0, len(main_content), 10):  # Process 10 elements at a time
        page_content = main_content[i:i+10]
        
        # Create two-column layout for each page
        content_table = Table(
            [[page_content, sidebar if i == 0 else '']],  # Only add sidebar on first content page
            colWidths=[doc.width * 0.7, doc.width * 0.25],
            style=[
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ]
        )
        story.append(content_table)
        story.append(PageBreak())
    
    # Add flashcards (4 per page)
    story.append(Paragraph("Flashcards", title_style))
    for i in range(0, len(content['flashcards']), 4):
        cards_page = content['flashcards'][i:i+4]
        for card in cards_page:
            card_table = create_flashcard(
                card['front'],
                card['back'],
                doc.width - 72,
                100
            )
            story.append(card_table)
            story.append(Spacer(1, 20))
        story.append(PageBreak())
    
    # Add exercises
    story.append(Paragraph("Exercises", title_style))
    for i, exercise in enumerate(content['exercises'], 1):
        story.append(Paragraph(f"Exercise {i}:", heading_style))
        story.append(Paragraph(exercise['question'], body_style))
        story.append(Spacer(1, 10))
        story.append(Paragraph("Solution:", heading_style))
        story.append(Paragraph(exercise['solution'], body_style))
        story.append(Spacer(1, 20))
    
    # Build PDF
    doc.build(story)
    return buffer.getvalue()