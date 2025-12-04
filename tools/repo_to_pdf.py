"""
Repository to PDF Converter

Converts all source code and text files in a repository to a single PDF document.
Files are organized with headers showing file paths, and code is formatted with
proper line numbers for easy reference.

Usage:
    python tools/repo_to_pdf.py [--output OUTPUT_FILE] [--path REPO_PATH]

Examples:
    # Convert current repository to PDF
    python tools/repo_to_pdf.py

    # Specify output file name
    python tools/repo_to_pdf.py --output my_repo.pdf

    # Convert a specific directory
    python tools/repo_to_pdf.py --path /path/to/repo --output repo.pdf
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Generator

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    PageBreak,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT


# File extensions to include in the PDF
INCLUDE_EXTENSIONS = {
    '.py', '.md', '.txt', '.rst', '.json', '.yaml', '.yml',
    '.ini', '.cfg', '.toml', '.sh', '.bash', '.html', '.css',
    '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp', '.go',
    '.rb', '.php', '.sql', '.gitignore', '.env.example'
}

# Files without extensions to include
INCLUDE_FILES = {
    'Makefile', 'Dockerfile', 'README', 'LICENSE', 'CHANGELOG',
    'requirements.txt', 'requirement.txt', 'setup.py', 'pyproject.toml',
    '.gitignore', '.env.example'
}

# Directories to exclude
EXCLUDE_DIRS = {
    '.git', '__pycache__', 'node_modules', '.venv', 'venv',
    'env', '.env', 'dist', 'build', '.pytest_cache',
    '.mypy_cache', '.tox', 'htmlcov', '.coverage',
    'checkpoints', 'runs', '.eggs', '*.egg-info'
}

# Files to exclude
EXCLUDE_FILES = {
    '.DS_Store', 'Thumbs.db', '*.pyc', '*.pyo', '*.pyd',
    '*.so', '*.dll', '*.exe', '*.bin', '*.pkl', '*.pickle',
    '*.h5', '*.hdf5', '*.pt', '*.pth', '*.ckpt', '*.csv'
}


def should_include_file(file_path: Path) -> bool:
    """Check if a file should be included in the PDF."""
    # Check if any parent directory should be excluded
    for parent in file_path.parents:
        if parent.name in EXCLUDE_DIRS:
            return False
    
    # Check file extension
    extension = file_path.suffix.lower()
    if extension in INCLUDE_EXTENSIONS:
        return True
    
    # Check file name
    if file_path.name in INCLUDE_FILES:
        return True
    
    return False


def is_binary_file(file_path: Path) -> bool:
    """Check if a file is binary by reading a sample."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(8192)
            # Check for null bytes (common indicator of binary files)
            if b'\x00' in chunk:
                return True
        return False
    except (IOError, OSError):
        return True


def get_repo_files(repo_path: Path) -> Generator[Path, None, None]:
    """Get all files in the repository that should be included in the PDF."""
    for root, dirs, files in os.walk(repo_path):
        # Remove excluded directories from dirs to prevent walking into them
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file_name in sorted(files):
            file_path = Path(root) / file_name
            if should_include_file(file_path) and not is_binary_file(file_path):
                yield file_path


def read_file_content(file_path: Path) -> str:
    """Read the content of a file, handling encoding errors."""
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, IOError):
            continue
    return "[Error: Could not read file content]"


def escape_xml(text: str) -> str:
    """Escape XML special characters for ReportLab."""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;'))


def create_pdf(repo_path: Path, output_file: str) -> None:
    """Create a PDF document from all repository files."""
    # Set up the document
    doc = SimpleDocTemplate(
        output_file,
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    # Define styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    file_header_style = ParagraphStyle(
        'FileHeader',
        parent=styles['Heading2'],
        fontSize=12,
        spaceBefore=18,
        spaceAfter=6,
        textColor=colors.darkgreen,
        backColor=colors.lightgrey
    )
    
    code_style = ParagraphStyle(
        'Code',
        fontName='Courier',
        fontSize=7,
        leading=9,
        alignment=TA_LEFT,
        leftIndent=0,
        rightIndent=0
    )
    
    info_style = ParagraphStyle(
        'Info',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=12,
        textColor=colors.grey
    )
    
    # Build document content
    story = []
    
    # Title page
    repo_name = repo_path.name
    story.append(Paragraph(f"Repository: {escape_xml(repo_name)}", title_style))
    story.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        info_style
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Table of contents header
    story.append(Paragraph("Table of Contents", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    # Collect all files first for TOC
    files_list = list(get_repo_files(repo_path))
    
    # Add table of contents
    for idx, file_path in enumerate(files_list, 1):
        relative_path = file_path.relative_to(repo_path)
        story.append(Paragraph(
            f"{idx}. {escape_xml(str(relative_path))}",
            styles['Normal']
        ))
    
    story.append(PageBreak())
    
    # Add each file
    for idx, file_path in enumerate(files_list, 1):
        relative_path = file_path.relative_to(repo_path)
        
        # File header
        story.append(Paragraph(
            f"File {idx}: {escape_xml(str(relative_path))}",
            file_header_style
        ))
        
        # File content
        content = read_file_content(file_path)
        
        # Add line numbers and format content
        lines = content.split('\n')
        numbered_lines = []
        for line_num, line in enumerate(lines, 1):
            # Escape XML characters and format with line number
            escaped_line = escape_xml(line)
            # Replace tabs with spaces for consistent display
            escaped_line = escaped_line.replace('\t', '    ')
            numbered_lines.append(f"{line_num:4d} | {escaped_line}")
        
        formatted_content = '\n'.join(numbered_lines)
        
        # Use Preformatted for code content to preserve formatting
        story.append(Preformatted(formatted_content, code_style))
        
        # Add page break after each file (except the last one)
        if idx < len(files_list):
            story.append(PageBreak())
    
    # Build the PDF
    doc.build(story)
    
    print(f"PDF created successfully: {output_file}")
    print(f"Total files included: {len(files_list)}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Convert repository files to a single PDF document.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--output', '-o',
        default='repository_code.pdf',
        help='Output PDF file name (default: repository_code.pdf)'
    )
    
    parser.add_argument(
        '--path', '-p',
        default='.',
        help='Path to the repository (default: current directory)'
    )
    
    args = parser.parse_args()
    
    repo_path = Path(args.path).resolve()
    
    if not repo_path.exists():
        print(f"Error: Path does not exist: {repo_path}")
        return 1
    
    if not repo_path.is_dir():
        print(f"Error: Path is not a directory: {repo_path}")
        return 1
    
    output_file = args.output
    if not output_file.endswith('.pdf'):
        output_file += '.pdf'
    
    print(f"Converting repository: {repo_path}")
    print(f"Output file: {output_file}")
    print()
    
    create_pdf(repo_path, output_file)
    
    return 0


if __name__ == '__main__':
    exit(main())
