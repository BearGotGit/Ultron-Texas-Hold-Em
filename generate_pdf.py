"""
Generate a PDF containing all repository source files.

This script collects all relevant source files from the repository
and generates a well-formatted PDF document with syntax highlighting.

Usage:
    python generate_pdf.py
    python generate_pdf.py --output my_output.pdf
    python generate_pdf.py --exclude tests --exclude data
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

from fpdf import FPDF


# File extensions to include
INCLUDE_EXTENSIONS = {
    '.py',      # Python files
    '.md',      # Markdown files
    '.txt',     # Text files
    '.ini',     # Config files
    '.yml',     # YAML files
    '.yaml',    # YAML files
    '.json',    # JSON files
    '.sh',      # Shell scripts
    '.cfg',     # Config files
    '.toml',    # TOML files
}

# Directories to exclude by default
DEFAULT_EXCLUDE_DIRS = {
    '.git',
    '__pycache__',
    '.venv',
    'venv',
    '.env',
    'node_modules',
    '.pytest_cache',
    '.mypy_cache',
    'checkpoints',
    'runs',
    '.github',
}


class RepositoryPDF(FPDF):
    """Custom PDF class for repository documentation."""

    def __init__(self, repo_name: str):
        super().__init__()
        self.repo_name = repo_name
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        """Add header to each page."""
        self.set_font('Courier', 'B', 10)
        self.cell(0, 10, self.repo_name, border=0, align='L')
        self.ln(5)

    def footer(self):
        """Add footer with page number to each page."""
        self.set_y(-15)
        self.set_font('Courier', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', border=0, align='C')

    def add_title_page(self, repo_path: str, file_count: int):
        """Add a title page with repository information."""
        self.add_page()
        self.set_font('Courier', 'B', 24)
        self.ln(40)
        self.cell(0, 20, self.repo_name, align='C')
        self.ln(30)

        self.set_font('Courier', '', 12)
        self.cell(0, 10, 'Repository Source Code', align='C')
        self.ln(20)

        self.set_font('Courier', '', 10)
        self.cell(0, 8, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', align='C')
        self.ln(8)
        self.cell(0, 8, f'Total Files: {file_count}', align='C')
        self.ln(8)
        self.cell(0, 8, f'Source: {repo_path}', align='C')

    def add_table_of_contents(self, files: list):
        """Add a table of contents page."""
        self.add_page()
        self.set_font('Courier', 'B', 16)
        self.cell(0, 15, 'Table of Contents', align='L')
        self.ln(15)

        self.set_font('Courier', '', 9)
        for i, file_path in enumerate(files, 1):
            # Truncate long paths
            display_path = file_path
            if len(display_path) > 70:
                display_path = '...' + display_path[-67:]
            self.cell(0, 6, f'{i:3}. {display_path}', align='L')
            self.ln(6)

            # Add new page if needed
            if self.get_y() > 270:
                self.add_page()
                self.set_font('Courier', 'B', 16)
                self.cell(0, 15, 'Table of Contents (continued)', align='L')
                self.ln(15)
                self.set_font('Courier', '', 9)

    def add_file_content(self, file_path: str, content: str, file_number: int):
        """Add a file's content to the PDF."""
        self.add_page()

        # File header
        self.set_font('Courier', 'B', 12)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, f'File {file_number}: {file_path}', border=1, fill=True)
        self.ln(12)

        # File content
        self.set_font('Courier', '', 8)

        # Split content into lines and add line numbers
        lines = content.split('\n')
        line_num_width = len(str(len(lines)))

        for i, line in enumerate(lines, 1):
            # Format line number
            line_num = str(i).rjust(line_num_width)

            # Handle long lines by wrapping
            # Replace tabs with spaces
            line = line.replace('\t', '    ')

            # Calculate max chars per line (accounting for line number)
            max_chars = 100

            if len(line) > max_chars:
                # Wrap long lines
                chunks = [line[j:j+max_chars] for j in range(0, len(line), max_chars)]
                for chunk_idx, chunk in enumerate(chunks):
                    if chunk_idx == 0:
                        self._add_code_line(line_num, chunk)
                    else:
                        # Continuation line (no line number)
                        self._add_code_line(' ' * line_num_width, '  ' + chunk)
            else:
                self._add_code_line(line_num, line)

    def _add_code_line(self, line_num: str, text: str):
        """Add a single line of code with line number."""
        # Sanitize text for PDF (remove characters that cause issues)
        safe_text = ''
        for char in text:
            if ord(char) < 128:  # Only ASCII characters
                safe_text += char
            else:
                # Replace non-ASCII with a placeholder
                safe_text += '?'

        try:
            self.cell(0, 4, f'{line_num} | {safe_text}')
            self.ln(4)
        except Exception:
            # If line still causes issues, skip it
            self.cell(0, 4, f'{line_num} | [Content could not be displayed]')
            self.ln(4)


def collect_files(
    repo_path: str,
    exclude_dirs: set = None,
    include_extensions: set = None
) -> list:
    """
    Collect all relevant files from the repository.

    Args:
        repo_path: Path to the repository root
        exclude_dirs: Set of directory names to exclude
        include_extensions: Set of file extensions to include

    Returns:
        List of relative file paths
    """
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS
    if include_extensions is None:
        include_extensions = INCLUDE_EXTENSIONS

    files = []
    repo_path = Path(repo_path)

    for root, dirs, filenames in os.walk(repo_path):
        # Remove excluded directories from dirs to prevent walking into them
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for filename in filenames:
            file_path = Path(root) / filename
            relative_path = file_path.relative_to(repo_path)

            # Check if file extension is in include list
            if file_path.suffix.lower() in include_extensions:
                files.append(str(relative_path))

    # Sort files for consistent ordering
    files.sort()
    return files


def read_file_content(repo_path: str, file_path: str) -> str:
    """
    Read the content of a file.

    Args:
        repo_path: Path to the repository root
        file_path: Relative path to the file

    Returns:
        File content as string
    """
    full_path = Path(repo_path) / file_path
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with latin-1 encoding as fallback
        with open(full_path, 'r', encoding='latin-1') as f:
            return f.read()
    except Exception as e:
        return f'[Error reading file: {e}]'


def generate_pdf(
    repo_path: str,
    output_path: str = None,
    exclude_dirs: list = None,
    include_extensions: list = None
) -> str:
    """
    Generate a PDF containing all repository source files.

    Args:
        repo_path: Path to the repository root
        output_path: Path for the output PDF file
        exclude_dirs: Additional directories to exclude
        include_extensions: Additional file extensions to include

    Returns:
        Path to the generated PDF file
    """
    repo_path = os.path.abspath(repo_path)
    repo_name = os.path.basename(repo_path)

    # Set default output path
    if output_path is None:
        output_path = os.path.join(repo_path, f'{repo_name}_source_code.pdf')

    # Merge default exclude dirs with user-specified ones
    exclude_set = DEFAULT_EXCLUDE_DIRS.copy()
    if exclude_dirs:
        exclude_set.update(exclude_dirs)

    # Merge default extensions with user-specified ones
    include_set = INCLUDE_EXTENSIONS.copy()
    if include_extensions:
        include_set.update(f'.{ext.lstrip(".")}' for ext in include_extensions)

    # Collect files
    print(f'Scanning repository: {repo_path}')
    files = collect_files(repo_path, exclude_set, include_set)
    print(f'Found {len(files)} files to include')

    if not files:
        print('No files found to include in PDF.')
        return None

    # Create PDF
    pdf = RepositoryPDF(repo_name)

    # Add title page
    pdf.add_title_page(repo_path, len(files))

    # Add table of contents
    pdf.add_table_of_contents(files)

    # Add each file's content
    for i, file_path in enumerate(files, 1):
        print(f'Processing ({i}/{len(files)}): {file_path}')
        content = read_file_content(repo_path, file_path)
        pdf.add_file_content(file_path, content, i)

    # Save PDF
    print(f'\nSaving PDF to: {output_path}')
    pdf.output(output_path)
    print('PDF generated successfully!')

    return output_path


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate a PDF containing all repository source files.'
    )
    parser.add_argument(
        '--repo',
        type=str,
        default='.',
        help='Path to the repository (default: current directory)'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Output PDF file path (default: <repo_name>_source_code.pdf)'
    )
    parser.add_argument(
        '--exclude',
        '-e',
        action='append',
        default=[],
        help='Additional directories to exclude (can be used multiple times)'
    )
    parser.add_argument(
        '--include-ext',
        action='append',
        default=[],
        help='Additional file extensions to include (can be used multiple times)'
    )

    args = parser.parse_args()

    output_path = generate_pdf(
        repo_path=args.repo,
        output_path=args.output,
        exclude_dirs=args.exclude if args.exclude else None,
        include_extensions=args.include_ext if args.include_ext else None
    )

    if output_path:
        print(f'\nGenerated PDF: {output_path}')
        print(f'File size: {os.path.getsize(output_path) / 1024:.1f} KB')


if __name__ == '__main__':
    main()
