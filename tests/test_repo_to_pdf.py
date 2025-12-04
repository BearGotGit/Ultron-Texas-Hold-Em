"""Tests for the repo_to_pdf utility."""

import os
import tempfile
from pathlib import Path

import pytest


class TestRepoToPdfHelpers:
    """Test helper functions in repo_to_pdf module."""

    def test_should_include_python_file(self):
        """Test that Python files are included."""
        from tools.repo_to_pdf import should_include_file
        
        path = Path("/some/repo/test.py")
        assert should_include_file(path) is True

    def test_should_include_markdown_file(self):
        """Test that Markdown files are included."""
        from tools.repo_to_pdf import should_include_file
        
        path = Path("/some/repo/README.md")
        assert should_include_file(path) is True

    def test_should_exclude_pycache(self):
        """Test that __pycache__ directories are excluded."""
        from tools.repo_to_pdf import should_include_file
        
        path = Path("/some/repo/__pycache__/test.pyc")
        assert should_include_file(path) is False

    def test_should_exclude_git(self):
        """Test that .git directories are excluded."""
        from tools.repo_to_pdf import should_include_file
        
        path = Path("/some/repo/.git/config")
        assert should_include_file(path) is False

    def test_should_include_gitignore(self):
        """Test that .gitignore files are included."""
        from tools.repo_to_pdf import should_include_file
        
        path = Path("/some/repo/.gitignore")
        assert should_include_file(path) is True

    def test_escape_xml(self):
        """Test XML escaping for special characters."""
        from tools.repo_to_pdf import escape_xml
        
        assert escape_xml("Hello & World") == "Hello &amp; World"
        assert escape_xml("<tag>") == "&lt;tag&gt;"
        assert escape_xml("a < b > c") == "a &lt; b &gt; c"

    def test_read_file_content(self):
        """Test reading file content."""
        from tools.repo_to_pdf import read_file_content
        
        # Create a temporary file to test
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content\nLine 2")
            temp_path = f.name
        
        try:
            content = read_file_content(Path(temp_path))
            assert "Test content" in content
            assert "Line 2" in content
        finally:
            os.unlink(temp_path)

    def test_is_binary_file_text(self):
        """Test that text files are not identified as binary."""
        from tools.repo_to_pdf import is_binary_file
        
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a text file\n")
            temp_path = f.name
        
        try:
            assert is_binary_file(Path(temp_path)) is False
        finally:
            os.unlink(temp_path)

    def test_is_binary_file_binary(self):
        """Test that binary files are identified as binary."""
        from tools.repo_to_pdf import is_binary_file
        
        # Create a temporary binary file with null bytes
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False) as f:
            f.write(b'\x00\x01\x02\x03')
            temp_path = f.name
        
        try:
            assert is_binary_file(Path(temp_path)) is True
        finally:
            os.unlink(temp_path)


class TestRepoToPdfIntegration:
    """Integration tests for repo_to_pdf."""

    def test_get_repo_files(self):
        """Test getting files from a test directory."""
        from tools.repo_to_pdf import get_repo_files
        
        # Use the tools directory as a test
        tools_path = Path(__file__).parent.parent / "tools"
        files = list(get_repo_files(tools_path))
        
        # Should find at least some Python files
        py_files = [f for f in files if f.suffix == '.py']
        assert len(py_files) > 0
        
        # repo_to_pdf.py should be in the list
        file_names = [f.name for f in files]
        assert "repo_to_pdf.py" in file_names

    def test_create_pdf(self):
        """Test creating a PDF from a test directory."""
        from tools.repo_to_pdf import create_pdf
        
        # Create a temporary directory with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            (Path(temp_dir) / "test.py").write_text("print('hello')\n")
            (Path(temp_dir) / "README.md").write_text("# Test\n\nThis is a test.\n")
            
            # Create output PDF
            output_pdf = Path(temp_dir) / "output.pdf"
            
            create_pdf(Path(temp_dir), str(output_pdf))
            
            # Check that PDF was created
            assert output_pdf.exists()
            assert output_pdf.stat().st_size > 0
            
            # Verify it's a valid PDF
            with open(output_pdf, 'rb') as f:
                header = f.read(5)
                assert header == b'%PDF-'
