from pathlib import Path

import pytest

from doxearch.doc_parser.parsers.pdf_parser import PDFParser


@pytest.fixture
def pdf_parser():
    """Fixture to provide a PDFParser instance."""
    return PDFParser()


@pytest.fixture
def example_pdf_path():
    """Fixture to provide the path to the example PDF."""
    return Path(__file__).parent.parent / "example_documents" / "pdf.pdf"


def test_pdf_parser_parse(pdf_parser, example_pdf_path):
    """Test that parse method works correctly."""
    result = pdf_parser.parse(example_pdf_path)
    assert isinstance(result, str)
    assert len(result) > 0
    assert result.strip()
