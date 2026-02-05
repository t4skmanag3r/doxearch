from pathlib import Path

import pytest

from doxearch.doc_parser.parsers.docx_parser import DocxParser


@pytest.fixture
def docx_parser():
    """Fixture to provide a DocxParser instance."""
    return DocxParser()


@pytest.fixture
def example_docx_path():
    """Fixture to provide path to example DOCX file."""
    return Path(__file__).parent.parent / "example_documents" / "docx.docx"


def test_docx_parser_parse(docx_parser, example_docx_path):
    """Test that parse method works correctly."""
    result = docx_parser.parse(example_docx_path)
    assert isinstance(result, str)
    assert len(result) > 0
    assert result.strip()
