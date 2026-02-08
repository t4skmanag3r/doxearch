import os
import tempfile

import pytest
from sqlalchemy import inspect

from doxearch.doc_index.sqlite_index.sqlite_index import (
    CorpusStats,
    Document,
    DocumentFrequency,
    InvertedIndex,
    SQLiteIndex,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database file path"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def index(temp_db_path):
    """Create a fresh SQLiteIndex instance for each test."""
    return SQLiteIndex(db_path=temp_db_path)


def test_sqlite_index_initialization(temp_db_path):
    """Test that SQLiteIndex initializes and creates all required tables"""
    index = SQLiteIndex(db_path=temp_db_path)

    # Verify the database file was created
    assert os.path.exists(temp_db_path)

    # Verify all required tables exist
    inspector = inspect(index.engine)
    table_names = set(inspector.get_table_names())

    expected_tables = {
        "documents",
        "inverted_index",
        "document_frequency",
        "corpus_stats",
    }

    assert expected_tables.issubset(table_names)


def test_add_document(index):
    """Test that add_document correctly adds documents and updates all related tables."""
    # Prepare test data
    doc_id_1 = "doc_1"
    term_freq_1 = {"search": 3, "engine": 2, "document": 2}
    filepath_1 = "/path/to/doc1.pdf"

    doc_id_2 = "doc_2"
    term_freq_2 = {"search": 5, "indexing": 1, "retrieval": 1}
    filepath_2 = "/path/to/doc2.pdf"

    # Add first document
    index.add_document(doc_id_1, term_freq_1, filepath_1)

    session = index.Session()
    try:
        # Verify document record
        doc1 = session.query(Document).filter_by(doc_id=doc_id_1).first()
        assert doc1 is not None
        assert doc1.doc_id == doc_id_1
        assert doc1.file_path == filepath_1
        assert doc1.term_count == 7  # 3 + 2 + 2
        assert doc1.unique_terms == 3
        assert doc1.last_indexed > 0

        # Verify inverted index entries
        inverted_entries_1 = (
            session.query(InvertedIndex).filter_by(doc_id=doc_id_1).all()
        )
        assert len(inverted_entries_1) == 3
        for entry in inverted_entries_1:
            assert entry.term_frequency == term_freq_1[entry.term]

        # Verify document frequency entries
        search_df = session.query(DocumentFrequency).filter_by(term="search").first()
        assert search_df.doc_count == 1
        assert search_df.total_frequency == 3

        # Verify corpus statistics
        corpus_stat = (
            session.query(CorpusStats).filter_by(stat_name="total_documents").first()
        )
        assert corpus_stat.stat_value == 1
        assert index.get_document_count() == 1
    finally:
        session.close()

    # Add second document with overlapping term
    index.add_document(doc_id_2, term_freq_2, filepath_2)

    session = index.Session()
    try:
        # Verify second document record
        doc2 = session.query(Document).filter_by(doc_id=doc_id_2).first()
        assert doc2 is not None
        assert doc2.term_count == 7  # 5 + 1 + 1

        # Verify document frequency updated for overlapping term "search"
        search_df = session.query(DocumentFrequency).filter_by(term="search").first()
        assert search_df.doc_count == 2
        assert search_df.total_frequency == 8  # 3 + 5

        # Verify document frequency for new terms
        indexing_df = (
            session.query(DocumentFrequency).filter_by(term="indexing").first()
        )
        assert indexing_df.doc_count == 1
        assert indexing_df.total_frequency == 1

        # Verify corpus statistics updated
        corpus_stat = (
            session.query(CorpusStats).filter_by(stat_name="total_documents").first()
        )
        assert corpus_stat.stat_value == 2
        assert index.get_document_count() == 2
    finally:
        session.close()

    # Test duplicate document ID raises error
    with pytest.raises(ValueError, match="already exists in the index"):
        index.add_document(doc_id_1, term_freq_1, filepath_1)

    # Verify document count unchanged after failed duplicate
    assert index.get_document_count() == 2
