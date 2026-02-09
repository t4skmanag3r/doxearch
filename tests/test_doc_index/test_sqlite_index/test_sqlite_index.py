import os
import tempfile

import pytest
from sqlalchemy import inspect

from doxearch.doc_index.sqlite_index.exceptions import (
    DocumentExistsError,
    DocumentNotFoundError,
)
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
    with pytest.raises(DocumentExistsError, match="already exists in the index"):
        index.add_document(doc_id_1, term_freq_1, filepath_1)

    # Verify document count unchanged after failed duplicate
    assert index.get_document_count() == 2


def test_remove_document(index):
    """Test that remove_document correctly removes documents and updates all related tables."""
    # Prepare test data - add three documents
    doc_id_1 = "doc_1"
    term_freq_1 = {"search": 3, "engine": 2, "document": 2}
    filepath_1 = "/path/to/doc1.pdf"

    doc_id_2 = "doc_2"
    term_freq_2 = {"search": 5, "indexing": 1, "retrieval": 1}
    filepath_2 = "/path/to/doc2.pdf"

    doc_id_3 = "doc_3"
    term_freq_3 = {"engine": 1, "indexing": 2, "database": 3}
    filepath_3 = "/path/to/doc3.pdf"

    # Add all documents
    index.add_document(doc_id_1, term_freq_1, filepath_1)
    index.add_document(doc_id_2, term_freq_2, filepath_2)
    index.add_document(doc_id_3, term_freq_3, filepath_3)

    # Verify initial state
    assert index.get_document_count() == 3

    session = index.Session()
    try:
        # Verify initial document frequencies
        search_df = session.query(DocumentFrequency).filter_by(term="search").first()
        assert search_df.doc_count == 2
        assert search_df.total_frequency == 8  # 3 + 5

        engine_df = session.query(DocumentFrequency).filter_by(term="engine").first()
        assert engine_df.doc_count == 2
        assert engine_df.total_frequency == 3  # 2 + 1

        indexing_df = (
            session.query(DocumentFrequency).filter_by(term="indexing").first()
        )
        assert indexing_df.doc_count == 2
        assert indexing_df.total_frequency == 3  # 1 + 2
    finally:
        session.close()

    # Remove doc_1 (contains: search, engine, document)
    index.remove_document(doc_id_1)

    session = index.Session()
    try:
        # Verify document was removed
        doc1 = session.query(Document).filter_by(doc_id=doc_id_1).first()
        assert doc1 is None
        assert index.get_document_count() == 2

        # Verify inverted index entries for doc_1 were removed
        inverted_entries_1 = (
            session.query(InvertedIndex).filter_by(doc_id=doc_id_1).all()
        )
        assert len(inverted_entries_1) == 0

        # Verify document frequencies updated for "search" (only in doc_2 now)
        search_df = session.query(DocumentFrequency).filter_by(term="search").first()
        assert search_df.doc_count == 1
        assert search_df.total_frequency == 5

        # Verify document frequencies updated for "engine" (only in doc_3 now)
        engine_df = session.query(DocumentFrequency).filter_by(term="engine").first()
        assert engine_df.doc_count == 1
        assert engine_df.total_frequency == 1

        # Verify "document" term was removed (only in doc_1)
        document_df = (
            session.query(DocumentFrequency).filter_by(term="document").first()
        )
        assert document_df is None

        # Verify corpus statistics updated
        corpus_stat = (
            session.query(CorpusStats).filter_by(stat_name="total_documents").first()
        )
        assert corpus_stat.stat_value == 2
    finally:
        session.close()

    # Remove doc_2 (contains: search, indexing, retrieval)
    index.remove_document(doc_id_2)

    session = index.Session()
    try:
        # Verify document count
        assert index.get_document_count() == 1

        # Verify "search" term was removed (was only in doc_1 and doc_2)
        search_df = session.query(DocumentFrequency).filter_by(term="search").first()
        assert search_df is None

        # Verify "retrieval" term was removed (only in doc_2)
        retrieval_df = (
            session.query(DocumentFrequency).filter_by(term="retrieval").first()
        )
        assert retrieval_df is None

        # Verify "indexing" still exists (also in doc_3)
        indexing_df = (
            session.query(DocumentFrequency).filter_by(term="indexing").first()
        )
        assert indexing_df.doc_count == 1
        assert indexing_df.total_frequency == 2

        # Verify corpus statistics
        corpus_stat = (
            session.query(CorpusStats).filter_by(stat_name="total_documents").first()
        )
        assert corpus_stat.stat_value == 1
    finally:
        session.close()

    # Remove last document (doc_3)
    index.remove_document(doc_id_3)

    session = index.Session()
    try:
        # Verify all documents removed
        assert index.get_document_count() == 0

        # Verify all document frequency entries removed
        all_df = session.query(DocumentFrequency).all()
        assert len(all_df) == 0

        # Verify all inverted index entries removed
        all_inverted = session.query(InvertedIndex).all()
        assert len(all_inverted) == 0

        # Verify corpus statistics removed (no documents left)
        corpus_stat = (
            session.query(CorpusStats).filter_by(stat_name="total_documents").first()
        )
        assert corpus_stat is None
    finally:
        session.close()

    # Test removing non-existent document raises error
    with pytest.raises(DocumentNotFoundError, match="does not exist in the index"):
        index.remove_document("non_existent_doc")


def test_remove_document_with_shared_terms(index):
    """Test removing documents when multiple documents share the same terms."""
    # Add documents with overlapping terms
    doc_id_1 = "doc_1"
    term_freq_1 = {"python": 5, "programming": 3, "language": 2}
    filepath_1 = "/path/to/doc1.pdf"

    doc_id_2 = "doc_2"
    term_freq_2 = {"python": 8, "programming": 4, "code": 6}
    filepath_2 = "/path/to/doc2.pdf"

    doc_id_3 = "doc_3"
    term_freq_3 = {"python": 3, "language": 5, "syntax": 2}
    filepath_3 = "/path/to/doc3.pdf"

    index.add_document(doc_id_1, term_freq_1, filepath_1)
    index.add_document(doc_id_2, term_freq_2, filepath_2)
    index.add_document(doc_id_3, term_freq_3, filepath_3)

    # Remove middle document
    index.remove_document(doc_id_2)

    session = index.Session()
    try:
        # Verify "python" still exists (in doc_1 and doc_3)
        python_df = session.query(DocumentFrequency).filter_by(term="python").first()
        assert python_df.doc_count == 2
        assert python_df.total_frequency == 8  # 5 + 3

        # Verify "programming" still exists (only in doc_1 now)
        programming_df = (
            session.query(DocumentFrequency).filter_by(term="programming").first()
        )
        assert programming_df.doc_count == 1
        assert programming_df.total_frequency == 3

        # Verify "code" was removed (only in doc_2)
        code_df = session.query(DocumentFrequency).filter_by(term="code").first()
        assert code_df is None

        # Verify "language" still exists (in doc_1 and doc_3)
        language_df = (
            session.query(DocumentFrequency).filter_by(term="language").first()
        )
        assert language_df.doc_count == 2
        assert language_df.total_frequency == 7  # 2 + 5

        # Verify document count
        assert index.get_document_count() == 2
    finally:
        session.close()


def test_update_document(index):
    """Test that update_document correctly updates documents and all related statistics."""
    # Add initial documents
    doc_id_1 = "doc_1"
    term_freq_1 = {"python": 5, "programming": 3, "language": 2}
    filepath_1 = "/path/to/doc1.pdf"

    doc_id_2 = "doc_2"
    term_freq_2 = {"python": 8, "code": 4}
    filepath_2 = "/path/to/doc2.pdf"

    index.add_document(doc_id_1, term_freq_1, filepath_1)
    index.add_document(doc_id_2, term_freq_2, filepath_2)

    assert index.get_document_count() == 2

    session = index.Session()
    try:
        # Verify initial state
        python_df = session.query(DocumentFrequency).filter_by(term="python").first()
        assert python_df.doc_count == 2
        assert python_df.total_frequency == 13  # 5 + 8

        programming_df = (
            session.query(DocumentFrequency).filter_by(term="programming").first()
        )
        assert programming_df.doc_count == 1
        assert programming_df.total_frequency == 3
    finally:
        session.close()

    # Update doc_1 with completely different terms
    new_term_freq_1 = {"java": 7, "code": 3, "object": 5}
    new_filepath_1 = "/path/to/updated_doc1.pdf"

    index.update_document(doc_id_1, new_term_freq_1, new_filepath_1)

    session = index.Session()
    try:
        # Verify document count unchanged
        assert index.get_document_count() == 2

        # Verify document record updated
        doc1 = session.query(Document).filter_by(doc_id=doc_id_1).first()
        assert doc1 is not None
        assert doc1.file_path == new_filepath_1
        assert doc1.term_count == 15  # 7 + 3 + 5
        assert doc1.unique_terms == 3

        # Verify old terms removed/updated
        # "python" should only have doc_2 now
        python_df = session.query(DocumentFrequency).filter_by(term="python").first()
        assert python_df.doc_count == 1
        assert python_df.total_frequency == 8

        # "programming" should be completely removed (was only in doc_1)
        programming_df = (
            session.query(DocumentFrequency).filter_by(term="programming").first()
        )
        assert programming_df is None

        # "language" should be completely removed (was only in doc_1)
        language_df = (
            session.query(DocumentFrequency).filter_by(term="language").first()
        )
        assert language_df is None

        # Verify new terms added
        # "java" should be new
        java_df = session.query(DocumentFrequency).filter_by(term="java").first()
        assert java_df.doc_count == 1
        assert java_df.total_frequency == 7

        # "code" should now be in both documents
        code_df = session.query(DocumentFrequency).filter_by(term="code").first()
        assert code_df.doc_count == 2
        assert code_df.total_frequency == 7  # 3 + 4

        # "object" should be new
        object_df = session.query(DocumentFrequency).filter_by(term="object").first()
        assert object_df.doc_count == 1
        assert object_df.total_frequency == 5

        # Verify inverted index updated for doc_1
        inverted_entries = session.query(InvertedIndex).filter_by(doc_id=doc_id_1).all()
        assert len(inverted_entries) == 3
        inverted_terms = {
            entry.term: entry.term_frequency for entry in inverted_entries
        }
        assert inverted_terms == new_term_freq_1

        # Verify corpus statistics unchanged
        corpus_stat = (
            session.query(CorpusStats).filter_by(stat_name="total_documents").first()
        )
        assert corpus_stat.stat_value == 2
    finally:
        session.close()

    # Update doc_1 again with some overlapping terms
    updated_term_freq_1 = {"java": 10, "python": 2, "testing": 4}
    index.update_document(doc_id_1, updated_term_freq_1, new_filepath_1)

    session = index.Session()
    try:
        # Verify "java" updated
        java_df = session.query(DocumentFrequency).filter_by(term="java").first()
        assert java_df.doc_count == 1
        assert java_df.total_frequency == 10

        # Verify "code" removed from doc_1 (only in doc_2 now)
        code_df = session.query(DocumentFrequency).filter_by(term="code").first()
        assert code_df.doc_count == 1
        assert code_df.total_frequency == 4

        # Verify "object" removed (was only in doc_1)
        object_df = session.query(DocumentFrequency).filter_by(term="object").first()
        assert object_df is None

        # Verify "python" now in both documents again
        python_df = session.query(DocumentFrequency).filter_by(term="python").first()
        assert python_df.doc_count == 2
        assert python_df.total_frequency == 10  # 2 + 8

        # Verify "testing" is new
        testing_df = session.query(DocumentFrequency).filter_by(term="testing").first()
        assert testing_df.doc_count == 1
        assert testing_df.total_frequency == 4
    finally:
        session.close()

    # Test updating non-existent document raises error
    with pytest.raises(DocumentNotFoundError, match="does not exist in the index"):
        index.update_document("non_existent_doc", {"term": 1}, "/path/to/fake.pdf")
