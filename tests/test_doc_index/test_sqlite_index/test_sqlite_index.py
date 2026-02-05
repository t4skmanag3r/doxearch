import os
import tempfile

import pytest
from sqlalchemy import inspect

from doxearch.doc_index.sqlite_index.sqlite_index import SQLiteIndex


@pytest.fixture
def temp_db_path():
    """Create a temporary database file path"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


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
