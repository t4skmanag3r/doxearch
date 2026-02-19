import os
from pathlib import Path

import pytest

from doxearch.context_manager import DirectoryContextManager
from doxearch.exceptions import (
    DirectoryAlreadyIndexedError,
    DirectoryNotFoundError,
    InvalidDirectoryPathError,
)


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path for testing."""
    db_path = tmp_path / "test_context_manager.db"
    yield str(db_path)
    if db_path.exists():
        os.remove(db_path)


@pytest.fixture
def context_manager(temp_db_path):
    """Create a fresh ContextManager instance for each test."""
    return DirectoryContextManager(db_path=temp_db_path)


def test_add_indexed_directory(context_manager, tmp_path):
    """Test adding a new indexed directory"""
    directory_path = "/home/user/documents"
    db_path = str(tmp_path / "test_index.db")
    model_name = "en_core_web_sm"
    model_version = "3.7.0"

    context_manager.add_indexed_directory(
        directory_path,
        db_path,
        model_name,
        model_version,
        lemmatization_enabled=True,
        stemming_enabled=False,
    )

    # Verify directory was added and is active
    active_dir = context_manager.get_active_directory()
    assert active_dir is not None
    assert active_dir["directory_path"] == directory_path
    assert active_dir["db_path"] == db_path
    assert active_dir["tokenizer_model_name"] == model_name
    assert active_dir["tokenizer_model_version"] == model_version
    assert active_dir["lemmatization_enabled"] is True
    assert active_dir["stemming_enabled"] is False


def test_add_indexed_directory_without_version(context_manager, tmp_path):
    """Test adding a directory without specifying model version"""
    directory_path = "/home/user/documents"
    db_path = str(tmp_path / "test_index.db")
    model_name = "en_core_web_sm"

    context_manager.add_indexed_directory(
        directory_path,
        db_path,
        model_name,
        lemmatization_enabled=True,
        stemming_enabled=False,
    )

    active_dir = context_manager.get_active_directory()
    assert active_dir["tokenizer_model_version"] is None
    assert active_dir["lemmatization_enabled"] is True
    assert active_dir["stemming_enabled"] is False


def test_add_indexed_directory_with_stemming(context_manager, tmp_path):
    """Test adding a directory with stemming enabled"""
    directory_path = "/home/user/documents"
    db_path = str(tmp_path / "test_index.db")
    model_name = "lt_core_news_sm"

    context_manager.add_indexed_directory(
        directory_path,
        db_path,
        model_name,
        lemmatization_enabled=False,
        stemming_enabled=True,
    )

    active_dir = context_manager.get_active_directory()
    assert active_dir["lemmatization_enabled"] is False
    assert active_dir["stemming_enabled"] is True


def test_add_duplicate_directory_raises_error(context_manager, tmp_path):
    """Test that adding a duplicate directory raises an error"""
    directory_path = "/home/user/documents"
    db_path = str(tmp_path / "test_index.db")
    model_name = "en_core_web_sm"

    context_manager.add_indexed_directory(
        directory_path,
        db_path,
        model_name,
        lemmatization_enabled=True,
        stemming_enabled=False,
    )

    with pytest.raises(DirectoryAlreadyIndexedError, match="already indexed"):
        context_manager.add_indexed_directory(
            directory_path,
            db_path,
            model_name,
            lemmatization_enabled=True,
            stemming_enabled=False,
        )


def test_add_invalid_directory_path_raises_error(context_manager, tmp_path):
    """Test that invalid directory paths raise an error"""
    db_path = str(tmp_path / "test_index.db")

    with pytest.raises(InvalidDirectoryPathError):
        context_manager.add_indexed_directory(
            "",
            db_path,
            "en_core_web_sm",
            lemmatization_enabled=True,
            stemming_enabled=False,
        )

    with pytest.raises(InvalidDirectoryPathError):
        context_manager.add_indexed_directory(
            None,
            db_path,
            "en_core_web_sm",
            lemmatization_enabled=True,
            stemming_enabled=False,
        )


def test_remove_indexed_directory(context_manager, tmp_path):
    """Test removing (deactivating) an indexed directory"""
    directory_path = "/home/user/documents"
    db_path = str(tmp_path / "test_index.db")
    model_name = "en_core_web_sm"

    context_manager.add_indexed_directory(
        directory_path,
        db_path,
        model_name,
        lemmatization_enabled=True,
        stemming_enabled=False,
    )

    # Verify directory is active
    active_dir = context_manager.get_active_directory()
    assert active_dir is not None

    # Remove directory
    context_manager.remove_indexed_directory(directory_path)

    # Verify no active directory exists
    active_dir = context_manager.get_active_directory()
    assert active_dir is None


def test_remove_nonexistent_directory_raises_error(context_manager):
    """Test that removing a non-existent directory raises an error"""
    with pytest.raises(DirectoryNotFoundError, match="not found in index"):
        context_manager.remove_indexed_directory("/nonexistent/path")


def test_get_active_directory_when_none_exists(context_manager):
    """Test getting active directory when none exists"""
    active_dir = context_manager.get_active_directory()
    assert active_dir is None


def test_get_active_directory_returns_correct_directory(context_manager, tmp_path):
    """Test that get_active_directory returns the correct active directory"""
    dir1 = "/home/user/docs1"
    dir2 = "/home/user/docs2"
    db_path1 = str(tmp_path / "index1.db")
    db_path2 = str(tmp_path / "index2.db")
    model = "en_core_web_sm"

    # Add first directory (will be active)
    context_manager.add_indexed_directory(
        dir1, db_path1, model, lemmatization_enabled=True, stemming_enabled=False
    )

    # Add second directory (will also be active, replacing first)
    context_manager.add_indexed_directory(
        dir2, db_path2, model, lemmatization_enabled=True, stemming_enabled=False
    )

    active_dir = context_manager.get_active_directory()
    assert active_dir["directory_path"] == dir2


def test_set_active_directory(context_manager, tmp_path):
    """Test setting a directory as active"""
    dir1 = "/home/user/docs1"
    dir2 = "/home/user/docs2"
    db_path1 = str(tmp_path / "index1.db")
    db_path2 = str(tmp_path / "index2.db")
    model = "en_core_web_sm"

    # Add two directories
    context_manager.add_indexed_directory(
        dir1, db_path1, model, lemmatization_enabled=True, stemming_enabled=False
    )
    context_manager.add_indexed_directory(
        dir2, db_path2, model, lemmatization_enabled=True, stemming_enabled=False
    )

    # Set first directory as active
    result = context_manager.set_active_directory(dir1)

    assert result["directory_path"] == dir1
    assert result["tokenizer_model_name"] == model
    assert result["lemmatization_enabled"] is True
    assert result["stemming_enabled"] is False

    # Verify it's the active directory
    active_dir = context_manager.get_active_directory()
    assert active_dir["directory_path"] == dir1


def test_set_active_directory_deactivates_others(context_manager, tmp_path):
    """Test that setting a directory as active deactivates all others"""
    dir1 = "/home/user/docs1"
    dir2 = "/home/user/docs2"
    dir3 = "/home/user/docs3"
    db_path1 = str(tmp_path / "index1.db")
    db_path2 = str(tmp_path / "index2.db")
    db_path3 = str(tmp_path / "index3.db")
    model = "en_core_web_sm"

    # Add three directories
    context_manager.add_indexed_directory(
        dir1, db_path1, model, lemmatization_enabled=True, stemming_enabled=False
    )
    context_manager.add_indexed_directory(
        dir2, db_path2, model, lemmatization_enabled=True, stemming_enabled=False
    )
    context_manager.add_indexed_directory(
        dir3, db_path3, model, lemmatization_enabled=True, stemming_enabled=False
    )

    # Set second directory as active
    context_manager.set_active_directory(dir2)

    # Verify only dir2 is active
    active_dir = context_manager.get_active_directory()
    assert active_dir["directory_path"] == dir2

    # Set first directory as active
    context_manager.set_active_directory(dir1)

    # Verify only dir1 is active now
    active_dir = context_manager.get_active_directory()
    assert active_dir["directory_path"] == dir1


def test_set_active_directory_nonexistent_raises_error(context_manager):
    """Test that setting a non-existent directory as active raises an error"""
    with pytest.raises(DirectoryNotFoundError, match="not found in index"):
        context_manager.set_active_directory("/nonexistent/path")


def test_set_active_directory_returns_full_info(context_manager, tmp_path):
    """Test that set_active_directory returns complete directory information"""
    directory_path = "/home/user/documents"
    db_path = str(tmp_path / "test_index.db")
    model_name = "en_core_web_sm"
    model_version = "3.7.0"

    context_manager.add_indexed_directory(
        directory_path,
        db_path,
        model_name,
        model_version,
        lemmatization_enabled=True,
        stemming_enabled=False,
    )

    result = context_manager.set_active_directory(directory_path)

    assert result["directory_path"] == directory_path
    assert result["db_path"] == db_path
    assert result["tokenizer_model_name"] == model_name
    assert result["tokenizer_model_version"] == model_version
    assert result["lemmatization_enabled"] is True
    assert result["stemming_enabled"] is False
