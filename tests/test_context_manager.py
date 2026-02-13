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


def test_add_indexed_directory(context_manager):
    """Test adding a new indexed directory"""
    directory_path = "/home/user/documents"
    model_name = "en_core_web_sm"
    model_version = "3.7.0"

    context_manager.add_indexed_directory(directory_path, model_name, model_version)

    # Verify directory was added and is active
    active_dir = context_manager.get_active_directory()
    assert active_dir is not None
    assert active_dir["directory_path"] == directory_path
    assert active_dir["tokenizer_model_name"] == model_name
    assert active_dir["tokenizer_model_version"] == model_version


def test_add_indexed_directory_without_version(context_manager):
    """Test adding a directory without specifying model version"""
    directory_path = "/home/user/documents"
    model_name = "en_core_web_sm"

    context_manager.add_indexed_directory(directory_path, model_name)

    active_dir = context_manager.get_active_directory()
    assert active_dir["tokenizer_model_version"] is None


def test_add_duplicate_directory_raises_error(context_manager):
    """Test that adding a duplicate directory raises an error"""
    directory_path = "/home/user/documents"
    model_name = "en_core_web_sm"

    context_manager.add_indexed_directory(directory_path, model_name)

    with pytest.raises(DirectoryAlreadyIndexedError, match="already indexed"):
        context_manager.add_indexed_directory(directory_path, model_name)


def test_add_invalid_directory_path_raises_error(context_manager):
    """Test that invalid directory paths raise an error"""
    with pytest.raises(InvalidDirectoryPathError):
        context_manager.add_indexed_directory("", "en_core_web_sm")

    with pytest.raises(InvalidDirectoryPathError):
        context_manager.add_indexed_directory(None, "en_core_web_sm")


def test_remove_indexed_directory(context_manager):
    """Test removing (deactivating) an indexed directory"""
    directory_path = "/home/user/documents"
    model_name = "en_core_web_sm"

    context_manager.add_indexed_directory(directory_path, model_name)

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


def test_get_active_directory_returns_correct_directory(context_manager):
    """Test that get_active_directory returns the correct active directory"""
    dir1 = "/home/user/docs1"
    dir2 = "/home/user/docs2"
    model = "en_core_web_sm"

    # Add first directory (will be active)
    context_manager.add_indexed_directory(dir1, model)

    # Add second directory (will also be active, replacing first)
    context_manager.add_indexed_directory(dir2, model)

    active_dir = context_manager.get_active_directory()
    assert active_dir["directory_path"] == dir2


def test_set_active_directory(context_manager):
    """Test setting a directory as active"""
    dir1 = "/home/user/docs1"
    dir2 = "/home/user/docs2"
    model = "en_core_web_sm"

    # Add two directories
    context_manager.add_indexed_directory(dir1, model)
    context_manager.add_indexed_directory(dir2, model)

    # Set first directory as active
    result = context_manager.set_active_directory(dir1)

    assert result["directory_path"] == dir1
    assert result["tokenizer_model_name"] == model

    # Verify it's the active directory
    active_dir = context_manager.get_active_directory()
    assert active_dir["directory_path"] == dir1


def test_set_active_directory_deactivates_others(context_manager):
    """Test that setting a directory as active deactivates all others"""
    dir1 = "/home/user/docs1"
    dir2 = "/home/user/docs2"
    dir3 = "/home/user/docs3"
    model = "en_core_web_sm"

    # Add three directories
    context_manager.add_indexed_directory(dir1, model)
    context_manager.add_indexed_directory(dir2, model)
    context_manager.add_indexed_directory(dir3, model)

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


def test_set_active_directory_returns_full_info(context_manager):
    """Test that set_active_directory returns complete directory information"""
    directory_path = "/home/user/documents"
    model_name = "en_core_web_sm"
    model_version = "3.7.0"

    context_manager.add_indexed_directory(directory_path, model_name, model_version)

    result = context_manager.set_active_directory(directory_path)

    assert result["directory_path"] == directory_path
    assert result["tokenizer_model_name"] == model_name
    assert result["tokenizer_model_version"] == model_version
