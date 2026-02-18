from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from doxearch.exceptions import (
    DirectoryAlreadyIndexedError,
    DirectoryNotFoundError,
    InvalidDirectoryPathError,
)

Base = declarative_base()


class IndexedDirectory(Base):
    __tablename__ = "indexed_directories"

    id = Column(Integer, primary_key=True)
    directory_path = Column(String, unique=True, nullable=False)
    db_path = Column(String, nullable=False)
    tokenizer_model_name = Column(String, nullable=False)
    tokenizer_model_version = Column(String, nullable=True)
    is_active = Column(Integer, default=0)  # 0 = inactive, 1 = active


class DirectoryContextManager:
    def __init__(self, db_path: str = "context_manager.db"):
        """Initialize the context manager with the given database path."""
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Provide a transactional scope for database operations."""
        session = self.Session()
        try:
            yield session
        finally:
            session.close()

    def add_indexed_directory(
        self,
        directory_path: str,
        db_path: str,
        model_name: str,
        model_version: str | None = None,
    ) -> None:
        """
        Add a new indexed directory to the context manager.

        Args:
            directory_path: Absolute path to the directory
            db_path: Path to the database file for this directory
            model_name: Name of the tokenizer model used
            model_version: Version of the tokenizer model (optional)

        Raises:
            InvalidDirectoryPathError: If directory_path is invalid
            DirectoryAlreadyIndexedError: If directory is already indexed
        """
        if not directory_path or not isinstance(directory_path, str):
            raise InvalidDirectoryPathError("Directory path must be a non-empty string")

        with self.get_session() as session:
            # Check if directory already exists
            existing = (
                session.query(IndexedDirectory)
                .filter_by(directory_path=directory_path)
                .first()
            )

            if existing:
                raise DirectoryAlreadyIndexedError(directory_path)

            # Deactivate all existing directories
            session.query(IndexedDirectory).update({"is_active": 0})

            # Add new directory as active
            new_directory = IndexedDirectory(
                directory_path=directory_path,
                db_path=db_path,
                tokenizer_model_name=model_name,
                tokenizer_model_version=model_version,
                is_active=1,
            )

            session.add(new_directory)
            session.commit()

    def remove_indexed_directory(self, directory_path: str) -> None:
        """
        Remove an indexed directory from the context manager.

        Args:
            directory_path: Path to the directory to remove

        Raises:
            DirectoryNotFoundError: If directory is not in the index
        """
        with self.get_session() as session:
            directory = (
                session.query(IndexedDirectory)
                .filter_by(directory_path=directory_path)
                .first()
            )

            if not directory:
                raise DirectoryNotFoundError(directory_path)

            session.delete(directory)
            session.commit()

    def get_active_directory(self) -> dict | None:
        """
        Get the currently active directory (the one marked as active).

        Returns:
            Dictionary with directory information or None if no active directory exists.
            Dictionary contains: directory_path, db_path, tokenizer_model_name, tokenizer_model_version
        """
        with self.get_session() as session:
            active_dir = session.query(IndexedDirectory).filter_by(is_active=1).first()

            if not active_dir:
                return None

            return {
                "directory_path": active_dir.directory_path,
                "db_path": active_dir.db_path,
                "tokenizer_model_name": active_dir.tokenizer_model_name,
                "tokenizer_model_version": active_dir.tokenizer_model_version,
            }

    def set_active_directory(self, directory_path: str) -> dict:
        """
        Get a directory by path and mark it as active (deactivating all others).

        Args:
            directory_path: Path to the directory to activate

        Returns:
            Dictionary with the activated directory information

        Raises:
            DirectoryNotFoundError: If directory is not in the index
        """
        with self.get_session() as session:
            # Find the directory to activate
            target_dir = (
                session.query(IndexedDirectory)
                .filter_by(directory_path=directory_path)
                .first()
            )

            if not target_dir:
                raise DirectoryNotFoundError(directory_path)

            # Deactivate all directories
            session.query(IndexedDirectory).update({"is_active": 0})

            # Activate the target directory
            target_dir.is_active = 1
            session.commit()

            return {
                "directory_path": target_dir.directory_path,
                "db_path": target_dir.db_path,
                "tokenizer_model_name": target_dir.tokenizer_model_name,
                "tokenizer_model_version": target_dir.tokenizer_model_version,
            }

    def get_directory_info(self, directory_path: str) -> dict | None:
        """
        Get information about a specific directory.

        Args:
            directory_path: Path to the directory

        Returns:
            Dictionary with directory information or None if not found
        """
        with self.get_session() as session:
            directory = (
                session.query(IndexedDirectory)
                .filter_by(directory_path=directory_path)
                .first()
            )

            if not directory:
                return None

            return {
                "directory_path": directory.directory_path,
                "db_path": directory.db_path,
                "tokenizer_model_name": directory.tokenizer_model_name,
                "tokenizer_model_version": directory.tokenizer_model_version,
                "is_active": bool(directory.is_active),
            }

    def get_all_directories(self) -> list[dict]:
        """Get all indexed directories with their information.

        Returns:
            list[dict]: List of dictionaries containing directory information

        Example:
            directories = context_manager.get_all_directories()
            for dir_info in directories:
                print(f"{dir_info['directory_path']}: {dir_info['is_active']}")
        """
        with self.get_session() as session:
            directories = session.query(IndexedDirectory).all()
            return [
                {
                    "directory_path": directory.directory_path,
                    "db_path": directory.db_path,
                    "tokenizer_model_name": directory.tokenizer_model_name,
                    "tokenizer_model_version": directory.tokenizer_model_version,
                    "is_active": directory.is_active,
                }
                for directory in directories
            ]
