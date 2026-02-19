from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import Boolean, Column, Integer, String, create_engine
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
    is_active = Column(Boolean, default=0)  # 0 = inactive, 1 = active
    lemmatization_enabled = Column(Boolean, default=1)
    stemming_enabled = Column(Boolean, default=0)


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
        tokenizer_model_name: str,
        model_version: str | None = None,
        lemmatization_enabled: bool = True,
        stemming_enabled: bool = False,
    ) -> dict:
        """
        Add a new indexed directory to the context.

        Args:
            directory_path: Path to the directory
            db_path: Path to the SQLite database file
            tokenizer_model_name: Name of the tokenizer model used
            model_version: Version of the tokenizer model (optional)
            lemmatization_enabled: Whether lemmatization is enabled
            stemming_enabled: Whether stemming is enabled

        Returns:
            Dictionary with the added directory information

        Raises:
            DirectoryAlreadyIndexedError: If directory is already indexed
            InvalidDirectoryPathError: If directory path is invalid
        """
        if not directory_path:
            raise InvalidDirectoryPathError(directory_path)

        with self.get_session() as session:
            # Check if directory already exists
            existing = (
                session.query(IndexedDirectory)
                .filter_by(directory_path=directory_path)
                .first()
            )

            if existing:
                raise DirectoryAlreadyIndexedError(directory_path)

            # Deactivate all other directories
            session.query(IndexedDirectory).update({"is_active": False})

            # Create new directory entry
            new_directory = IndexedDirectory(
                directory_path=directory_path,
                db_path=db_path,
                tokenizer_model_name=tokenizer_model_name,
                tokenizer_model_version=model_version,
                is_active=True,
                lemmatization_enabled=lemmatization_enabled,
                stemming_enabled=stemming_enabled,
            )

            session.add(new_directory)
            session.commit()

            # Refresh to get the latest state and extract data before session closes
            session.refresh(new_directory)
            result = {
                "directory_path": new_directory.directory_path,
                "db_path": new_directory.db_path,
                "tokenizer_model_name": new_directory.tokenizer_model_name,
                "tokenizer_model_version": new_directory.tokenizer_model_version,
                "is_active": bool(new_directory.is_active),
                "lemmatization_enabled": bool(new_directory.lemmatization_enabled),
                "stemming_enabled": bool(new_directory.stemming_enabled),
            }

        return result

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
        Get the currently active directory.

        Returns:
            Dictionary with active directory information or None if no active directory
        """
        with self.get_session() as session:
            directory = (
                session.query(IndexedDirectory).filter_by(is_active=True).first()
            )

            if not directory:
                return None

            # Extract data before session closes
            return {
                "directory_path": directory.directory_path,
                "db_path": directory.db_path,
                "tokenizer_model_name": directory.tokenizer_model_name,
                "tokenizer_model_version": directory.tokenizer_model_version,
                "is_active": bool(directory.is_active),
                "lemmatization_enabled": bool(directory.lemmatization_enabled),
                "stemming_enabled": bool(directory.stemming_enabled),
            }

    def set_active_directory(self, directory_path: str) -> dict:
        """
        Set a directory as the active directory.

        Args:
            directory_path: Path to the directory to set as active

        Returns:
            Dictionary with the activated directory information

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

            # Deactivate all directories
            session.query(IndexedDirectory).update({"is_active": False})

            # Activate the specified directory
            directory.is_active = True
            session.commit()

            # Refresh and extract data before session closes
            session.refresh(directory)
            result = {
                "directory_path": directory.directory_path,
                "db_path": directory.db_path,
                "tokenizer_model_name": directory.tokenizer_model_name,
                "tokenizer_model_version": directory.tokenizer_model_version,
                "is_active": bool(directory.is_active),
                "lemmatization_enabled": bool(directory.lemmatization_enabled),
                "stemming_enabled": bool(directory.stemming_enabled),
            }

        return result

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
                "lemmatization_enabled": bool(directory.lemmatization_enabled),
                "stemming_enabled": bool(directory.stemming_enabled),
            }

    def get_all_directories(self) -> list[dict]:
        """
        Get all indexed directories.

        Returns:
            List of dictionaries with directory information
        """
        with self.get_session() as session:
            directories = session.query(IndexedDirectory).all()

            # Extract data before session closes
            return [
                {
                    "directory_path": directory.directory_path,
                    "db_path": directory.db_path,
                    "tokenizer_model_name": directory.tokenizer_model_name,
                    "tokenizer_model_version": directory.tokenizer_model_version,
                    "is_active": bool(directory.is_active),
                    "lemmatization_enabled": bool(directory.lemmatization_enabled),
                    "stemming_enabled": bool(directory.stemming_enabled),
                }
                for directory in directories
            ]
