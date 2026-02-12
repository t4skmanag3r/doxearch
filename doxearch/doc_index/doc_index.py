from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator

from doxearch.doc_index.types import (
    DocumentMetadata,
    TermDocumentPosting,
    TermFrequency,
)


class DocIndex(ABC):
    @abstractmethod
    def add_document(
        self,
        document_id: str,
        term_frequencies: dict[str, int],
        filename: str,
        filepath: str,
    ) -> None:
        """Add a document to the index."""

    @abstractmethod
    def remove_document(self, document_id: str) -> None:
        """Remove a document from the index."""

    @abstractmethod
    def update_document(
        self,
        document_id: str,
        term_frequencies: dict[str, int],
        filename: str,
        filepath: str,
    ) -> None:
        """Update an existing document in the index."""

    @abstractmethod
    def update_document_file_path(
        self, document_id: str, filename: str, filepath: str
    ) -> None:
        """Update the file path of a document in the index."""

    @abstractmethod
    def document_exists(self, document_id: str) -> bool:
        """Check if a document exists in the index."""

    @abstractmethod
    def check_bulk_documents_exist(self, document_ids: list[str]) -> dict[str, bool]:
        """Check if multiple documents exist in the index. (Bulk db operation)."""

    @abstractmethod
    def get_document_count(self) -> int:
        """Get the total number of documents in the index."""

    def __contains__(self, document_id: str) -> bool:
        """Allow 'document_id in index' syntax."""
        return self.document_exists(document_id)

    def __len__(self) -> int:
        """Allow len(index) syntax."""
        return self.get_document_count()

    @abstractmethod
    def get_term_frequencies(self, terms: list[str]) -> list[TermFrequency]:
        """
        Get document frequency information for multiple terms.

        Args:
            terms: List of terms to look up

        Returns:
            List of TermFrequency objects
        """

    @abstractmethod
    def get_postings(self, terms: list[str]) -> list[TermDocumentPosting]:
        """
        Get all postings (term-document pairs) for the given terms.

        Args:
            terms: List of terms to look up

        Returns:
            List of TermDocumentPosting objects with normalized TF values
        """

    @abstractmethod
    def get_documents_metadata(self, doc_ids: list[str]) -> list[DocumentMetadata]:
        """
        Get metadata for multiple documents.

        Args:
            doc_ids: List of document IDs

        Returns:
            List of DocumentMetadata objects
        """

    @abstractmethod
    def get_documents_by_folder(self, folder_path: str) -> list[DocumentMetadata]:
        """
        Get all documents that belong to a specific folder.

        Args:
            folder_path: Path to the folder

        Returns:
            List of DocumentMetadata objects
        """

    @abstractmethod
    @contextmanager
    def get_session(self) -> Generator:
        """
        Context manager for database sessions (if applicable).
        For non-database implementations, this can be a no-op.
        """

    @abstractmethod
    def add_documents_batch(
        self, documents: list[tuple[str, dict[str, int], str, str]]
    ) -> None:
        """
        Add multiple documents in a single transaction (bulk operation).
        """
