from abc import ABC, abstractmethod


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
