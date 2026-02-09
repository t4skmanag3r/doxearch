from abc import ABC, abstractmethod


class DocIndex(ABC):
    @abstractmethod
    def add_document(
        self, document_id: str, term_frequencies: dict[str, int], filepath: str
    ) -> None:
        """Add a document to the index."""

    @abstractmethod
    def remove_document(self, document_id: str) -> None:
        """Remove a document from the index."""

    @abstractmethod
    def update_document(
        self, document_id: str, term_frequencies: dict[str, int], filepath: str
    ) -> None:
        """Update an existing document in the index."""

    @abstractmethod
    def document_exists(self, document_id: str) -> bool:
        """Check if a document exists in the index."""

    @abstractmethod
    def get_document_count(self) -> int:
        """Get the total number of documents in the index."""

    @abstractmethod
    def reindex(self) -> None:
        """Reindex all documents."""

    def __contains__(self, document_id: str) -> bool:
        """Allow 'document_id in index' syntax."""
        return self.document_exists(document_id)

    def __len__(self) -> int:
        """Allow len(index) syntax."""
        return self.get_document_count()
