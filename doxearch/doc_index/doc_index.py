from abc import ABC


class DocIndex(ABC):
    def add_document(
        self, document_id: str, term_frequencies: dict[str, int], filepath: str
    ) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def remove_document(self, document_id: int) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def update_document(
        self, document_id: int, term_frequencies: dict[str, int], filepath: str
    ) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def document_exists(self, document_id: int) -> bool:
        raise NotImplementedError("Subclasses must implement this method")

    def get_document_count(self) -> int:
        raise NotImplementedError("Subclasses must implement this method")

    def reindex(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def __contains__(self, document_id: int) -> bool:
        return self.document_exists(document_id)

    def __len__(self) -> int:
        return self.get_document_count()
