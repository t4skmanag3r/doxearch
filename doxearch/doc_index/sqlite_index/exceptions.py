"""Custom exceptions for SQLite document index operations."""


class SQLiteIndexError(Exception):
    """Base exception for all SQLite index errors."""


class DocumentExistsError(SQLiteIndexError):
    """Raised when attempting to add a document that already exists in the index."""

    def __init__(self, document_id: str):
        self.document_id = document_id
        super().__init__(
            f"Document with id '{document_id}' already exists in the index"
        )


class DocumentNotFoundError(SQLiteIndexError):
    """Raised when attempting to access a document that doesn't exist in the index."""

    def __init__(self, document_id: str):
        self.document_id = document_id
        super().__init__(
            f"Document with id '{document_id}' does not exist in the index"
        )


class InvalidTermFrequencyError(SQLiteIndexError):
    """Raised when term frequencies are invalid (e.g., negative values, empty dict)."""

    def __init__(self, message: str = "Invalid term frequencies provided"):
        super().__init__(message)


class InvalidDocumentIdError(SQLiteIndexError):
    """Raised when document ID is invalid (e.g., empty string, None)."""

    def __init__(self, message: str = "Invalid document ID provided"):
        super().__init__(message)


class InvalidFilePathError(SQLiteIndexError):
    """Raised when file path is invalid (e.g., empty string, None)."""

    def __init__(self, message: str = "Invalid file path provided"):
        super().__init__(message)


class DatabaseOperationError(SQLiteIndexError):
    """Raised when a database operation fails."""

    def __init__(self, operation: str, original_error: Exception):
        self.operation = operation
        self.original_error = original_error
        super().__init__(
            f"Database operation '{operation}' failed: {str(original_error)}"
        )


class CorruptedIndexError(SQLiteIndexError):
    """Raised when the index is in an inconsistent or corrupted state."""

    def __init__(self, message: str = "Index is in a corrupted state"):
        super().__init__(message)
