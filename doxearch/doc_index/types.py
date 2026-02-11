from dataclasses import dataclass


@dataclass
class DocumentMetadata:
    """Generic document metadata structure."""

    doc_id: str
    filename: str
    file_path: str
    term_count: int
    unique_terms: int
    last_indexed: str


@dataclass
class TermDocumentPosting:
    """Represents a term's occurrence in a document."""

    term: str
    doc_id: str
    normalized_tf: float


@dataclass
class TermFrequency:
    """Document frequency information for a term."""

    term: str
    doc_count: int
