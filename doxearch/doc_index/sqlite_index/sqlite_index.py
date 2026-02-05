from datetime import datetime, timezone

from sqlalchemy import Column, ForeignKey, Index, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from doxearch.doc_index.doc_index import DocIndex

Base = declarative_base()


class Document(Base):
    """
    Represents a document in the corpus
    """

    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True)
    term_count = Column(Integer, nullable=False)  # Total terms in document
    unique_terms = Column(Integer, nullable=False)  # Number of unique terms
    file_path = Column(String(255), nullable=False)
    last_indexed = Column(
        Integer, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    term_frequencies = relationship(
        "InvertedIndex", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Document(doc_id='{self.doc_id}', terms={self.term_count})>"


class InvertedIndex(Base):
    """
    Inverted index: maps terms to documents
    Core table for TF-IDF calculations
    """

    __tablename__ = "inverted_index"

    term = Column(String, primary_key=True)
    doc_id = Column(
        String, ForeignKey("documents.doc_id", ondelete="CASCADE"), primary_key=True
    )
    term_frequency = Column(Integer, nullable=False)  # Raw count in document

    # Relationships
    document = relationship("Document", back_populates="term_frequencies")
    term_stats = relationship("DocumentFrequency", back_populates="postings")

    # Indexes for fast lookups
    __table_args__ = (
        Index("idx_inverted_term", "term"),
        Index("idx_inverted_doc", "doc_id"),
    )

    def __repr__(self):
        return f"<InvertedIndex(term='{self.term}', doc='{self.doc_id}', freq={self.term_frequency})>"


class DocumentFrequency(Base):
    """
    Document frequency cache: how many documents contain each term
    Used for IDF calculation
    """

    __tablename__ = "document_frequency"

    term = Column(String, primary_key=True)
    doc_count = Column(Integer, nullable=False)  # Number of docs containing term
    total_frequency = Column(Integer, nullable=False)  # Total occurrences across corpus

    # Relationships
    postings = relationship("InvertedIndex", back_populates="term_stats")

    __table_args__ = (Index("idx_df_doc_count", "doc_count"),)

    def __repr__(self):
        return f"<DocumentFrequency(term='{self.term}', docs={self.doc_count})>"


class CorpusStats(Base):
    """
    Global corpus statistics
    """

    __tablename__ = "corpus_stats"

    stat_name = Column(String, primary_key=True)
    stat_value = Column(Integer, nullable=False)

    def __repr__(self):
        return f"<CorpusStats({self.stat_name}={self.stat_value})>"


class SQLiteIndex(DocIndex):
    def __init__(self, db_path: str = "doxearch.db"):
        """Initialize SQLite index with the given database path."""
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)

    def add_document(self, document_id: int, document_text: list[str]) -> None: ...

    def remove_document(self, document_id: int) -> None: ...

    def get_document_count(self) -> int: ...
