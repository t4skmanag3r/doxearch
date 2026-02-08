from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

from sqlalchemy import Column, ForeignKey, Index, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

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

    term = Column(String, ForeignKey("document_frequency.term"), primary_key=True)
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
        self.Session = sessionmaker(bind=self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions with automatic rollback on error.

        Yields:
            Session: SQLAlchemy session

        Example:
            with self.get_session() as session:
                # Do database operations
                session.add(document)
                session.commit()
        """
        session = self.Session()
        try:
            yield session
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def add_document(
        self, document_id: str, term_frequencies: dict[str, int], filepath: str
    ) -> None:
        """Add a document to the index with pre-computed term frequencies.

        Args:
            document_id (str): Unique identifier for the document.
            term_frequencies (dict[str, int]): Dictionary mapping terms to their frequencies in the document
                e.g., {"search": 5, "engine": 3, "document": 7}
            filepath (str): Path to the original document file

        Raises:
            ValueError: If document_id already exists in the index
        """
        with self.get_session() as session:
            self._validate_document_id(session, document_id)
            self._create_document_record(
                session, document_id, term_frequencies, filepath
            )
            self._update_document_frequencies(session, term_frequencies)
            session.flush()
            self._create_inverted_index_entries(session, document_id, term_frequencies)
            self._update_corpus_statistics(session)
            session.commit()

    def _validate_document_id(self, session, document_id: str) -> None:
        """Check if document already exists in the index.

        Args:
            session: SQLAlchemy session
            document_id (str): Document identifier to validate

        Raises:
            ValueError: If document_id already exists
        """
        existing_doc = session.query(Document).filter_by(doc_id=document_id).first()
        if existing_doc:
            raise ValueError(
                f"Document with id '{document_id}' already exists in the index"
            )

    def _create_document_record(
        self,
        session,
        document_id: str,
        term_frequencies: dict[str, int],
        filepath: str,
    ) -> None:
        """Create and add a document record to the session.

        Args:
            session: SQLAlchemy session
            document_id (str): Unique document identifier
            term_frequencies (dict[str, int]): Term frequency mapping
            filepath (str): Path to the document file
        """
        total_terms = sum(term_frequencies.values())
        unique_terms = len(term_frequencies)

        document = Document(
            doc_id=document_id,
            term_count=total_terms,
            unique_terms=unique_terms,
            file_path=filepath,
            last_indexed=int(datetime.now(timezone.utc).timestamp()),
        )
        session.add(document)

    def _update_document_frequencies(
        self, session, term_frequencies: dict[str, int]
    ) -> None:
        """Update or create document frequency entries for all terms.

        Args:
            session: SQLAlchemy session
            term_frequencies (dict[str, int]): Term frequency mapping
        """
        for term, frequency in term_frequencies.items():
            df_entry = session.query(DocumentFrequency).filter_by(term=term).first()

            if df_entry:
                # Term exists, update counts
                session.query(DocumentFrequency).filter_by(term=term).update(
                    {
                        "doc_count": DocumentFrequency.doc_count + 1,
                        "total_frequency": DocumentFrequency.total_frequency
                        + frequency,
                    }
                )
            else:
                # New term, create entry
                df_entry = DocumentFrequency(
                    term=term, doc_count=1, total_frequency=frequency
                )
                session.add(df_entry)

    def _create_inverted_index_entries(
        self, session, document_id: str, term_frequencies: dict[str, int]
    ) -> None:
        """Create inverted index entries for all terms in the document.

        Args:
            session: SQLAlchemy session
            document_id (str): Document identifier
            term_frequencies (dict[str, int]): Term frequency mapping
        """
        for term, frequency in term_frequencies.items():
            inverted_entry = InvertedIndex(
                term=term, doc_id=document_id, term_frequency=frequency
            )
            session.add(inverted_entry)

    def _update_corpus_statistics(self, session) -> None:
        """Update global corpus statistics (total document count).

        Args:
            session: SQLAlchemy session
        """
        total_docs_stat = (
            session.query(CorpusStats).filter_by(stat_name="total_documents").first()
        )

        if total_docs_stat:
            session.query(CorpusStats).filter_by(stat_name="total_documents").update(
                {"stat_value": CorpusStats.stat_value + 1}
            )
        else:
            total_docs_stat = CorpusStats(stat_name="total_documents", stat_value=1)
            session.add(total_docs_stat)

    def remove_document(self, document_id: int) -> None: ...

    def update_document(self, document_id: int, document_text: list[str]) -> None: ...

    def document_exists(self, document_id: int) -> bool:
        """Check if a document exists in the index.
        Args:
            document_id: ID of the document to check

        Returns:
            bool: True if the document exists, False otherwise
        """
        with self.get_session() as session:
            return (
                session.query(Document).filter_by(doc_id=document_id).first()
                is not None
            )

    def get_document_count(self) -> int:
        """Get total count of documents in the index

        Returns:
            int: total count of documents in the index
        """
        with self.get_session() as session:
            return session.query(Document).count()

    def reindex(self) -> None: ...
