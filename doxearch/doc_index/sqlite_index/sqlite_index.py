# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false

import math
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

from sqlalchemy import Column, Float, ForeignKey, Index, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

from doxearch.doc_index.doc_index import DocIndex
from doxearch.doc_index.sqlite_index.exceptions import (
    DocumentExistsError,
    DocumentNotFoundError,
    InvalidDocumentIdError,
    InvalidFilePathError,
    InvalidTermFrequencyError,
)
from doxearch.doc_index.types import (
    DocumentMetadata,
    TermDocumentPosting,
    TermFrequency,
)

Base = declarative_base()


class Document(Base):
    """
    Represents a document in the corpus
    """

    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(255), nullable=False)
    term_count = Column(Integer, nullable=False)  # Total terms in document
    unique_terms = Column(Integer, nullable=False)  # Number of unique terms
    last_indexed = Column(
        Integer, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    term_frequencies = relationship(
        "InvertedIndex", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Document(doc_id='{self.doc_id}', filename='{self.filename}, terms={self.term_count})>"


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
    normalized_tf = Column(
        Float, nullable=False
    )  # Normalized term frequency for TF-IDF

    # Relationships
    document = relationship("Document", back_populates="term_frequencies")
    term_stats = relationship("DocumentFrequency", back_populates="postings")

    # Indexes for fast lookups
    __table_args__ = (
        Index("idx_inverted_term", "term"),
        Index("idx_inverted_doc", "doc_id"),
    )

    def __repr__(self):
        return f"<InvertedIndex(term='{self.term}', doc='{self.doc_id}', freq={self.term_frequency}, norm_tf={self.normalized_tf:.4f})>"


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
        self,
        document_id: str,
        term_frequencies: dict[str, int],
        filename: str,
        filepath: str,
    ) -> None:
        """Add a document to the index with pre-computed term frequencies.

        Args:
            document_id (str): Unique identifier for the document.
            term_frequencies (dict[str, int]): Dictionary mapping terms to their frequencies in the document
                e.g., {"search": 5, "engine": 3, "document": 7}
            filename (str): Original filename of the document
            filepath (str): Path to the original document file

        Raises:
            InvalidDocumentIdError: If document_id is empty or None
            InvalidTermFrequencyError: If term_frequencies is empty or contains invalid values
            InvalidFilePathError: If filepath is empty or None
            DocumentExistsError: If document_id already exists in the index
        """
        self._validate_add_document_inputs(document_id, term_frequencies, filepath)

        with self.get_session() as session:
            self._validate_document_id(session, document_id)
            self._create_document_record(
                session, document_id, term_frequencies, filename, filepath
            )
            self._update_document_frequencies(session, term_frequencies)
            session.flush()
            self._create_inverted_index_entries(session, document_id, term_frequencies)
            self._update_corpus_statistics(session)
            session.commit()

    def _validate_add_document_inputs(
        self, document_id: str, term_frequencies: dict[str, int], filepath: str
    ) -> None:
        """Validate inputs for add_document method.

        Args:
            document_id (str): Document identifier to validate
            term_frequencies (dict[str, int]): Term frequencies to validate
            filepath (str): File path to validate

        Raises:
            InvalidDocumentIdError: If document_id is empty or None
            InvalidTermFrequencyError: If term_frequencies is invalid
            InvalidFilePathError: If filepath is empty or None
        """
        if not document_id or not isinstance(document_id, str):
            raise InvalidDocumentIdError("Document ID must be a non-empty string")

        if not filepath or not isinstance(filepath, str):
            raise InvalidFilePathError("File path must be a non-empty string")

        if not term_frequencies:
            raise InvalidTermFrequencyError(
                "Term frequencies dictionary cannot be empty"
            )

        if not isinstance(term_frequencies, dict):
            raise InvalidTermFrequencyError("Term frequencies must be a dictionary")

        for term, freq in term_frequencies.items():
            if not isinstance(term, str) or not term:
                raise InvalidTermFrequencyError(
                    f"Term must be a non-empty string, got: {term}"
                )
            if not isinstance(freq, int) or freq <= 0:
                raise InvalidTermFrequencyError(
                    f"Frequency for term '{term}' must be a positive integer, got: {freq}"
                )

    def _validate_document_id(self, session, document_id: str) -> None:
        """Check if document already exists in the index.

        Args:
            session: SQLAlchemy session
            document_id (str): Document identifier to validate

        Raises:
            DocumentExistsError: If document_id already exists
        """
        existing_doc = session.query(Document).filter_by(doc_id=document_id).first()
        if existing_doc:
            raise DocumentExistsError(document_id)

    def _create_document_record(
        self,
        session,
        document_id: str,
        term_frequencies: dict[str, int],
        filename: str,
        filepath: str,
    ) -> None:
        """Create and add a document record to the session.

        Args:
            session: SQLAlchemy session
            document_id (str): Unique document identifier
            term_frequencies (dict[str, int]): Term frequency mapping
            filename (str): Original filename of the document
            filepath (str): Path to the document file
        """
        total_terms = sum(term_frequencies.values())
        unique_terms = len(term_frequencies)

        document = Document(
            doc_id=document_id,
            term_count=total_terms,
            unique_terms=unique_terms,
            filename=filename,
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
        """Create inverted index entries for a document.

        Args:
            session: SQLAlchemy session
            document_id (str): Document identifier
            term_frequencies (dict[str, int]): Term frequency mapping
        """

        # Calculate document length for normalization
        doc_length = sum(term_frequencies.values())

        for term, frequency in term_frequencies.items():
            # Calculate normalized TF using L2 normalization
            normalized_tf = frequency / math.sqrt(doc_length)

            inverted_entry = InvertedIndex(
                term=term,
                doc_id=document_id,
                term_frequency=frequency,
                normalized_tf=normalized_tf,
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

    def remove_document(self, document_id: str) -> None:
        """Remove a document from the index and update all related statistics.

        This method:
        1. Removes the document record (cascades to inverted index entries)
        2. Updates document frequency counts for all terms in the document
        3. Removes document frequency entries for terms that no longer appear in any document
        4. Updates corpus statistics

        Args:
            document_id (str): Unique identifier of the document to remove

        Raises:
            InvalidDocumentIdError: If document_id is empty or None
            DocumentNotFoundError: If document_id does not exist in the index
        """
        if not document_id or not isinstance(document_id, str):
            raise InvalidDocumentIdError("Document ID must be a non-empty string")

        with self.get_session() as session:
            # Verify document exists
            document = session.query(Document).filter_by(doc_id=document_id).first()
            if not document:
                raise DocumentNotFoundError(document_id)

            # Get all terms and their frequencies from this document before deletion
            inverted_entries = (
                session.query(InvertedIndex.term, InvertedIndex.term_frequency)
                .filter_by(doc_id=document_id)
                .all()
            )
            term_frequencies: dict[str, int] = {
                term: frequency for term, frequency in inverted_entries
            }
            # Delete document (cascades to inverted index entries due to relationship)
            session.delete(document)
            session.flush()

            # Update document frequencies for each term
            self._decrement_document_frequencies(session, term_frequencies)

            # Update corpus statistics
            self._decrement_corpus_statistics(session)

            session.commit()

    def _decrement_document_frequencies(
        self, session, term_frequencies: dict[str, int]
    ) -> None:
        """Decrement document frequency counts and remove entries for terms no longer in corpus.

        Args:
            session: SQLAlchemy session
            term_frequencies (dict[str, int]): Term frequency mapping from removed document
        """
        for term, frequency in term_frequencies.items():
            df_entry = session.query(DocumentFrequency).filter_by(term=term).first()

            if df_entry:
                new_doc_count = df_entry.doc_count - 1
                new_total_frequency = df_entry.total_frequency - frequency

                if new_doc_count == 0:
                    # Term no longer appears in any document, remove entry
                    session.delete(df_entry)
                else:
                    # Update counts
                    session.query(DocumentFrequency).filter_by(term=term).update(
                        {
                            "doc_count": new_doc_count,
                            "total_frequency": new_total_frequency,
                        }
                    )

    def _decrement_corpus_statistics(self, session) -> None:
        """Decrement global corpus statistics (total document count).

        Args:
            session: SQLAlchemy session
        """
        total_docs_stat = (
            session.query(CorpusStats).filter_by(stat_name="total_documents").first()
        )

        if total_docs_stat:
            new_value = total_docs_stat.stat_value - 1
            if new_value == 0:
                # No documents left, remove the stat entry
                session.delete(total_docs_stat)
            else:
                session.query(CorpusStats).filter_by(
                    stat_name="total_documents"
                ).update({"stat_value": new_value})

    def update_document(
        self,
        document_id: str,
        term_frequencies: dict[str, int],
        filename: str,
        filepath: str,
    ) -> None:
        """Update an existing document in the index with new term frequencies.

        This method removes the old document and adds it back with new data,
        effectively updating all related statistics and inverted index entries.

        Args:
            document_id (str): Unique identifier of the document to update
            term_frequencies (dict[str, int]): New term frequency mapping
            filename (str): Name of the document file (can be updated)
            filepath (str): Path to the document file (can be updated)

        Raises:
            InvalidDocumentIdError: If document_id is empty or None
            InvalidTermFrequencyError: If term_frequencies is invalid
            InvalidFilePathError: If filepath is empty or None
            DocumentNotFoundError: If document_id does not exist in the index
        """
        self._validate_add_document_inputs(document_id, term_frequencies, filepath)

        # Remove the old document (this validates existence and updates all stats)
        self.remove_document(document_id)

        # Add the document back with new data
        self.add_document(document_id, term_frequencies, filename, filepath)

    def update_document_file_path(
        self, document_id: str, filename: str, filepath: str
    ) -> None:
        """Update the file path of a document in the index.

        This method updates only the filename and filepath of an existing document
        without modifying its term frequencies or other statistics.

        Args:
            document_id (str): Unique identifier of the document to update
            filename (str): New filename for the document
            filepath (str): New file path for the document

        Raises:
            InvalidDocumentIdError: If document_id is empty or None
            InvalidFilePathError: If filepath is empty or None
            DocumentNotFoundError: If document_id does not exist in the index
        """
        if not document_id or not isinstance(document_id, str):
            raise InvalidDocumentIdError("Document ID must be a non-empty string")

        if not filepath or not isinstance(filepath, str):
            raise InvalidFilePathError("File path must be a non-empty string")

        with self.get_session() as session:
            # Verify document exists
            document = session.query(Document).filter_by(doc_id=document_id).first()
            if not document:
                raise DocumentNotFoundError(document_id)

            # Update filename and filepath
            document.filename = filename
            document.file_path = filepath
            document.last_indexed = int(datetime.now(timezone.utc).timestamp())

            session.commit()

    def document_exists(self, document_id: str) -> bool:
        """Check if a document exists in the index.
        Args:
            document_id (str): ID of the document to check

        Returns:
            bool: True if the document exists, False otherwise
        """
        with self.get_session() as session:
            return (
                session.query(Document).filter_by(doc_id=document_id).first()
                is not None
            )

    def check_bulk_documents_exist(self, document_ids: list[str]) -> dict[str, bool]:
        """Check if multiple documents exist in the index. (Bulk db operation).
        Args:
            document_ids (list[str]): List of IDs of documents to check

        Returns:
            dict[str, bool]: A dictionary mapping document IDs to their existence status
        """
        if not document_ids:
            return {}

        with self.get_session() as session:
            # Single query to fetch all existing document IDs
            existing_docs = (
                session.query(Document.doc_id)
                .filter(Document.doc_id.in_(document_ids))
                .all()
            )

            # Convert to set for O(1) lookup
            existing_ids = {doc.doc_id for doc in existing_docs}

            # Build result dictionary
            return {doc_id: doc_id in existing_ids for doc_id in document_ids}

    def get_document_count(self) -> int:
        """Get total count of documents in the index

        Returns:
            int: total count of documents in the index
        """
        with self.get_session() as session:
            return session.query(Document).count()

    def get_term_frequencies(self, terms: list[str]) -> list[TermFrequency]:
        """Get document frequency information for multiple terms."""
        with self.get_session() as session:
            df_entries = (
                session.query(DocumentFrequency)
                .filter(DocumentFrequency.term.in_(terms))
                .all()
            )

            return [
                TermFrequency(term=df.term, doc_count=df.doc_count) for df in df_entries
            ]

    def get_postings(self, terms: list[str]) -> list[TermDocumentPosting]:
        """Get all postings for the given terms."""
        with self.get_session() as session:
            postings = (
                session.query(InvertedIndex).filter(InvertedIndex.term.in_(terms)).all()
            )

            return [
                TermDocumentPosting(
                    term=p.term,
                    doc_id=p.doc_id,
                    normalized_tf=p.normalized_tf,
                )
                for p in postings
            ]

    def get_documents_metadata(self, doc_ids: list[str]) -> list[DocumentMetadata]:
        """Get metadata for multiple documents."""
        with self.get_session() as session:
            documents = (
                session.query(Document).filter(Document.doc_id.in_(doc_ids)).all()
            )

            return [
                DocumentMetadata(
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    file_path=doc.file_path,
                    term_count=doc.term_count,
                    unique_terms=doc.unique_terms,
                    last_indexed=doc.last_indexed,
                )
                for doc in documents
            ]

    def get_documents_by_folder(self, folder_path: str) -> list[DocumentMetadata]:
        """Get all documents in a specific folder."""
        with self.get_session() as session:
            documents = (
                session.query(Document)
                .filter(Document.file_path.like(f"{folder_path}%"))
                .all()
            )

            return [
                DocumentMetadata(
                    doc_id=str(doc.doc_id),
                    filename=str(doc.filename),
                    file_path=str(doc.file_path),
                    term_count=int(doc.term_count),
                    unique_terms=int(doc.unique_terms),
                    last_indexed=int(doc.last_indexed),
                )
                for doc in documents
            ]

    def add_documents_batch(
        self, documents: list[tuple[str, dict[str, int], str, str]]
    ) -> None:
        """
        Add multiple documents in a single transaction (bulk operation).

        This is significantly faster than calling add_document() multiple times
        because it uses a single database transaction and batches operations.

        Args:
            documents: List of (document_id, term_frequencies, filename, filepath) tuples

        Raises:
            InvalidDocumentIdError: If any document_id is invalid
            InvalidTermFrequencyError: If any term_frequencies is invalid
            InvalidFilePathError: If any filepath is invalid
            DocumentExistsError: If any document_id already exists
            DatabaseOperationError: If the batch operation fails
        """
        if not documents:
            return

        with self.get_session() as session:
            try:
                # Validate all documents first (fail fast)
                for doc_id, term_freq, filename, filepath in documents:
                    self._validate_add_document_inputs(doc_id, term_freq, filepath)
                    self._validate_document_id(session, doc_id)

                # Aggregate all term frequencies across all documents
                global_term_frequencies: dict[str, int] = {}
                global_term_doc_counts: dict[str, int] = {}

                for doc_id, term_freq, filename, filepath in documents:
                    for term, freq in term_freq.items():
                        global_term_frequencies[term] = (
                            global_term_frequencies.get(term, 0) + freq
                        )
                        global_term_doc_counts[term] = (
                            global_term_doc_counts.get(term, 0) + 1
                        )

                # Bulk insert document records
                doc_records = []
                for doc_id, term_freq, filename, filepath in documents:
                    total_terms = sum(term_freq.values())
                    unique_terms = len(term_freq)

                    doc_records.append(
                        Document(
                            doc_id=doc_id,
                            term_count=total_terms,
                            unique_terms=unique_terms,
                            filename=filename,
                            file_path=filepath,
                            last_indexed=int(datetime.now(timezone.utc).timestamp()),
                        )
                    )

                session.bulk_save_objects(doc_records)
                session.flush()

                # Update document frequencies in bulk
                existing_terms = (
                    session.query(DocumentFrequency)
                    .filter(DocumentFrequency.term.in_(global_term_frequencies.keys()))
                    .all()
                )

                existing_term_map = {df.term: df for df in existing_terms}
                new_df_records = []

                for term, total_freq in global_term_frequencies.items():
                    doc_count_increment = global_term_doc_counts[term]

                    if term in existing_term_map:
                        # Update existing term
                        df_entry = existing_term_map[term]
                        df_entry.doc_count += doc_count_increment
                        df_entry.total_frequency += total_freq
                    else:
                        # Create new term entry
                        new_df_records.append(
                            DocumentFrequency(
                                term=term,
                                doc_count=doc_count_increment,
                                total_frequency=total_freq,
                            )
                        )

                if new_df_records:
                    session.bulk_save_objects(new_df_records)

                session.flush()

                # Bulk insert inverted index entries
                inverted_records = []
                for doc_id, term_freq, filename, filepath in documents:
                    doc_length = sum(term_freq.values())

                    for term, frequency in term_freq.items():
                        normalized_tf = frequency / math.sqrt(doc_length)

                        inverted_records.append(
                            InvertedIndex(
                                term=term,
                                doc_id=doc_id,
                                term_frequency=frequency,
                                normalized_tf=normalized_tf,
                            )
                        )

                session.bulk_save_objects(inverted_records)
                session.flush()

                # Update corpus statistics
                total_docs_stat = (
                    session.query(CorpusStats)
                    .filter_by(stat_name="total_documents")
                    .first()
                )

                num_new_docs = len(documents)
                if total_docs_stat:
                    total_docs_stat.stat_value += num_new_docs
                else:
                    session.add(
                        CorpusStats(
                            stat_name="total_documents", stat_value=num_new_docs
                        )
                    )

                session.commit()

            except Exception as e:
                session.rollback()
                from doxearch.doc_index.sqlite_index.exceptions import (
                    DatabaseOperationError,
                )

                raise DatabaseOperationError("add_documents_batch", e) from e

    def get_all_documents(self) -> list[Document]:
        """Get all documents from the index ordered by filename.

        Returns:
            list[Document]: List of all documents ordered by filename

        Example:
            documents = index.get_all_documents_ordered()
            for doc in documents:
                print(f"{doc.filename}: {doc.term_count} terms")
        """
        with self.get_session() as session:
            return session.query(Document).order_by(Document.filename).all()

    def get_document_by_filepath(self, filepath: str) -> Document | None:
        """Get a document by its file path.

        Args:
            filepath (str): The file path to search for

        Returns:
            Document | None: The document if found, None otherwise

        Example:
            doc = index.get_document_by_filepath("/path/to/file.pdf")
            if doc:
                print(f"Found document: {doc.filename}")
        """
        with self.get_session() as session:
            return session.query(Document).filter_by(file_path=filepath).first()
