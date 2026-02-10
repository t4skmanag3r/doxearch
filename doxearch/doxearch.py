import platform
from pathlib import Path
from typing import Counter

from doxearch.doc_index.doc_index import DocIndex
from doxearch.doc_index.sqlite_index.exceptions import (
    DocumentExistsError,
    InvalidTermFrequencyError,
)
from doxearch.doc_index.sqlite_index.sqlite_index import (
    Document,
    DocumentFrequency,
    InvertedIndex,
)
from doxearch.doc_parser.parsers.pdf_parser import PDFParser
from doxearch.tf_idf.tf_idf import compute_idf, compute_tf_idf
from doxearch.tokenizer.tokenizer import Tokenizer
from doxearch.utils.file_hash import compute_file_hash


def get_app_data_dir() -> Path:
    """Get the appropriate application data directory based on the operating system."""
    system = platform.system()

    if system == "Windows":
        # Windows: %APPDATA%\doxearch
        app_data = Path.home() / "AppData" / "Roaming" / "doxearch"
    elif system == "Linux":
        # Linux: ~/.local/share/doxearch
        app_data = Path.home() / ".local" / "share" / "doxearch"
    else:
        # Fallback to home directory
        app_data = Path.home() / ".doxearch"

    # Create directory if it doesn't exist
    app_data.mkdir(parents=True, exist_ok=True)

    return app_data


class Doxearch:
    def __init__(self, index: DocIndex, tokenizer: Tokenizer):
        self.index = index
        self.tokenizer = tokenizer
        self.pdf_doc_parser = PDFParser()

    def index_folder(self, folder_path: Path):
        indexed_documents = 0
        files = list(folder_path.rglob("*.pdf"))

        # Compute hashes for all files
        file_hashes = {
            str(file_path): compute_file_hash(file_path) for file_path in files
        }

        # Check which documents already exist using their hashes
        documents_exist = self.index.check_bulk_documents_exist(
            list(file_hashes.values())
        )

        # Clean up documents that no longer exist in the folder
        self._cleanup_missing_documents(folder_path, set(file_hashes.values()))

        # Filter files that don't exist in the index
        filtered_files = [
            Path(file_path)
            for file_path, file_hash in file_hashes.items()
            if not documents_exist[file_hash]
        ]

        for file_path in filtered_files:
            try:
                text = self.pdf_doc_parser.parse(file_path)
                tokens = self.tokenizer.tokenize(text)
                # Use Counter to get raw term counts (int), not normalized frequencies (float)
                term_counts = dict(Counter(tokens))

                # Use file hash as document ID
                doc_id = file_hashes[str(file_path)]
                filename = file_path.name

                self.index.add_document(doc_id, term_counts, filename, str(file_path))
                indexed_documents += 1
                print(f"Indexed document: {file_path.name}")
            except (DocumentExistsError, InvalidTermFrequencyError):
                pass

        print(f"\nIndexed {indexed_documents} documents.")

    def _cleanup_missing_documents(
        self, folder_path: Path, current_file_hashes: set[str]
    ):
        """
        Remove documents from the index that no longer exist in the specified folder.

        Args:
            folder_path: The folder being indexed
            current_file_hashes: Set of file hashes for files currently in the folder
        """

        removed_count = 0
        folder_path_str = str(folder_path.resolve())

        with self.index.get_session() as session:
            # Get all documents that belong to this folder
            documents_in_folder = (
                session.query(Document)
                .filter(Document.file_path.like(f"{folder_path_str}%"))
                .all()
            )

            # Find documents that no longer exist
            for doc in documents_in_folder:
                if doc.doc_id not in current_file_hashes:
                    # Verify the file actually doesn't exist
                    if not Path(doc.file_path).exists():
                        try:
                            self.index.remove_document(doc.doc_id)
                            removed_count += 1
                            print(f"Removed missing document: {doc.filename}")
                        except Exception as e:
                            print(f"Failed to remove document {doc.filename}: {e}")

        if removed_count > 0:
            print(f"\nRemoved {removed_count} missing documents from index.")

    def search(self, query: str, top_k: int = 10) -> list[dict[str, str | float]]:
        """
        Search for documents relevant to the query using pre-computed normalized TF-IDF scores.

        Args:
            query: The search query string
            top_k: Number of top results to return (default: 10)

        Returns:
            List of dictionaries containing document metadata and scores:
            [
                {
                    "doc_id": str,
                    "filename": str,
                    "filepath": str,
                    "score": float
                },
                ...
            ]
            Sorted by relevance score in descending order
        """

        # Tokenize the query
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        # Get unique query terms
        query_terms = list(set(query_tokens))

        # Get total document count for IDF calculation
        total_docs = self.index.get_document_count()
        if total_docs == 0:
            return []

        # Dictionary to store document scores: {doc_id: score}
        doc_scores = {}

        with self.index.get_session() as session:
            # Fetch all document frequencies in one query
            df_entries = (
                session.query(DocumentFrequency)
                .filter(DocumentFrequency.term.in_(query_terms))
                .all()
            )

            # Create a mapping of term -> idf
            term_idf = {}
            for df_entry in df_entries:
                idf = compute_idf(total_docs, df_entry.doc_count)
                term_idf[df_entry.term] = idf

            # Skip if no query terms found in corpus
            if not term_idf:
                return []

            # Fetch all relevant postings in one query
            # Now we use the pre-computed normalized_tf instead of raw term_frequency
            postings = (
                session.query(InvertedIndex)
                .filter(InvertedIndex.term.in_(term_idf.keys()))
                .all()
            )

            # Calculate TF-IDF scores using pre-normalized TF values
            for posting in postings:
                idf = term_idf[posting.term]

                # Use pre-computed normalized TF - no need to fetch document metadata!
                tf_idf_score = compute_tf_idf(posting.normalized_tf, idf)

                # Accumulate score for this document
                if posting.doc_id in doc_scores:
                    doc_scores[posting.doc_id] += tf_idf_score
                else:
                    doc_scores[posting.doc_id] = tf_idf_score

        # Sort documents by score in descending order
        sorted_doc_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        # Fetch document metadata for top results
        results = []
        with self.index.get_session() as session:
            from doxearch.doc_index.sqlite_index.sqlite_index import Document

            for doc_id, score in sorted_doc_ids:
                doc = session.query(Document).filter_by(doc_id=doc_id).first()
                if doc:
                    results.append(
                        {
                            "doc_id": doc_id,
                            "filename": doc.filename,
                            "filepath": doc.file_path,
                            "score": score,
                        }
                    )

        return results
