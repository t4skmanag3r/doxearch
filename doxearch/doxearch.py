import platform
from pathlib import Path
from typing import Counter

from doxearch.doc_index.doc_index import DocIndex
from doxearch.doc_index.sqlite_index.exceptions import (
    DocumentExistsError,
    InvalidTermFrequencyError,
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

    def search(self, query: str, top_k: int = 10) -> list[dict[str, str | float]]:
        """Search for documents using the abstract index interface."""

        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        query_terms = list(set(query_tokens))
        total_docs = self.index.get_document_count()
        if total_docs == 0:
            return []

        doc_scores = {}

        # Use abstract interface methods instead of direct SQLAlchemy queries
        term_frequencies = self.index.get_term_frequencies(query_terms)

        # Create term -> IDF mapping
        term_idf = {}
        for tf in term_frequencies:
            idf = compute_idf(total_docs, tf.doc_count)
            term_idf[tf.term] = idf

        if not term_idf:
            return []

        # Get postings using abstract interface
        postings = self.index.get_postings(list(term_idf.keys()))

        # Calculate TF-IDF scores
        for posting in postings:
            idf = term_idf[posting.term]
            tf_idf_score = compute_tf_idf(posting.normalized_tf, idf)

            if posting.doc_id in doc_scores:
                doc_scores[posting.doc_id] += tf_idf_score
            else:
                doc_scores[posting.doc_id] = tf_idf_score

        # Sort and get top results
        sorted_doc_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        # Fetch metadata using abstract interface
        doc_ids = [doc_id for doc_id, _ in sorted_doc_ids]
        documents_metadata = self.index.get_documents_metadata(doc_ids)

        # Create lookup dict for O(1) access
        metadata_dict = {doc.doc_id: doc for doc in documents_metadata}

        # Build results
        results = []
        for doc_id, score in sorted_doc_ids:
            if doc_id in metadata_dict:
                doc = metadata_dict[doc_id]
                results.append(
                    {
                        "doc_id": doc_id,
                        "filename": doc.filename,
                        "filepath": doc.file_path,
                        "score": score,
                    }
                )

        return results

    def _cleanup_missing_documents(
        self, folder_path: Path, current_file_hashes: set[str]
    ):
        """Remove documents that no longer exist in the folder."""
        removed_count = 0
        folder_path_str = str(folder_path.resolve())

        # Use abstract interface instead of direct session access
        documents_in_folder = self.index.get_documents_by_folder(folder_path_str)

        for doc in documents_in_folder:
            if doc.doc_id not in current_file_hashes:
                if not Path(doc.file_path).exists():
                    try:
                        self.index.remove_document(doc.doc_id)
                        removed_count += 1
                        print(f"Removed missing document: {doc.filename}")
                    except Exception as e:
                        print(f"Failed to remove document {doc.filename}: {e}")

        if removed_count > 0:
            print(f"\nRemoved {removed_count} missing documents from index.")
