import os
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from doxearch.doc_index.doc_index import DocIndex
from doxearch.doc_parser.parsers.pdf_parser import PDFParser
from doxearch.tf_idf.tf_idf import compute_idf, compute_tf_idf
from doxearch.tokenizer.tokenizer import Tokenizer
from doxearch.utils.file_hash import compute_file_hash


class Doxearch:
    def __init__(
        self,
        folder_path: Path,
        index: DocIndex,
        tokenizer: Tokenizer,
        fuzzy_threshold: float = 0.8,
    ):
        """Initialize Doxearch.

        Args:
            folder_path: Path to folder containing documents
            index: Document index instance
            tokenizer: Tokenizer instance for processing documents and search queries
        """
        self.folder_path = folder_path
        self.index = index
        self.tokenizer = tokenizer
        self.pdf_doc_parser = PDFParser()
        self.fuzzy_threshold = fuzzy_threshold

    def index_folder(self, batch_size: int = 100):
        """Index all PDF documents in a folder using batch database operations.

        Args:
            batch_size: Number of documents to insert in a single batch (default: 100)
        """
        start_time = time.time()
        files = list(self.folder_path.rglob("*.pdf"))

        print(f"Found {len(files)} PDF files to process\n")

        # Compute file hashes
        file_hashes, hash_time = self._compute_file_hashes(files)

        # Check which documents already exist
        documents_exist, check_time = self._check_existing_documents(file_hashes)

        # Clean up documents that no longer exist
        cleanup_time = self._cleanup_documents(
            self.folder_path, set(file_hashes.values())
        )

        # Filter files that need to be indexed
        filtered_files = self._filter_new_files(file_hashes, documents_exist)

        if not filtered_files:
            print("No new documents to index.")
            return

        print(f"Processing {len(filtered_files)} new documents\n")

        # Process and index documents in batches
        indexed_count, skipped_count, parse_time, tokenize_time, index_time = (
            self._process_and_index_documents(filtered_files, file_hashes, batch_size)
        )

        total_time = time.time() - start_time

        # Print summary
        self._print_indexing_summary(
            total_files=len(files),
            indexed_count=indexed_count,
            skipped_count=skipped_count,
            hash_time=hash_time,
            check_time=check_time,
            cleanup_time=cleanup_time,
            parse_time=parse_time,
            tokenize_time=tokenize_time,
            index_time=index_time,
            total_time=total_time,
        )

    def _compute_file_hashes(self, files: list[Path]) -> tuple[dict[str, str], float]:
        """Compute hashes for all files.

        Args:
            files: List of file paths

        Returns:
            Tuple of (file_path -> hash mapping, time taken)
        """
        hash_start = time.time()
        file_hashes = {
            str(file_path): compute_file_hash(file_path) for file_path in files
        }
        hash_time = time.time() - hash_start

        print(
            f"Hash computation took: {hash_time:.2f}s "
            f"({hash_time/len(files)*1000:.1f}ms per file)"
        )

        return file_hashes, hash_time

    def _check_existing_documents(
        self, file_hashes: dict[str, str]
    ) -> tuple[dict[str, bool], float]:
        """Check which documents already exist in the index.

        Args:
            file_hashes: Mapping of file paths to their hashes

        Returns:
            Tuple of (hash -> exists mapping, time taken)
        """
        check_start = time.time()
        documents_exist = self.index.check_bulk_documents_exist(
            list(file_hashes.values())
        )
        check_time = time.time() - check_start

        print(f"Bulk existence check took: {check_time:.2f}s")

        return documents_exist, check_time

    def _cleanup_documents(
        self, folder_path: Path, current_file_hashes: set[str]
    ) -> float:
        """Clean up documents that no longer exist in the folder.

        Args:
            folder_path: Path to the folder being indexed
            current_file_hashes: Set of hashes for files currently in the folder

        Returns:
            Time taken for cleanup
        """
        cleanup_start = time.time()
        self._cleanup_missing_documents(folder_path, current_file_hashes)
        cleanup_time = time.time() - cleanup_start

        print(f"Cleanup took: {cleanup_time:.2f}s\n")

        return cleanup_time

    def _filter_new_files(
        self, file_hashes: dict[str, str], documents_exist: dict[str, bool]
    ) -> list[Path]:
        """Filter files that don't exist in the index.

        Args:
            file_hashes: Mapping of file paths to their hashes
            documents_exist: Mapping of hashes to existence status

        Returns:
            List of file paths that need to be indexed
        """
        return [
            Path(file_path)
            for file_path, file_hash in file_hashes.items()
            if not documents_exist[file_hash]
        ]

    def _process_and_index_documents(
        self, files: list[Path], file_hashes: dict[str, str], batch_size: int
    ) -> tuple[int, int, float, float, float]:
        """Process documents and index them in batches.

        Args:
            files: List of file paths to process
            file_hashes: Mapping of file paths to their hashes
            batch_size: Number of documents per batch

        Returns:
            Tuple of (indexed_count, skipped_count, parse_time, tokenize_time, index_time)
        """
        documents_batch = []
        indexed_count = 0
        skipped_count = 0
        parse_time = 0
        tokenize_time = 0
        index_time = 0

        for file_path in files:
            try:
                # Parse and tokenize document
                doc_id = file_hashes[str(file_path)]
                term_counts, parse_duration, tokenize_duration = (
                    self._parse_and_tokenize_document(file_path)
                )

                parse_time += parse_duration
                tokenize_time += tokenize_duration

                if term_counts is None:
                    skipped_count += 1
                    continue

                # Add to batch
                documents_batch.append(
                    (doc_id, term_counts, file_path.name, str(file_path))
                )

                # Insert batch when it reaches batch_size
                if len(documents_batch) >= batch_size:
                    batch_index_time = self._index_batch(documents_batch)
                    index_time += batch_index_time
                    indexed_count += len(documents_batch)
                    print(f"Indexed batch: {indexed_count}/{len(files)}")
                    documents_batch = []

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                skipped_count += 1

        # Insert remaining documents
        if documents_batch:
            batch_index_time = self._index_batch(documents_batch)
            index_time += batch_index_time
            indexed_count += len(documents_batch)
            print(f"Indexed final batch: {indexed_count}/{len(files)}")

        return indexed_count, skipped_count, parse_time, tokenize_time, index_time

    def _parse_and_tokenize_document(
        self, file_path: Path
    ) -> tuple[dict[str, int] | None, float, float]:
        """Parse and tokenize a single document.

        Args:
            file_path: Path to the document

        Returns:
            Tuple of (term_counts or None if skipped, parse_time, tokenize_time)
        """
        # Parse PDF
        parse_start = time.time()
        text = self.pdf_doc_parser.parse(file_path)
        parse_time = time.time() - parse_start

        if not text.strip():
            print(f"Skipped empty document: {file_path.name}")
            return None, parse_time, 0

        # Tokenize
        tokenize_start = time.time()
        tokens = self.tokenizer.tokenize(text)
        tokenize_time = time.time() - tokenize_start

        if not tokens:
            print(f"Skipped document with no tokens: {file_path.name}")
            return None, parse_time, tokenize_time

        term_counts = dict(Counter(tokens))

        # Log successful processing
        print(
            f"Processed: {file_path.name} "
            f"(parse: {parse_time*1000:.0f}ms, "
            f"tokenize: {tokenize_time*1000:.0f}ms, "
            f"terms: {len(term_counts)} unique)"
        )

        return term_counts, parse_time, tokenize_time

    def _print_indexing_summary(
        self,
        total_files: int,
        indexed_count: int,
        skipped_count: int,
        hash_time: float,
        check_time: float,
        cleanup_time: float,
        parse_time: float,
        tokenize_time: float,
        index_time: float,
        total_time: float,
    ):
        """Print a summary of the indexing operation.

        Args:
            total_files: Total number of files found
            indexed_count: Number of documents successfully indexed
            skipped_count: Number of documents skipped
            hash_time: Time spent computing hashes
            check_time: Time spent checking existing documents
            cleanup_time: Time spent cleaning up missing documents
            parse_time: Time spent parsing PDFs
            tokenize_time: Time spent tokenizing
            index_time: Time spent indexing
            total_time: Total time for the entire operation
        """
        print("\n" + "=" * 60)
        print("=== Indexing Summary ===")
        print("=" * 60)
        print(f"\nTotal files found: {total_files}")
        print(f"Documents indexed: {indexed_count}")
        print(f"Documents skipped: {skipped_count}")
        print("\n--- Timing Breakdown ---")
        print(
            f"Hash computation:    {hash_time:.2f}s ({hash_time/total_time*100:.1f}%)"
        )
        print(
            f"Existence check:     {check_time:.2f}s ({check_time/total_time*100:.1f}%)"
        )
        print(
            f"Cleanup:             {cleanup_time:.2f}s ({cleanup_time/total_time*100:.1f}%)"
        )

        if indexed_count > 0:
            print(
                f"PDF parsing:         {parse_time:.2f}s ({parse_time/total_time*100:.1f}%)"
            )
            print(
                f"Tokenization:        {tokenize_time:.2f}s ({tokenize_time/total_time*100:.1f}%)"
            )
            print(
                f"Database indexing:   {index_time:.2f}s ({index_time/total_time*100:.1f}%)"
            )
            print("\n--- Per Document Averages ---")
            print(f"Parse time:          {parse_time/indexed_count*1000:.1f}ms")
            print(f"Tokenize time:       {tokenize_time/indexed_count*1000:.1f}ms")
            print(f"Index time:          {index_time/indexed_count*1000:.1f}ms")

        print(f"\nTotal time:          {total_time:.2f}s")
        print(f"Throughput:          {indexed_count/total_time:.1f} docs/sec")
        print("=" * 60)

    def _index_batch(
        self, documents_batch: list[tuple[str, dict[str, int], str, str]]
    ) -> float:
        """Index a batch of documents.

        Args:
            documents_batch: List of (doc_id, term_counts, filename, filepath) tuples

        Returns:
            Time taken to index the batch
        """
        index_start = time.time()
        self.index.add_documents_batch(documents_batch)
        return time.time() - index_start

    def _find_similar_terms_optimized(
        self,
        query_term: str,
        all_terms: list[str],
        threshold: float,
        max_candidates: int = 1000,
    ) -> list[str]:
        """Find terms similar to the query term using optimized fuzzy matching.

        Uses multiple optimization strategies:
        1. Length-based pre-filtering
        2. First character filtering
        3. Limited candidate checking

        Args:
            query_term: The term to find matches for
            all_terms: List of all terms in the index
            threshold: Minimum similarity score (0.0-1.0)
            max_candidates: Maximum number of candidates to check

        Returns:
            List of similar terms
        """
        query_lower = query_term.lower()
        query_len = len(query_lower)

        # Pre-filter candidates based on length difference
        # If threshold is 0.8, terms can't differ by more than ~20% in length
        max_len_diff = int(query_len * (1 - threshold)) + 1

        candidates = []
        for term in all_terms:
            term_len = len(term)
            # Quick length check
            if abs(term_len - query_len) <= max_len_diff:
                # Quick first character check (optional, but helps)
                if query_lower[0] == term[0].lower():
                    candidates.append(term)

        # Limit candidates to avoid excessive computation
        if len(candidates) > max_candidates:
            # Prioritize exact prefix matches
            candidates.sort(
                key=lambda t: (
                    not t.lower().startswith(query_lower[:3]),  # Prefer prefix matches
                    abs(len(t) - query_len),  # Then by length similarity
                )
            )
            candidates = candidates[:max_candidates]

        # Now do actual similarity computation on filtered candidates
        similar_terms = []
        for term in candidates:
            similarity = SequenceMatcher(None, query_lower, term.lower()).ratio()
            if similarity >= threshold:
                similar_terms.append(term)

        return similar_terms

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_fuzzy: bool = True,
        fuzzy_threshold: float = 0.8,
    ) -> list[dict[str, str | float]]:
        """Search for documents using the abstract index interface.

        Args:
            query: Search query string
            top_k: Number of top results to return
            use_fuzzy: Whether to use fuzzy matching for typos
            fuzzy_threshold: Override default fuzzy matching threshold

        Returns:
            List of search results with document metadata and scores
        """
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        query_terms = list(set(query_tokens))
        total_docs = self.index.get_document_count()
        if total_docs == 0:
            return []

        # Apply fuzzy matching if enabled
        if use_fuzzy:
            threshold = fuzzy_threshold or self.fuzzy_threshold
            expanded_terms = self._expand_query_terms_fuzzy(query_terms, threshold)
            query_terms = expanded_terms

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

        # Fetch metadata
        doc_ids = [doc_id for doc_id, _ in sorted_doc_ids]
        documents_metadata = self.index.get_documents_metadata(doc_ids)

        # Check if file paths still exist
        update_files = []
        for doc in documents_metadata:
            if not os.path.exists(doc.file_path):
                update_files.append(doc.doc_id)

        # Update file paths for moved documents
        if update_files:
            self._update_moved_or_renamed_documents(update_files)
            # Fetch updated documents
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

    def _expand_query_terms_fuzzy(
        self, query_terms: list[str], threshold: float
    ) -> list[str]:
        """Expand query terms using optimized fuzzy matching.

        Args:
            query_terms: Original query terms
            threshold: Minimum similarity threshold

        Returns:
            Expanded list of terms including fuzzy matches
        """
        # Get all unique terms from the index
        all_terms = self.index.get_all_terms()

        # Early exit if index is empty
        if not all_terms:
            return query_terms

        expanded_terms = set(query_terms)  # Start with original terms

        for query_term in query_terms:
            # Find similar terms with optimizations
            similar_terms = self._find_similar_terms_optimized(
                query_term, all_terms, threshold
            )
            expanded_terms.update(similar_terms)

        return list(expanded_terms)

    def _update_moved_or_renamed_documents(self, doc_ids: list[str]) -> None:
        """Find and update file paths/file names for documents that have been moved or renamed.

        Args:
            doc_ids: List of document IDs (file hashes) whose files no longer exist at their stored paths
        """
        if not doc_ids:
            return

        updated_count = 0
        not_found_count = 0

        # Get current metadata for documents that need updating
        documents_metadata = self.index.get_documents_metadata(doc_ids)

        # Get all current PDF files in the folder
        current_files = list(self.folder_path.rglob("*.pdf"))

        # Compute hashes for all current files
        print(f"\nSearching for {len(doc_ids)} moved documents...")
        current_file_hashes = {
            compute_file_hash(file_path): file_path for file_path in current_files
        }

        # Try to find each document by its hash
        for doc in documents_metadata:
            if doc.doc_id in current_file_hashes:
                # Found the file in a new location
                new_file_path = current_file_hashes[doc.doc_id]
                try:
                    self.index.update_document_file_path(
                        doc.doc_id, new_file_path.name, str(new_file_path)
                    )
                    updated_count += 1
                    print(f"Updated path for: {doc.filename} -> {new_file_path}")
                except Exception as e:
                    print(f"Failed to update path for {doc.filename}: {e}")
            else:
                # File not found anywhere in the folder
                not_found_count += 1
                print(f"Document not found in folder: {doc.filename}")

        if updated_count > 0:
            print(f"\nUpdated {updated_count} moved document(s)")
        if not_found_count > 0:
            print(f"Could not locate {not_found_count} document(s)")

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
