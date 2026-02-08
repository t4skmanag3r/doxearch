from doxearch.doc_index.sqlite_index.sqlite_index import SQLiteIndex


def main():
    # Initialize the index
    index = SQLiteIndex()

    # Mock data: pre-computed term frequencies
    # Simulating what would come from tokenization + TF computation
    mock_term_frequencies_1 = {
        "sample": 1,
        "document": 1,
        "this": 1,
        "is": 1,
        "a": 1,
        "another": 1,
        "sentence": 1,
        "here": 1,
    }

    mock_term_frequencies_2 = {
        "search": 3,
        "engine": 2,
        "document": 2,
        "indexing": 1,
        "retrieval": 1,
    }

    mock_term_frequencies_3 = {"sample": 2, "text": 1, "document": 1, "search": 1}

    try:
        # Add first document
        print("Adding document 1...")
        index.add_document(
            document_id="doc_1",
            term_frequencies=mock_term_frequencies_1,
            filepath="/mock/path/document1.pdf",
        )
        print(f"✓ Document 1 added. Total documents: {index.get_document_count()}")

        # Add second document
        print("\nAdding document 2...")
        index.add_document(
            document_id="doc_2",
            term_frequencies=mock_term_frequencies_2,
            filepath="/mock/path/document2.pdf",
        )
        print(f"✓ Document 2 added. Total documents: {index.get_document_count()}")

        # Add third document
        print("\nAdding document 3...")
        index.add_document(
            document_id="doc_3",
            term_frequencies=mock_term_frequencies_3,
            filepath="/mock/path/document3.pdf",
        )
        print(f"✓ Document 3 added. Total documents: {index.get_document_count()}")

        # Test duplicate document (should raise error)
        print("\nTrying to add duplicate document...")
        try:
            index.add_document(
                document_id="doc_1",
                term_frequencies=mock_term_frequencies_1,
                filepath="/mock/path/duplicate.pdf",
            )
        except ValueError as e:
            print(f"✓ Duplicate check working: {e}")

        print(f"\n=== Final Statistics ===")
        print(f"Total documents in index: {index.get_document_count()}")

    except Exception as e:
        print(f"✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
