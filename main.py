from pathlib import Path

from doxearch.doc_index.sqlite_index.sqlite_index import SQLiteIndex
from doxearch.doxearch import Doxearch, get_app_data_dir
from doxearch.tokenizer.spacy_tokenizer.spacy_tokenizer import SpacyTokenizer


def main():
    print("=== Doxearch - Document Indexing Test ===\n")

    # Initialize Doxearch
    print("Initializing Doxearch...")
    try:
        app_data_dir = get_app_data_dir()
        db_path = app_data_dir / "doxearch.db"
        index = SQLiteIndex(db_path=str(db_path))
        tokenizer = SpacyTokenizer(model="lt_core_news_sm")
        doxearch = Doxearch(index, tokenizer)
        print("✓ Doxearch initialized successfully")
        print(f"✓ Database location: {doxearch.index.engine.url}")
    except Exception as e:
        print(f"✗ Error initializing Doxearch: {e}")
        return

    # Check current document count
    doc_count = doxearch.index.get_document_count()
    print(f"✓ Current documents in index: {doc_count}\n")

    # Test folder path - adjust this to your test documents location
    test_folder = Path("/run/media/eleos/CORSAIR/Isakymai/")

    # Check if test folder exists
    if not test_folder.exists():
        print(f"✗ Test folder does not exist: {test_folder}")
        print(
            "\nPlease create a test folder with some PDF files, or update the path in main.py"
        )
        print("Example:")
        print(f"  mkdir -p {test_folder}")
        print(f"  # Add some PDF files to {test_folder}")
        return

    # Check if folder has PDF files
    pdf_files = list(test_folder.rglob("*.pdf"))
    if not pdf_files:
        print(f"✗ No PDF files found in: {test_folder}")
        print("\nPlease add some PDF files to the test folder")
        return

    print(f"Found {len(pdf_files)} PDF file(s) in {test_folder}")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    print()

    # Index the folder
    print("=== Starting Indexing Process ===\n")
    try:
        doxearch.index_folder(test_folder)
        print("\n✓ Indexing completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during indexing: {e}")
        import traceback

        traceback.print_exc()
        return

    # Show final statistics
    print("\n=== Index Statistics ===")
    final_count = doxearch.index.get_document_count()
    print(f"Total documents indexed: {final_count}")

    # Show some sample documents from the index
    print("\n=== Sample Indexed Documents ===")
    with doxearch.index.get_session() as session:
        from doxearch.doc_index.sqlite_index.sqlite_index import Document

        documents = session.query(Document).limit(5).all()
        for doc in documents:
            print(f"\nDocument ID: {doc.doc_id}")
            print(f"  File: {Path(doc.file_path).name}")
            print(f"  Terms: {doc.term_count} total, {doc.unique_terms} unique")
            print(f"  Indexed: {doc.last_indexed}")

    # Interactive search mode
    print("\n" + "=" * 60)
    print("=== Interactive Search Mode ===")
    print("=" * 60)
    print("Enter your search queries (or 'quit' to exit)\n")

    while True:
        try:
            user_query = input("Search> ").strip()

            if user_query.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            if not user_query:
                continue

            results = doxearch.search(user_query, top_k=10)

            if not results:
                print("  No results found.\n")
            else:
                print(f"\n  Found {len(results)} result(s):\n")
                for rank, (doc_id, score) in enumerate(results, 1):
                    filename = Path(doc_id).name
                    print(f"  {rank}. {filename} (score: {score:.4f})")
                print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"  ✗ Error: {e}\n")


if __name__ == "__main__":
    main()
