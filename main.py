from doxearch.doc_index.sqlite_index.sqlite_index import SQLiteIndex


def main():
    index = SQLiteIndex()
    index.add_document(1, ["This is a sample document.", "Another sentence here."])
    print(index.get_document_count())


if __name__ == "__main__":
    main()
