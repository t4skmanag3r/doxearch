from pathlib import Path

from pypdf import PdfReader

from doxearch.doc_parser.doc_parser import DocParser


class PDFParser(DocParser):
    def parse(self, file_path: Path) -> str:
        reader = PdfReader(str(file_path))
        text = "\n".join([page.extract_text() for page in reader.pages])
        return text
