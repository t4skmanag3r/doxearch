from pathlib import Path

from docx import Document

from doxearch.doc_parser.doc_parser import DocParser


class DocxParser(DocParser):
    def parse(self, file_path: Path) -> str:
        doc = Document(str(file_path))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
