
from PyQt6.QtCore import QThread, pyqtSignal

from doxearch.doxearch import Doxearch


class IndexWorker(QThread):
    """Worker thread for indexing operations."""

    progress = pyqtSignal(str)  # status message
    finished = pyqtSignal(int)  # document count
    error = pyqtSignal(str)  # error message

    def __init__(self, doxearch: Doxearch):
        super().__init__()
        self.doxearch = doxearch

    def run(self):
        """Run the indexing operation."""
        try:
            self.progress.emit("Indexing documents...")
            self.doxearch.index_folder()
            doc_count = self.doxearch.index.get_document_count()
            self.finished.emit(doc_count)
        except Exception as e:
            self.error.emit(str(e))
