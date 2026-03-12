
from PyQt6.QtCore import QThread, pyqtSignal

from doxearch.doxearch import Doxearch


class SearchWorker(QThread):
    """Worker thread for performing searches."""

    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(
        self,
        doxearch: Doxearch,
        query: str,
        top_k: int,
        use_fuzzy: bool = True,
    ):
        super().__init__()
        self.doxearch = doxearch
        self.query = query
        self.top_k = top_k
        self.use_fuzzy = use_fuzzy

    def run(self):
        """Run the search operation."""
        try:
            results = self.doxearch.search(
                self.query,
                top_k=self.top_k,
                use_fuzzy=self.use_fuzzy,
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))
