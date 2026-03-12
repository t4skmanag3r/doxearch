
from PyQt6.QtCore import QThread, pyqtSignal


class ModelDownloadWorker(QThread):
    """Worker thread for downloading models."""

    progress = pyqtSignal(int, int, str)  # downloaded, total, message
    finished = pyqtSignal(str, str)  # model_name, message
    error = pyqtSignal(str)

    def __init__(self, model_manager, model_name: str):
        super().__init__()
        self.model_manager = model_manager
        self.model_name = model_name

    def run(self):
        """Download the model."""
        try:

            def progress_callback(downloaded: int, total: int, message: str):
                self.progress.emit(downloaded, total, message)

            success, message = self.model_manager.download_model(
                self.model_name, progress_callback=progress_callback
            )

            if success:
                self.finished.emit(self.model_name, message)
            else:
                self.error.emit(message)

        except Exception as e:
            self.error.emit(str(e))
