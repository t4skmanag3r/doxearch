import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget

from doxearch.context_manager import DirectoryContextManager
from doxearch.doxearch import Doxearch
from doxearch.model_manager import ModelManager
from doxearch.utils.app_dir import get_app_data_dir
from doxearch_gui.tabs.directories_tab import DirectoriesTab
from doxearch_gui.tabs.documents_tab import DocumentsTab
from doxearch_gui.tabs.index_tab import IndexTab
from doxearch_gui.tabs.models_tab import ModelsTab
from doxearch_gui.tabs.search_tab import SearchTab


class DoxearchGUI(QMainWindow):
    """Main GUI window for Doxearch."""

    def __init__(self):
        super().__init__()

        # Initialize context manager
        self.app_data_dir = get_app_data_dir()
        self.context_db_path = self.app_data_dir / "context_manager.db"
        self.context_manager = DirectoryContextManager(
            db_path=str(self.context_db_path)
        )

        self.current_doxearch: Optional[Doxearch] = None

        # Initialize model manager
        self.models_dir = self.app_data_dir / "models"
        self.model_manager = ModelManager(models_dir=self.models_dir)

        self.setWindowTitle("Doxearch - Document Search Engine")
        self.setGeometry(100, 100, 1000, 700)

        self.init_ui()
        self.load_active_directory()

    def init_ui(self):
        """Initialize the user interface."""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create tab widget
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Create tabs
        self.search_tab = SearchTab(self)
        self.index_tab = IndexTab(self)
        self.documents_tab = DocumentsTab(self)
        self.directories_tab = DirectoriesTab(self)
        self.models_tab = ModelsTab(self)

        # Add tabs
        tabs.addTab(self.search_tab, "Search")
        tabs.addTab(self.index_tab, "Index")
        tabs.addTab(self.documents_tab, "Documents")
        tabs.addTab(self.directories_tab, "Directories")
        tabs.addTab(self.models_tab, "Models")

    def load_active_directory(self):
        """Load and display the active directory."""
        active_dir = self.context_manager.get_active_directory()

        if active_dir:
            directory_path = active_dir["directory_path"]
            self.current_doxearch = self._create_doxearch_instance(active_dir)

            # Update all tabs
            self.search_tab.update_active_directory(
                directory_path, self.current_doxearch
            )
            self.documents_tab.update_active_directory(
                directory_path, self.current_doxearch
            )
        else:
            self.current_doxearch = None
            self.search_tab.update_active_directory(None, None)
            self.documents_tab.update_active_directory(None, None)

    def _create_doxearch_instance(self, active_dir: dict) -> Optional[Doxearch]:
        """Create a Doxearch instance from directory info."""
        from PyQt6.QtWidgets import QMessageBox

        from doxearch.doc_index.sqlite_index.sqlite_index import SQLiteIndex
        from doxearch.tokenizer.spacy_tokenizer.spacy_tokenizer import SpacyTokenizer

        directory_path = active_dir["directory_path"]
        db_path = active_dir["db_path"]
        model = active_dir["tokenizer_model_name"]
        lemmatization_enabled = active_dir.get("lemmatization_enabled", True)
        stemming_enabled = active_dir.get("stemming_enabled", False)

        try:
            # Check if model is available
            if not self.model_manager.is_model_available(model):
                QMessageBox.warning(
                    self,
                    "Model Not Available",
                    f"The model '{model}' required for the active directory is not available.\n\n"
                    f"Directory: {directory_path}\n\n"
                    f"Please download the model from the Models tab or set a different directory as active.",
                )
                return None

            index = SQLiteIndex(db_path=db_path)

            # Get model path from model manager
            model_path = None
            if self.model_manager.is_model_in_downloads(model):
                model_path = str(self.model_manager.models_dir / model)

            tokenizer = SpacyTokenizer(
                model=model,
                use_lemmatization=lemmatization_enabled,
                use_stemming=stemming_enabled,
                disable=["parser", "ner"],
                model_path=model_path,
            )
            return Doxearch(Path(directory_path), index, tokenizer)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Active Directory",
                f"Failed to load active directory:\n{directory_path}\n\nError: {str(e)}",
            )
            return None

    def refresh_all_tabs(self):
        """Refresh all tabs after changes."""
        self.load_active_directory()
        self.directories_tab.load_directories()
        self.models_tab.load_models_info()
        self.index_tab.populate_model_combo()


def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    window = DoxearchGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
