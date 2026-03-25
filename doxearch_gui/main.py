import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from doxearch.context_manager import DirectoryContextManager
from doxearch.doc_index.sqlite_index.sqlite_index import Document, SQLiteIndex
from doxearch.doxearch import Doxearch
from doxearch.exceptions import DirectoryAlreadyIndexedError, DirectoryNotFoundError
from doxearch.model_manager import ModelManager
from doxearch.tokenizer.spacy_tokenizer.spacy_tokenizer import SpacyTokenizer
from doxearch.utils.app_dir import get_app_data_dir
from doxearch.utils.general import get_db_path_for_directory


class IndexWorker(QThread):
    """Worker thread for indexing operations."""

    progress = pyqtSignal(str)  # status message
    finished = pyqtSignal(int)  # document count
    error = pyqtSignal(str)  # error message

    def __init__(self, doxearch: Doxearch):
        super().__init__()
        self.doxearch = doxearch

    def run(self):
        try:
            self.progress.emit("Indexing documents...")
            self.doxearch.index_folder()
            doc_count = self.doxearch.index.get_document_count()
            self.finished.emit(doc_count)
        except Exception as e:
            self.error.emit(str(e))


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
        self.index_worker: Optional[IndexWorker] = None
        self.search_worker: Optional[SearchWorker] = None
        self.model_download_worker: Optional[ModelDownloadWorker] = None

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

        # Add tabs
        tabs.addTab(self.create_search_tab(), "Search")
        tabs.addTab(self.create_index_tab(), "Index")
        tabs.addTab(self.create_documents_tab(), "Documents")
        tabs.addTab(self.create_directories_tab(), "Directories")
        tabs.addTab(self.create_models_tab(), "Models")

    def create_models_tab(self) -> QWidget:
        """Create the models management tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Info section
        info_label = QLabel(
            "Manage spaCy language models used for document processing. "
            "Models are required for indexing and searching documents."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(info_label)

        # Models directory info
        models_dir_layout = QHBoxLayout()
        models_dir_layout.addWidget(QLabel("Models Directory:"))
        self.models_dir_label = QLabel()
        self.models_dir_label.setStyleSheet("font-family: monospace;")
        models_dir_layout.addWidget(self.models_dir_label)

        models_dir_layout.addStretch()
        layout.addLayout(models_dir_layout)

        # Action buttons layout
        buttons_layout = QHBoxLayout()

        # Fetch button
        self.fetch_models_button = QPushButton("Fetch Models")
        self.fetch_models_button.clicked.connect(self.fetch_models)
        buttons_layout.addWidget(self.fetch_models_button)

        # Refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.load_models_info)
        buttons_layout.addWidget(refresh_button)

        buttons_layout.addStretch()

        # Storage info
        self.storage_label = QLabel("Total storage used: Calculating...")
        buttons_layout.addWidget(self.storage_label)

        layout.addLayout(buttons_layout)

        # Models table
        self.models_table = QTableWidget()
        self.models_table.setColumnCount(6)
        self.models_table.setHorizontalHeaderLabels(
            ["Model Name", "Language", "Version", "Size (MB)", "Status", "Description"]
        )
        self.models_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.models_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.models_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.models_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.models_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        self.models_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch
        )
        self.models_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.models_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.models_table)

        # Action buttons
        button_layout = QHBoxLayout()

        self.download_model_button = QPushButton("Download Selected")
        self.download_model_button.clicked.connect(self.download_selected_model)
        button_layout.addWidget(self.download_model_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Progress section
        self.model_progress_label = QLabel()
        self.model_progress_label.setVisible(False)
        layout.addWidget(self.model_progress_label)

        self.model_progress_bar = QProgressBar()
        self.model_progress_bar.setVisible(False)
        layout.addWidget(self.model_progress_bar)

        # Load models info
        self.load_models_info()

        return tab

    def fetch_models(self):
        """Fetch available models from spaCy compatibility.json."""
        self.fetch_models_button.setEnabled(False)
        self.fetch_models_button.setText("Fetching...")

        try:
            # Fetch models
            self.model_manager.MODEL_URLS = self.model_manager.fetch_available_models()

            # Refresh the table with newly fetched models
            self.load_models_info()

            models_count = len(self.model_manager.MODEL_URLS)
            QMessageBox.information(
                self,
                "Fetch Complete",
                f"Successfully fetched {models_count} available models from spaCy.",
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Fetch Error",
                f"Failed to fetch available models:\n{str(e)}",
            )
        finally:
            self.fetch_models_button.setEnabled(True)
            self.fetch_models_button.setText("Fetch Available Models")

    def create_documents_tab(self) -> QWidget:
        """Create the documents tab showing all indexed documents."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Active directory info
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("Active Directory:"))
        self.documents_active_dir_label = QLabel("None")
        self.documents_active_dir_label.setStyleSheet(
            "font-weight: bold; color: #2196F3;"
        )
        info_layout.addWidget(self.documents_active_dir_label)
        info_layout.addStretch()
        layout.addLayout(info_layout)

        # Refresh button
        refresh_layout = QHBoxLayout()
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.load_documents)
        refresh_layout.addWidget(refresh_button)

        # Document count label
        self.document_count_label = QLabel("Total documents: 0")
        refresh_layout.addWidget(self.document_count_label)
        refresh_layout.addStretch()
        layout.addLayout(refresh_layout)

        # Documents table
        self.documents_table = QTableWidget()
        self.documents_table.setColumnCount(5)
        self.documents_table.setHorizontalHeaderLabels(
            ["Filename", "Terms", "Unique Terms", "Last Indexed", "Path"]
        )
        self.documents_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.documents_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.documents_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.documents_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.documents_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Stretch
        )
        self.documents_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.documents_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.documents_table.setSortingEnabled(True)
        self.documents_table.cellDoubleClicked.connect(self.open_document_file)
        layout.addWidget(self.documents_table)

        # Action buttons
        button_layout = QHBoxLayout()

        open_file_button = QPushButton("Open File")
        open_file_button.clicked.connect(self.open_selected_document_file)
        button_layout.addWidget(open_file_button)

        open_folder_button = QPushButton("Open Folder")
        open_folder_button.clicked.connect(self.open_selected_document_folder)
        button_layout.addWidget(open_folder_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Load documents
        self.load_documents()

        return tab

    def load_documents(self):
        """Load and display all documents from the active index."""
        if not self.current_doxearch:
            self.documents_active_dir_label.setText("None")
            self.document_count_label.setText("Total documents: 0")
            self.documents_table.setRowCount(0)
            return

        # Update active directory label
        active_dir = self.context_manager.get_active_directory()
        if active_dir:
            self.documents_active_dir_label.setText(active_dir["directory_path"])
        else:
            self.documents_active_dir_label.setText("None")

        # Get all documents from the index using the new method
        try:
            documents = self.current_doxearch.index.get_all_documents()

            # Disable sorting while populating
            self.documents_table.setSortingEnabled(False)

            self.documents_table.setRowCount(len(documents))
            self.document_count_label.setText(f"Total documents: {len(documents)}")

            for row, doc in enumerate(documents):
                # Filename
                filename_item = QTableWidgetItem(doc.filename)
                self.documents_table.setItem(row, 0, filename_item)

                # Term count
                term_count_item = QTableWidgetItem(str(doc.term_count))
                term_count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.documents_table.setItem(row, 1, term_count_item)

                # Unique terms
                unique_terms_item = QTableWidgetItem(str(doc.unique_terms))
                unique_terms_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.documents_table.setItem(row, 2, unique_terms_item)

                # Last indexed (convert timestamp to readable format)
                last_indexed_dt = datetime.fromtimestamp(doc.last_indexed)
                last_indexed_str = last_indexed_dt.strftime("%Y-%m-%d %H:%M:%S")
                last_indexed_item = QTableWidgetItem(last_indexed_str)
                self.documents_table.setItem(row, 3, last_indexed_item)

                # File path
                path_item = QTableWidgetItem(doc.file_path)
                path_item.setToolTip("Double-click to open file")
                self.documents_table.setItem(row, 4, path_item)

            # Re-enable sorting after populating
            self.documents_table.setSortingEnabled(True)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Documents",
                f"An error occurred while loading documents:\n{e}",
            )

    def open_document_file(self, row: int, column: int):
        """Open the file from the selected document row."""
        if self.documents_table.rowCount() == 0:
            return

        filepath_item = self.documents_table.item(row, 4)
        if filepath_item:
            filepath = filepath_item.text()
            self.open_file(filepath)

    def open_selected_document_file(self):
        """Open the currently selected document file."""
        selected_row = self.documents_table.currentRow()
        if selected_row < 0:
            QMessageBox.warning(
                self, "No Selection", "Please select a document to open."
            )
            return

        filepath_item = self.documents_table.item(selected_row, 4)
        if filepath_item:
            filepath = filepath_item.text()
            self.open_file(filepath)

    def open_selected_document_folder(self):
        """Open the folder containing the currently selected document."""
        selected_row = self.documents_table.currentRow()
        if selected_row < 0:
            QMessageBox.warning(
                self, "No Selection", "Please select a document to open its folder."
            )
            return

        filepath_item = self.documents_table.item(selected_row, 4)
        if filepath_item:
            filepath = filepath_item.text()
            folder_path = str(Path(filepath).parent)

            if not Path(folder_path).exists():
                QMessageBox.warning(
                    self,
                    "Folder Not Found",
                    f"The folder does not exist:\n{folder_path}",
                )
                return

            try:
                if sys.platform == "win32":
                    os.startfile(folder_path)
                elif sys.platform == "darwin":  # macOS
                    subprocess.run(["open", folder_path])
                else:  # Linux and other Unix-like
                    subprocess.run(["xdg-open", folder_path])
            except Exception as e:
                QMessageBox.critical(
                    self, "Error Opening Folder", f"Could not open folder:\n{e}"
                )

    def create_search_tab(self) -> QWidget:
        """Create the search tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Active directory info
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("Active Directory:"))
        self.active_dir_label = QLabel("None")
        self.active_dir_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        info_layout.addWidget(self.active_dir_label)
        info_layout.addStretch()
        layout.addLayout(info_layout)

        # Search input
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Query:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search query...")
        self.search_input.returnPressed.connect(self.perform_search)
        search_layout.addWidget(self.search_input)

        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.perform_search)
        search_layout.addWidget(self.search_button)

        layout.addLayout(search_layout)

        # Search options layout
        options_layout = QHBoxLayout()

        # Top K selector
        options_layout.addWidget(QLabel("Number of results:"))
        self.topk_spinbox = QSpinBox()
        self.topk_spinbox.setMinimum(1)
        self.topk_spinbox.setMaximum(100)
        self.topk_spinbox.setValue(10)
        options_layout.addWidget(self.topk_spinbox)

        options_layout.addSpacing(20)

        # Fuzzy search checkbox
        self.fuzzy_checkbox = QCheckBox("Enable fuzzy matching")
        self.fuzzy_checkbox.setChecked(True)
        self.fuzzy_checkbox.setToolTip(
            "Enable fuzzy matching to find similar terms (helps with typos)"
        )
        options_layout.addWidget(self.fuzzy_checkbox)

        options_layout.addStretch()
        layout.addLayout(options_layout)

        # Results area
        layout.addWidget(QLabel("Results:"))
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Filename", "Score", "Path"])
        self.results_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.results_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.cellDoubleClicked.connect(self.open_result_file)
        layout.addWidget(self.results_table)

        # Action buttons for results
        result_buttons_layout = QHBoxLayout()
        open_file_button = QPushButton("Open File")
        open_file_button.clicked.connect(self.open_selected_file)
        result_buttons_layout.addWidget(open_file_button)

        open_folder_button = QPushButton("Open Folder")
        open_folder_button.clicked.connect(self.open_selected_folder)
        result_buttons_layout.addWidget(open_folder_button)

        result_buttons_layout.addStretch()
        layout.addLayout(result_buttons_layout)

        return tab

    def perform_search(self):
        """Perform a search query."""
        if not self.current_doxearch:
            QMessageBox.warning(
                self,
                "No Active Directory",
                "Please set an active directory or index a folder first.",
            )
            return

        query = self.search_input.text().strip()

        if not query:
            QMessageBox.warning(self, "Empty Query", "Please enter a search query.")
            return

        top_k = self.topk_spinbox.value()
        use_fuzzy = self.fuzzy_checkbox.isChecked()

        # Disable search during operation
        self.search_button.setEnabled(False)
        self.results_table.setRowCount(0)

        # First, reindex to catch any changes
        try:
            # Run index_folder to update the index with any new/modified files
            # This is efficient as it only processes new or changed files
            self.current_doxearch.index_folder()
        except Exception as e:
            self.search_button.setEnabled(True)
            QMessageBox.critical(
                self,
                "Indexing Error",
                f"An error occurred while updating the index:\n{e}",
            )
            return

        # Start search in worker thread
        self.search_worker = SearchWorker(
            self.current_doxearch, query, top_k, use_fuzzy
        )
        self.search_worker.finished.connect(self.on_search_finished)
        self.search_worker.error.connect(self.on_search_error)
        self.search_worker.start()

    def on_search_finished(self, results: list):
        """Handle search completion."""
        self.search_button.setEnabled(True)
        self.results_table.setRowCount(0)

        if not results:
            self.results_table.setRowCount(1)
            no_results_item = QTableWidgetItem("No results found")
            no_results_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(0, 0, no_results_item)
            self.results_table.setSpan(0, 0, 1, 3)
            return

        self.results_table.setRowCount(len(results))

        for row, result in enumerate(results):
            # Handle both dict and object results
            if isinstance(result, dict):
                filename = result.get("filename", "Unknown")
                score = result.get("score", 0.0)
                filepath = result.get("filepath", "Unknown")
            else:
                filename = result.filename
                score = result.score
                filepath = result.filepath

            # Filename
            filename_item = QTableWidgetItem(filename)
            self.results_table.setItem(row, 0, filename_item)

            # Score
            score_item = QTableWidgetItem(f"{score:.4f}")
            self.results_table.setItem(row, 1, score_item)

            # Path (clickable)
            path_item = QTableWidgetItem(filepath)
            path_item.setToolTip("Double-click to open file")
            self.results_table.setItem(row, 2, path_item)

    def open_result_file(self, row: int, column: int):
        """Open the file from the selected result row."""
        if self.results_table.rowCount() == 0:
            return

        filepath_item = self.results_table.item(row, 2)
        if filepath_item:
            filepath = filepath_item.text()
            self.open_file(filepath)

    def open_selected_file(self):
        """Open the currently selected file."""
        selected_row = self.results_table.currentRow()
        if selected_row < 0:
            QMessageBox.warning(self, "No Selection", "Please select a result to open.")
            return

        filepath_item = self.results_table.item(selected_row, 2)
        if filepath_item:
            filepath = filepath_item.text()
            self.open_file(filepath)

    def open_selected_folder(self):
        """Open the folder containing the currently selected file."""
        selected_row = self.results_table.currentRow()
        if selected_row < 0:
            QMessageBox.warning(
                self, "No Selection", "Please select a result to open its folder."
            )
            return

        filepath_item = self.results_table.item(selected_row, 2)
        if filepath_item:
            filepath = filepath_item.text()
            folder_path = str(Path(filepath).parent)

            # Open the folder directly
            if not Path(folder_path).exists():
                QMessageBox.warning(
                    self,
                    "Folder Not Found",
                    f"The folder does not exist:\n{folder_path}",
                )
                return

            try:
                if sys.platform == "win32":
                    os.startfile(folder_path)
                elif sys.platform == "darwin":  # macOS
                    subprocess.run(["open", folder_path])
                else:  # Linux and other Unix-like
                    subprocess.run(["xdg-open", folder_path])
            except Exception as e:
                QMessageBox.critical(
                    self, "Error Opening Folder", f"Could not open folder:\n{e}"
                )

    def open_file(self, filepath: str):
        """Open a file or folder using the system's default application."""
        path = Path(filepath)

        if not path.exists():
            QMessageBox.warning(
                self,
                "File Not Found",
                f"The file or folder does not exist:\n{filepath}",
            )
            return

        try:
            if sys.platform == "win32":
                os.startfile(filepath)
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", filepath])
            else:  # Linux and other Unix-like
                subprocess.run(["xdg-open", filepath])
        except Exception as e:
            QMessageBox.critical(
                self, "Error Opening File", f"Could not open file:\n{e}"
            )

    def create_index_tab(self) -> QWidget:
        """Create the index tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Folder selection
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Folder:"))
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Select folder to index...")
        folder_layout.addWidget(self.folder_input)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_folder)
        folder_layout.addWidget(browse_button)

        layout.addLayout(folder_layout)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        # Only add available models from ModelManager
        available_models = [
            model_info["name"]
            for model_info in self.model_manager.get_all_models_info()
            if self.model_manager.is_model_available(model_info["name"])
        ]
        if available_models:
            self.model_combo.addItems(available_models)
        else:
            self.model_combo.addItem("No models available")
            self.model_combo.setEnabled(False)
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        layout.addLayout(model_layout)

        # Tokenization options
        tokenization_label = QLabel("Tokenization Options:")
        tokenization_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(tokenization_label)

        # Lemmatization checkbox
        self.lemmatization_checkbox = QCheckBox("Enable Lemmatization")
        self.lemmatization_checkbox.setChecked(True)
        self.lemmatization_checkbox.setToolTip(
            "Convert words to their base form (e.g., 'running' → 'run')"
        )
        self.lemmatization_checkbox.stateChanged.connect(
            self.on_tokenization_option_changed
        )
        layout.addWidget(self.lemmatization_checkbox)

        # Stemming checkbox
        self.stemming_checkbox = QCheckBox("Enable Stemming")
        self.stemming_checkbox.setChecked(False)
        self.stemming_checkbox.setToolTip(
            "Apply stemming algorithm to reduce words to their root form"
        )
        self.stemming_checkbox.stateChanged.connect(self.on_tokenization_option_changed)
        layout.addWidget(self.stemming_checkbox)

        # Warning label for XOR constraint
        self.tokenization_warning_label = QLabel()
        self.tokenization_warning_label.setStyleSheet(
            "color: #FF9800; font-style: italic;"
        )
        self.tokenization_warning_label.setWordWrap(True)
        self.tokenization_warning_label.setVisible(False)
        layout.addWidget(self.tokenization_warning_label)

        # Force re-index checkbox
        self.force_checkbox = QCheckBox("Force re-indexing")
        layout.addWidget(self.force_checkbox)

        # Index button
        self.index_button = QPushButton("Start Indexing")
        self.index_button.clicked.connect(self.start_indexing)
        layout.addWidget(self.index_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status area
        layout.addWidget(QLabel("Status:"))
        self.index_status_text = QTextEdit()
        self.index_status_text.setReadOnly(True)
        layout.addWidget(self.index_status_text)

        layout.addStretch()

        return tab

    def on_tokenization_option_changed(self):
        """Handle changes to tokenization options (enforce XOR constraint)."""
        lemmatization_enabled = self.lemmatization_checkbox.isChecked()
        stemming_enabled = self.stemming_checkbox.isChecked()

        # If both are checked, show warning and uncheck the other one
        if lemmatization_enabled and stemming_enabled:
            # Determine which one was just checked
            sender = self.sender()
            if sender == self.lemmatization_checkbox:
                # Lemmatization was just checked, uncheck stemming
                self.stemming_checkbox.blockSignals(True)
                self.stemming_checkbox.setChecked(False)
                self.stemming_checkbox.blockSignals(False)
                self.tokenization_warning_label.setText(
                    "⚠ Lemmatization and stemming are mutually exclusive. Stemming has been disabled."
                )
            else:
                # Stemming was just checked, uncheck lemmatization
                self.lemmatization_checkbox.blockSignals(True)
                self.lemmatization_checkbox.setChecked(False)
                self.lemmatization_checkbox.blockSignals(False)
                self.tokenization_warning_label.setText(
                    "⚠ Lemmatization and stemming are mutually exclusive. Lemmatization has been disabled."
                )
            self.tokenization_warning_label.setVisible(True)
        else:
            self.tokenization_warning_label.setVisible(False)

    def create_directories_tab(self) -> QWidget:
        """Create the directories management tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.load_directories)
        layout.addWidget(refresh_button)

        # Directories table
        self.directories_table = QTableWidget()
        self.directories_table.setColumnCount(7)
        self.directories_table.setHorizontalHeaderLabels(
            [
                "Status",
                "Directory",
                "Model",
                "Version",
                "Lemmatization",
                "Stemming",
                "Database",
            ]
        )
        self.directories_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.directories_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.directories_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.directories_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.directories_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        self.directories_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.ResizeToContents
        )
        self.directories_table.horizontalHeader().setSectionResizeMode(
            6, QHeaderView.ResizeMode.ResizeToContents
        )
        layout.addWidget(self.directories_table)

        # Action buttons
        button_layout = QHBoxLayout()

        set_active_button = QPushButton("Set Active")
        set_active_button.clicked.connect(self.set_active_directory)
        button_layout.addWidget(set_active_button)

        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self.remove_directory)
        button_layout.addWidget(remove_button)

        button_layout.addStretch()

        open_app_dir_button = QPushButton("Open App Directory")
        open_app_dir_button.clicked.connect(self.open_app_directory)
        button_layout.addWidget(open_app_dir_button)

        layout.addLayout(button_layout)

        # Load directories
        self.load_directories()

        return tab

    def load_active_directory(self):
        """Load and display the active directory."""
        active_dir = self.context_manager.get_active_directory()

        if active_dir:
            directory_path = active_dir["directory_path"]

            # Update all active directory labels
            self.active_dir_label.setText(directory_path)
            self.documents_active_dir_label.setText(directory_path)

            # Initialize Doxearch instance for active directory
            folder = Path(directory_path)
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
                    self.active_dir_label.setText(
                        f"{directory_path} (⚠ Model unavailable)"
                    )
                    self.documents_active_dir_label.setText(
                        f"{directory_path} (⚠ Model unavailable)"
                    )
                    self.current_doxearch = None
                    self.documents_table.setRowCount(0)
                    self.document_count_label.setText("Total documents: 0")
                    return

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
                self.current_doxearch = Doxearch(folder, index, tokenizer)

                # Load documents for the active directory
                self.load_documents()
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Loading Active Directory",
                    f"Failed to load active directory:\n{directory_path}\n\nError: {str(e)}",
                )
                self.active_dir_label.setText(f"{directory_path} (⚠ Error)")
                self.documents_active_dir_label.setText(f"{directory_path} (⚠ Error)")
                self.current_doxearch = None
                self.documents_table.setRowCount(0)
                self.document_count_label.setText("Total documents: 0")
        else:
            self.active_dir_label.setText("None")
            self.documents_active_dir_label.setText("None")
            self.current_doxearch = None

            # Clear documents table
            self.documents_table.setRowCount(0)
            self.document_count_label.setText("Total documents: 0")

    def browse_folder(self):
        """Open folder browser dialog."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder to Index", str(Path.home())
        )

        if folder:
            self.folder_input.setText(folder)

    def start_indexing(self):
        """Start the indexing process."""
        folder_path = self.folder_input.text().strip()

        if not folder_path:
            QMessageBox.warning(
                self, "No Folder Selected", "Please select a folder to index."
            )
            return

        folder = Path(folder_path)

        if not folder.exists() or not folder.is_dir():
            QMessageBox.warning(
                self,
                "Invalid Folder",
                "The selected folder does not exist or is not a directory.",
            )
            return

        model = self.model_combo.currentText()
        force = self.force_checkbox.isChecked()
        lemmatization_enabled = self.lemmatization_checkbox.isChecked()
        stemming_enabled = self.stemming_checkbox.isChecked()

        # Validate tokenization options
        if lemmatization_enabled and stemming_enabled:
            QMessageBox.warning(
                self,
                "Invalid Configuration",
                "Lemmatization and stemming cannot be enabled at the same time. Please choose one or neither.",
            )
            return

        # Prepare database path
        folder_str = str(folder.resolve())
        indexes_dir = self.app_data_dir / "indexes"
        indexes_dir.mkdir(parents=True, exist_ok=True)
        db_path = get_db_path_for_directory(folder_str, self.app_data_dir)

        # Check if directory is already indexed
        is_already_indexed = False
        existing_dir_info = None
        try:
            self.context_manager.add_indexed_directory(
                folder_str,
                str(db_path),
                model,
                model_version=None,
                lemmatization_enabled=lemmatization_enabled,
                stemming_enabled=stemming_enabled,
            )
            self.index_status_text.append(f"✓ Registered directory: {folder_str}")
            self.index_status_text.append(f"  - Model: {model}")
            self.index_status_text.append(
                f"  - Lemmatization: {'Enabled' if lemmatization_enabled else 'Disabled'}"
            )
            self.index_status_text.append(
                f"  - Stemming: {'Enabled' if stemming_enabled else 'Disabled'}"
            )
        except DirectoryAlreadyIndexedError:
            is_already_indexed = True
            existing_dir_info = self.context_manager.get_directory_info(folder_str)

            if force:
                # Force re-indexing: wipe and rebuild with new settings
                try:
                    self.index_status_text.append(
                        f"🔄 Force re-indexing enabled - wiping existing index..."
                    )

                    # Remove the directory from context manager
                    self.context_manager.remove_indexed_directory(folder_str)

                    # Clear the data if the database exists
                    if db_path.exists():
                        temp_index = SQLiteIndex(str(db_path))
                        temp_index.clear_all_data()
                        temp_index.close()

                    # Re-register the directory with new settings
                    self.context_manager.add_indexed_directory(
                        folder_str,
                        str(db_path),
                        model,
                        model_version=None,
                        lemmatization_enabled=lemmatization_enabled,
                        stemming_enabled=stemming_enabled,
                    )
                    self.index_status_text.append(
                        f"✓ Re-registered directory with new settings"
                    )
                    self.index_status_text.append(f"  - Model: {model}")
                    self.index_status_text.append(
                        f"  - Lemmatization: {'Enabled' if lemmatization_enabled else 'Disabled'}"
                    )
                    self.index_status_text.append(
                        f"  - Stemming: {'Enabled' if stemming_enabled else 'Disabled'}"
                    )
                except Exception as e:
                    self.index_status_text.append(f"✗ Error wiping index: {e}")
                    QMessageBox.critical(
                        self,
                        "Error",
                        f"Failed to wipe existing index:\n{e}",
                    )
                    return
            else:
                # Normal re-indexing: update with existing settings (don't modify context record)
                self.index_status_text.append(
                    f"⚠ Directory already indexed - updating index with existing settings..."
                )
                self.index_status_text.append(
                    f"  - Model: {existing_dir_info['tokenizer_model_name']}"
                )
                self.index_status_text.append(
                    f"  - Lemmatization: {'Enabled' if existing_dir_info.get('lemmatization_enabled', True) else 'Disabled'}"
                )
                self.index_status_text.append(
                    f"  - Stemming: {'Enabled' if existing_dir_info.get('stemming_enabled', False) else 'Disabled'}"
                )
                self.index_status_text.append(
                    f"💡 Tip: Check 'Force re-indexing' to rebuild with new settings"
                )

                # Use existing settings for indexing (don't change context record)
                model = existing_dir_info["tokenizer_model_name"]
                lemmatization_enabled = existing_dir_info.get(
                    "lemmatization_enabled", True
                )
                stemming_enabled = existing_dir_info.get("stemming_enabled", False)
                db_path = Path(existing_dir_info["db_path"])

        # Create Doxearch instance
        index = SQLiteIndex(db_path=str(db_path))

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
        doxearch = Doxearch(folder, index, tokenizer)

        # Disable UI during indexing
        self.index_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

        # Start indexing in worker thread
        self.index_worker = IndexWorker(doxearch)
        self.index_worker.progress.connect(self.on_index_progress)
        self.index_worker.finished.connect(self.on_index_finished)
        self.index_worker.error.connect(self.on_index_error)
        self.index_worker.start()

    def on_index_progress(self, message: str):
        """Handle indexing progress updates."""
        self.index_status_text.append(message)

    def on_index_finished(self, doc_count: int):
        """Handle indexing completion."""
        self.index_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        self.index_status_text.append(f"✓ Indexing completed!")
        self.index_status_text.append(f"✓ Total documents indexed: {doc_count}")

        # Reload active directory and directories list
        self.load_active_directory()
        self.load_directories()

        QMessageBox.information(
            self, "Indexing Complete", f"Successfully indexed {doc_count} documents."
        )

    def on_index_error(self, error_message: str):
        """Handle indexing errors."""
        self.index_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        self.index_status_text.append(f"✗ Error: {error_message}")

        QMessageBox.critical(
            self,
            "Indexing Error",
            f"An error occurred during indexing:\n{error_message}",
        )

    def on_search_error(self, error_message: str):
        """Handle search errors."""
        self.search_button.setEnabled(True)
        self.results_table.clear()

        QMessageBox.critical(
            self, "Search Error", f"An error occurred during search:\n{error_message}"
        )

    def load_directories(self):
        """Load and display all indexed directories."""
        directories = self.context_manager.get_all_directories()

        # Update table to include tokenization options columns
        self.directories_table.setColumnCount(7)
        self.directories_table.setHorizontalHeaderLabels(
            [
                "Status",
                "Directory",
                "Model",
                "Version",
                "Lemmatization",
                "Stemming",
                "Database",
            ]
        )
        self.directories_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )

        self.directories_table.setRowCount(len(directories))

        for row, directory in enumerate(directories):
            # Status
            status = "✓ Active" if directory["is_active"] else "Inactive"
            status_item = QTableWidgetItem(status)
            if directory["is_active"]:
                font = QFont()
                font.setBold(True)
                status_item.setFont(font)
            self.directories_table.setItem(row, 0, status_item)

            # Directory path
            self.directories_table.setItem(
                row, 1, QTableWidgetItem(directory["directory_path"])
            )

            # Model name
            self.directories_table.setItem(
                row, 2, QTableWidgetItem(directory["tokenizer_model_name"])
            )

            # Model version
            version = directory["tokenizer_model_version"] or "N/A"
            self.directories_table.setItem(row, 3, QTableWidgetItem(version))

            # Lemmatization status
            lemmatization_enabled = directory.get("lemmatization_enabled", True)
            lemmatization_text = "✓ Enabled" if lemmatization_enabled else "Disabled"
            lemmatization_item = QTableWidgetItem(lemmatization_text)
            lemmatization_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.directories_table.setItem(row, 4, lemmatization_item)

            # Stemming status
            stemming_enabled = directory.get("stemming_enabled", False)
            stemming_text = "✓ Enabled" if stemming_enabled else "Disabled"
            stemming_item = QTableWidgetItem(stemming_text)
            stemming_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.directories_table.setItem(row, 5, stemming_item)

            # Database path
            db_name = Path(directory["db_path"]).name
            self.directories_table.setItem(row, 6, QTableWidgetItem(db_name))

    def set_active_directory(self):
        """Set the selected directory as active."""
        selected_row = self.directories_table.currentRow()

        if selected_row < 0:
            QMessageBox.warning(
                self, "No Selection", "Please select a directory to set as active."
            )
            return

        directory_path = self.directories_table.item(selected_row, 1).text()

        try:
            self.context_manager.set_active_directory(directory_path)
            self.load_active_directory()
            self.load_directories()

            QMessageBox.information(
                self, "Success", f"Set active directory to:\n{directory_path}"
            )
        except DirectoryNotFoundError as e:
            QMessageBox.critical(self, "Error", f"Directory not found:\n{e}")

    def remove_directory(self):
        """Remove the selected directory from the index."""
        selected_row = self.directories_table.currentRow()

        if selected_row < 0:
            QMessageBox.warning(
                self, "No Selection", "Please select a directory to remove."
            )
            return

        directory_path = self.directories_table.item(selected_row, 1).text()

        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to remove this directory from the index?\n\n{directory_path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.No:
            return

        try:
            # Get directory info before removing
            dir_info = self.context_manager.get_directory_info(directory_path)

            # Remove from context manager
            self.context_manager.remove_indexed_directory(directory_path)

            # Ask if user wants to delete the database file
            if dir_info:
                db_path = Path(dir_info["db_path"])
                if db_path.exists():
                    delete_db_reply = QMessageBox.question(
                        self,
                        "Delete Database",
                        f"Do you also want to delete the database file?\n\n{db_path}",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )

                    if delete_db_reply == QMessageBox.StandardButton.Yes:
                        db_path.unlink()

            # Reload directories and active directory
            self.load_directories()
            self.load_active_directory()

            QMessageBox.information(
                self,
                "Success",
                f"Successfully removed directory from index:\n{directory_path}",
            )
        except DirectoryNotFoundError as e:
            QMessageBox.critical(self, "Error", f"Directory not found:\n{e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{e}")

    def open_app_directory(self):
        """Open the app directory in the system file manager."""
        try:
            app_data_dir = get_app_data_dir()

            # Open directory based on platform
            if platform.system() == "Windows":
                subprocess.run(["explorer", str(app_data_dir)])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(app_data_dir)])
            else:  # Linux and other Unix-like systems
                subprocess.run(["xdg-open", str(app_data_dir)])

        except Exception as e:
            QMessageBox.warning(
                self, "Error", f"Failed to open database directory: {str(e)}"
            )

    def load_models_info(self):
        """Load and display information about available models."""
        # Update models directory label
        self.models_dir_label.setText(str(self.model_manager.models_dir))

        # Calculate and display storage usage (only downloaded models)
        try:
            total_size = self.model_manager.get_models_directory_size()
            size_mb = total_size / (1024 * 1024)
            self.storage_label.setText(f"Downloaded models storage: {size_mb:.2f} MB")
        except Exception as e:
            self.storage_label.setText(f"Downloaded models storage: Error - {str(e)}")

        # Get all models info
        models_info = self.model_manager.get_all_models_info()

        # Disable sorting while populating
        self.models_table.setSortingEnabled(False)

        # Populate table
        self.models_table.setRowCount(len(models_info))

        for row, model_info in enumerate(models_info):
            # Model name
            name_item = QTableWidgetItem(model_info["name"])
            self.models_table.setItem(row, 0, name_item)

            # Language
            lang_item = QTableWidgetItem(model_info["language"])
            self.models_table.setItem(row, 1, lang_item)

            # Version
            version_item = QTableWidgetItem(model_info["version"])
            self.models_table.setItem(row, 2, version_item)

            # Size
            size_item = QTableWidgetItem(f"{model_info['size_mb']:.1f}")
            size_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.models_table.setItem(row, 3, size_item)

            # Status with location
            location = model_info.get("location", "")
            if location:
                status = f"✓ Available ({location})"
                status_item = QTableWidgetItem(status)
                status_item.setForeground(Qt.GlobalColor.darkGreen)
            else:
                status_item = QTableWidgetItem("Not installed")
            self.models_table.setItem(row, 4, status_item)

            # Description
            desc_item = QTableWidgetItem(model_info["description"])
            self.models_table.setItem(row, 5, desc_item)

        # Re-enable sorting after populating
        self.models_table.setSortingEnabled(True)

    def download_selected_model(self):
        """Download the selected model."""
        selected_row = self.models_table.currentRow()
        if selected_row < 0:
            QMessageBox.warning(
                self, "No Selection", "Please select a model to download."
            )
            return

        model_name = self.models_table.item(selected_row, 0).text()

        # Check if already in downloads directory
        if self.model_manager.is_model_in_downloads(model_name):
            QMessageBox.information(
                self,
                "Already Downloaded",
                f"Model '{model_name}' is already in the downloads directory.",
            )
            return

        # Check if available from other sources
        if self.model_manager.is_model_installed(model_name):
            model_info = self.model_manager.get_model_info(model_name)
            location = (
                model_info.get("location", "unknown") if model_info else "unknown"
            )

            reply = QMessageBox.question(
                self,
                "Model Already Available",
                f"Model '{model_name}' is already available from: {location}\n\n"
                f"Do you still want to download it to the downloads directory?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.No:
                return

        # Get model info for confirmation
        model_info = self.model_manager.get_model_info(model_name)
        reply = QMessageBox.question(
            self,
            "Confirm Download",
            f"Download model '{model_name}'?\n\n"
            f"Language: {model_info['language']}\n"
            f"Version: {model_info['version']}\n"
            f"Size: ~{model_info['size_mb']} MB",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.No:
            return

        # Show progress widgets
        self.model_progress_label.setVisible(True)
        self.model_progress_bar.setVisible(True)
        self.model_progress_bar.setRange(0, 100)

        # Start download in worker thread
        self.model_download_worker = ModelDownloadWorker(self.model_manager, model_name)
        self.model_download_worker.progress.connect(self.on_model_download_progress)
        self.model_download_worker.finished.connect(self.on_model_download_finished)
        self.model_download_worker.error.connect(self.on_model_download_error)
        self.model_download_worker.start()

    def on_model_download_progress(self, downloaded: int, total: int, message: str):
        """Handle model download progress updates."""
        self.model_progress_label.setText(message)

        if total > 0:
            progress = int((downloaded / total) * 100)
            self.model_progress_bar.setValue(progress)

    def on_model_download_finished(self, model_name: str, message: str):
        """Handle model download completion."""
        self.download_model_button.setEnabled(True)
        self.model_progress_label.setVisible(False)
        self.model_progress_bar.setVisible(False)

        QMessageBox.information(
            self,
            "Download Complete",
            message,
        )

        # Refresh models table
        self.load_models_info()

        # Refresh model combo box in index tab
        self.refresh_model_combo()

    def refresh_model_combo(self):
        """Refresh the model combo box with currently available models."""
        current_selection = self.model_combo.currentText()

        self.model_combo.clear()

        # Get available models from ModelManager
        available_models = [
            model_info["name"]
            for model_info in self.model_manager.get_all_models_info()
            if self.model_manager.is_model_available(model_info["name"])
        ]

        if available_models:
            self.model_combo.addItems(available_models)
            self.model_combo.setEnabled(True)

            # Try to restore previous selection if it's still available
            if current_selection in available_models:
                self.model_combo.setCurrentText(current_selection)
        else:
            self.model_combo.addItem("No models available")
            self.model_combo.setEnabled(False)

    def on_model_download_error(self, error_message: str):
        """Handle model download errors."""
        self.download_model_button.setEnabled(True)
        self.model_progress_label.setVisible(False)
        self.model_progress_bar.setVisible(False)

        QMessageBox.critical(
            self,
            "Download Error",
            f"An error occurred while downloading the model:\n{error_message}",
        )


def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    window = DoxearchGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
