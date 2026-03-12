from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from doxearch.doxearch import Doxearch
from doxearch_gui.utils.file_operations import open_file, open_folder
from doxearch_gui.workers.search_worker import SearchWorker


class SearchTab(QWidget):
    """Search tab for performing document searches."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.current_doxearch: Optional[Doxearch] = None
        self.search_worker: Optional[SearchWorker] = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

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

    def update_active_directory(
        self, directory_path: Optional[str], doxearch: Optional[Doxearch]
    ):
        """Update the active directory display."""
        if directory_path and doxearch:
            self.active_dir_label.setText(directory_path)
            self.current_doxearch = doxearch
        else:
            self.active_dir_label.setText("None")
            self.current_doxearch = None
            self.results_table.setRowCount(0)

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

    def on_search_error(self, error_message: str):
        """Handle search errors."""
        self.search_button.setEnabled(True)
        self.results_table.clear()

        QMessageBox.critical(
            self, "Search Error", f"An error occurred during search:\n{error_message}"
        )

    def open_result_file(self, row: int, column: int):
        """Open the file from the selected result row."""
        if self.results_table.rowCount() == 0:
            return

        filepath_item = self.results_table.item(row, 2)
        if filepath_item:
            filepath = filepath_item.text()
            open_file(filepath, self)

    def open_selected_file(self):
        """Open the currently selected file."""
        selected_row = self.results_table.currentRow()
        if selected_row < 0:
            QMessageBox.warning(self, "No Selection", "Please select a result to open.")
            return

        filepath_item = self.results_table.item(selected_row, 2)
        if filepath_item:
            filepath = filepath_item.text()
            open_file(filepath, self)

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
            open_folder(filepath, self)
