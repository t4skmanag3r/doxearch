from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from doxearch.doxearch import Doxearch
from doxearch_gui.utils.file_operations import open_file, open_folder


class DocumentsTab(QWidget):
    """Documents tab showing all indexed documents."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.current_doxearch: Optional[Doxearch] = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

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

    def update_active_directory(
        self, directory_path: Optional[str], doxearch: Optional[Doxearch]
    ):
        """Update the active directory display."""
        if directory_path and doxearch:
            self.documents_active_dir_label.setText(directory_path)
            self.current_doxearch = doxearch
            self.load_documents()
        else:
            self.documents_active_dir_label.setText("None")
            self.current_doxearch = None
            self.document_count_label.setText("Total documents: 0")
            self.documents_table.setRowCount(0)

    def load_documents(self):
        """Load and display all documents from the active index."""
        if not self.current_doxearch:
            self.document_count_label.setText("Total documents: 0")
            self.documents_table.setRowCount(0)
            return

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
            open_file(filepath, self)

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
            open_file(filepath, self)

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
            open_folder(filepath, self)
