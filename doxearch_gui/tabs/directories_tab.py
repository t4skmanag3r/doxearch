from pathlib import Path

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class DirectoriesTab(QWidget):
    """Directories tab for managing indexed directories."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Info label
        info_label = QLabel(
            "Manage indexed directories. The active directory is used for searching."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(info_label)

        # Refresh button
        refresh_layout = QHBoxLayout()
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.load_directories)
        refresh_layout.addWidget(refresh_button)
        refresh_layout.addStretch()
        layout.addLayout(refresh_layout)

        # Directories table
        self.directories_table = QTableWidget()
        self.directories_table.setColumnCount(4)
        self.directories_table.setHorizontalHeaderLabels(
            ["Directory Path", "Model", "Lemmatization", "Stemming"]
        )
        self.directories_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.directories_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.directories_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.directories_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.directories_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.directories_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.directories_table)

        # Action buttons
        button_layout = QHBoxLayout()

        set_active_button = QPushButton("Set as Active")
        set_active_button.clicked.connect(self.set_active_directory)
        button_layout.addWidget(set_active_button)

        remove_button = QPushButton("Remove Directory")
        remove_button.clicked.connect(self.remove_directory)
        button_layout.addWidget(remove_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Load directories
        self.load_directories()

    def load_directories(self):
        """Load and display all indexed directories."""
        directories = self.parent.context_manager.get_all_directories()
        active_dir = self.parent.context_manager.get_active_directory()
        active_path = active_dir["directory_path"] if active_dir else None

        self.directories_table.setRowCount(len(directories))

        for row, directory in enumerate(directories):
            # Directory path
            path_item = QTableWidgetItem(directory["directory_path"])
            if directory["directory_path"] == active_path:
                path_item.setText(f"★ {directory['directory_path']}")
                # Highlight active directory
                from PyQt6.QtGui import QFont

                font = QFont()
                font.setBold(True)
                path_item.setFont(font)
            self.directories_table.setItem(row, 0, path_item)

            # Model
            model_item = QTableWidgetItem(directory["tokenizer_model_name"])
            self.directories_table.setItem(row, 1, model_item)

            # Lemmatization
            lemma_item = QTableWidgetItem(
                "✓" if directory.get("lemmatization_enabled", True) else "✗"
            )
            self.directories_table.setItem(row, 2, lemma_item)

            # Stemming
            stem_item = QTableWidgetItem(
                "✓" if directory.get("stemming_enabled", False) else "✗"
            )
            self.directories_table.setItem(row, 3, stem_item)

    def set_active_directory(self):
        """Set the selected directory as active."""
        selected_row = self.directories_table.currentRow()
        if selected_row < 0:
            QMessageBox.warning(
                self, "No Selection", "Please select a directory to set as active."
            )
            return

        path_item = self.directories_table.item(selected_row, 0)
        directory_path = path_item.text().replace("★ ", "")

        try:
            self.parent.context_manager.set_active_directory(directory_path)
            self.parent.refresh_all_tabs()
            QMessageBox.information(
                self,
                "Active Directory Set",
                f"Successfully set active directory to:\n{directory_path}",
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Setting Active Directory",
                f"Failed to set active directory:\n{str(e)}",
            )

    def remove_directory(self):
        """Remove the selected directory from the index."""
        selected_row = self.directories_table.currentRow()
        if selected_row < 0:
            QMessageBox.warning(
                self, "No Selection", "Please select a directory to remove."
            )
            return

        path_item = self.directories_table.item(selected_row, 0)
        directory_path = path_item.text().replace("★ ", "")

        # Get directory info to find the database path
        dir_info = self.parent.context_manager.get_directory_info(directory_path)
        if not dir_info:
            QMessageBox.warning(
                self,
                "Directory Not Found",
                f"Could not find directory information for:\n{directory_path}",
            )
            return

        db_path = Path(dir_info["db_path"])

        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to remove this directory?\n\n{directory_path}\n\n"
            "This will delete the index database but not the actual files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Remove from context manager
                self.parent.context_manager.remove_indexed_directory(directory_path)

                # Delete the database file if it exists
                if db_path.exists():
                    try:
                        db_path.unlink()
                    except Exception as e:
                        QMessageBox.warning(
                            self,
                            "Database Deletion Warning",
                            f"Directory removed from index, but failed to delete database file:\n{e}",
                        )

                self.parent.refresh_all_tabs()
                QMessageBox.information(
                    self,
                    "Directory Removed",
                    f"Successfully removed directory:\n{directory_path}",
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Removing Directory",
                    f"Failed to remove directory:\n{str(e)}",
                )
