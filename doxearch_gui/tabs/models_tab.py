from typing import Optional

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from doxearch_gui.workers.model_download_worker import ModelDownloadWorker


class ModelsTab(QWidget):
    """Models tab for managing spaCy language models."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.model_download_worker: Optional[ModelDownloadWorker] = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

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

        # Enable sorting
        self.models_table.setSortingEnabled(True)

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

    def load_models_info(self):
        """Load and display information about available models."""
        # Temporarily disable sorting while populating
        self.models_table.setSortingEnabled(False)

        # Update models directory label
        self.models_dir_label.setText(str(self.parent.models_dir))

        # Get all models info
        models_info = self.parent.model_manager.get_all_models_info()

        self.models_table.setRowCount(len(models_info))

        total_size = 0

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

            # Size (with numeric sorting)
            size_mb = model_info["size_mb"]
            size_item = QTableWidgetItem()
            size_item.setData(0, size_mb)  # Store numeric value for proper sorting
            size_item.setText(f"{size_mb:.1f}")
            self.models_table.setItem(row, 3, size_item)

            # Status
            status = model_info["status"]
            status_item = QTableWidgetItem(status)
            self.models_table.setItem(row, 4, status_item)

            # Description
            desc_item = QTableWidgetItem(model_info["description"])
            self.models_table.setItem(row, 5, desc_item)

            # Add to total size if installed
            if status == "Installed":
                total_size += size_mb

        # Re-enable sorting after populating
        self.models_table.setSortingEnabled(True)

        # Update storage label
        self.storage_label.setText(f"Total storage used: {total_size:.1f} MB")

    def fetch_models(self):
        """Fetch available models from spaCy compatibility.json."""
        self.fetch_models_button.setEnabled(False)
        self.fetch_models_button.setText("Fetching...")

        try:
            # Fetch models
            self.parent.model_manager.MODEL_URLS = (
                self.parent.model_manager.fetch_available_models()
            )

            # Refresh the table with newly fetched models
            self.load_models_info()

            models_count = len(self.parent.model_manager.MODEL_URLS)
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
            self.fetch_models_button.setText("Fetch Models")

    def download_selected_model(self):
        """Download the selected model."""
        selected_row = self.models_table.currentRow()
        if selected_row < 0:
            QMessageBox.warning(
                self, "No Selection", "Please select a model to download."
            )
            return

        model_name_item = self.models_table.item(selected_row, 0)
        model_name = model_name_item.text()

        status_item = self.models_table.item(selected_row, 4)
        status = status_item.text()

        if status == "Installed":
            QMessageBox.information(
                self, "Already Installed", f"Model '{model_name}' is already installed."
            )
            return

        if status == "Not Available":
            QMessageBox.warning(
                self,
                "Model Not Available",
                f"Model '{model_name}' is not available for download.\n\n"
                "Try clicking 'Fetch Models' to update the available models list.",
            )
            return

        # Disable download button during download
        self.download_model_button.setEnabled(False)
        self.model_progress_label.setVisible(True)
        self.model_progress_label.setText(f"Downloading {model_name}...")
        self.model_progress_bar.setVisible(True)
        self.model_progress_bar.setValue(0)

        # Start download in worker thread
        self.model_download_worker = ModelDownloadWorker(
            self.parent.model_manager, model_name
        )
        self.model_download_worker.progress.connect(self.on_download_progress)
        self.model_download_worker.finished.connect(self.on_download_finished)
        self.model_download_worker.error.connect(self.on_download_error)
        self.model_download_worker.start()

    def on_download_progress(self, downloaded: int, total: int, message: str):
        """Handle download progress updates."""
        if total > 0:
            progress = int((downloaded / total) * 100)
            self.model_progress_bar.setValue(progress)
        self.model_progress_label.setText(message)

    def on_download_finished(self, model_name: str, message: str):
        """Handle download completion."""
        self.download_model_button.setEnabled(True)
        self.model_progress_label.setVisible(False)
        self.model_progress_bar.setVisible(False)

        # Refresh models info
        self.load_models_info()

        # Refresh all tabs to update model availability
        self.parent.refresh_all_tabs()

        QMessageBox.information(
            self, "Download Complete", f"Successfully downloaded model: {model_name}"
        )

    def on_download_error(self, error_message: str):
        """Handle download errors."""
        self.download_model_button.setEnabled(True)
        self.model_progress_label.setVisible(False)
        self.model_progress_bar.setVisible(False)

        QMessageBox.critical(
            self, "Download Error", f"Failed to download model:\n{error_message}"
        )
